//===- mv_ternary.cc - Ternary Matrix-Vector for BitNet -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// Modified for BitNet ternary weight support.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

//===----------------------------------------------------------------------===//
// Ternary Weight Decoding Tables
//===----------------------------------------------------------------------===//
// BitNet uses ternary weights: -1, 0, +1
// In 3-bit packing mode: 3 ternary values are encoded in 4 bits (nibble)
// Encoding: nibble value maps to 3 ternary values via lookup table
// 
// The 16 possible nibble values (0-15) map to combinations of {-1, 0, +1}^3
// We store these as int8 for efficient processing

// Lookup table: nibble -> 3 ternary values (stored as 3 int8s)
// Based on BitNet's encoding scheme
alignas(32) static const int8_t TERNARY_DECODE_3BIT[16][4] = {
    { 0,  0,  0, 0},  // 0
    { 0,  0,  1, 0},  // 1
    { 0,  0, -1, 0},  // 2
    { 0,  1,  0, 0},  // 3
    { 0,  1,  1, 0},  // 4
    { 0,  1, -1, 0},  // 5
    { 0, -1,  0, 0},  // 6
    { 0, -1,  1, 0},  // 7
    { 0, -1, -1, 0},  // 8
    { 1,  0,  0, 0},  // 9
    { 1,  0,  1, 0},  // 10
    { 1,  0, -1, 0},  // 11
    { 1,  1,  0, 0},  // 12
    { 1,  1,  1, 0},  // 13 - all +1
    { 1,  1, -1, 0},  // 14
    {-1,  0,  0, 0},  // 15 - Note: sign handled separately in BitNet
};

// For 2-bit packing: 2 ternary values in 4 bits
alignas(32) static const int8_t TERNARY_DECODE_2BIT[16][2] = {
    {-1, -1},  // 0
    {-1,  0},  // 1
    {-1,  1},  // 2
    { 0, -1},  // 3
    { 0,  0},  // 4
    { 0,  1},  // 5
    { 1, -1},  // 6
    { 1,  0},  // 7
    { 1,  1},  // 8
    { 0,  0},  // 9-15 unused in standard encoding
    { 0,  0},
    { 0,  0},
    { 0,  0},
    { 0,  0},
    { 0,  0},
    { 0,  0},
};

//===----------------------------------------------------------------------===//
// Zero Initialization Kernels
//===----------------------------------------------------------------------===//

template <typename T, int M>
void zero_scalar(T *__restrict c) {
    for (int i = 0; i < M; i++) {
        c[i] = 0;
    }
}

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    T *__restrict c_end = c + M;
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
}

//===----------------------------------------------------------------------===//
// Scalar Ternary Matrix-Vector Multiplication
//===----------------------------------------------------------------------===//
// Computes C = A * b where A contains packed ternary weights
// 
// Parameters:
//   a_packed: Packed ternary weight matrix (M x K), nibble-packed
//   a_sign:   Sign bits for ternary values (M x K/8 bytes)
//   b:        Input activation vector (K elements)
//   c:        Output vector (M elements)
//   M, K:     Matrix dimensions

template <typename T_out, int M, int K>
void matvec_ternary_3bit_scalar(uint8_t *__restrict a_packed,
                                 uint8_t *__restrict a_sign,
                                 int16_t *__restrict b,
                                 T_out *__restrict c) {
    // In BitNet 3-bit mode:
    // - Each nibble (4 bits) encodes 3 ternary values
    // - Sign bits stored separately
    // - K must be divisible by 3 for proper alignment
    
    const int K3 = K / 3;  // Number of 3-value groups
    
    for (int row = 0; row < M; row++) {
        T_out sum = 0;
        
        for (int k3 = 0; k3 < K3; k3++) {
            // Get packed nibble (2 nibbles per byte)
            int byte_idx = k3 / 2;
            int nibble_idx = k3 % 2;
            uint8_t packed_byte = a_packed[row * (K3 + 1) / 2 + byte_idx];
            uint8_t nibble = (nibble_idx == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            
            // Get sign bit for this group
            int sign_byte_idx = k3 / 8;
            int sign_bit_idx = k3 % 8;
            uint8_t sign_byte = a_sign[row * (K3 + 7) / 8 + sign_byte_idx];
            int8_t sign = ((sign_byte >> sign_bit_idx) & 1) ? -1 : 1;
            
            // Decode 3 ternary values from nibble
            const int8_t *decoded = TERNARY_DECODE_3BIT[nibble];
            
            // Multiply-accumulate for 3 values
            int k_base = k3 * 3;
            for (int i = 0; i < 3 && (k_base + i) < K; i++) {
                int8_t weight = decoded[i] * sign;
                sum += weight * b[k_base + i];
            }
        }
        
        c[row] += sum;
    }
}

template <typename T_out, int M, int K>
void matvec_ternary_2bit_scalar(uint8_t *__restrict a_packed,
                                 int16_t *__restrict b,
                                 T_out *__restrict c) {
    // In BitNet 2-bit mode:
    // - Each nibble (4 bits) encodes 2 ternary values
    // - No separate sign bits needed
    
    const int K2 = K / 2;  // Number of 2-value groups
    
    for (int row = 0; row < M; row++) {
        T_out sum = 0;
        
        for (int k2 = 0; k2 < K2; k2++) {
            // Get packed nibble (2 nibbles per byte)
            int byte_idx = k2 / 2;
            int nibble_idx = k2 % 2;
            uint8_t packed_byte = a_packed[row * (K2 + 1) / 2 + byte_idx];
            uint8_t nibble = (nibble_idx == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            
            // Decode 2 ternary values from nibble
            const int8_t *decoded = TERNARY_DECODE_2BIT[nibble];
            
            // Multiply-accumulate for 2 values
            int k_base = k2 * 2;
            sum += decoded[0] * b[k_base];
            if (k_base + 1 < K) {
                sum += decoded[1] * b[k_base + 1];
            }
        }
        
        c[row] += sum;
    }
}

//===----------------------------------------------------------------------===//
// Vectorized Ternary Matrix-Vector Multiplication
//===----------------------------------------------------------------------===//
// Optimized version using AIE vector operations

template <typename T_out, typename T_acc, int M, int K, int r>
void matvec_ternary_3bit_vectorized(uint8_t *__restrict a_packed,
                                     uint8_t *__restrict a_sign,
                                     int16_t *__restrict b,
                                     T_out *__restrict c) {
    static_assert(M % r == 0, "M must be divisible by r");
    static_assert(K % 24 == 0, "K must be divisible by 24 for 3-bit vectorized");
    
    const int K3 = K / 3;
    
    // Process r rows at a time
    for (int row_base = 0; row_base < M; row_base += r) {
        aie::accum<T_acc, r> acc;
        acc = aie::zeros<T_acc, r>();
        
        for (int k3 = 0; k3 < K3; k3 += 8) {
            // Load 8 groups of 3 values = 24 activations
            aie::vector<int16_t, 24> b_vec;
            for (int i = 0; i < 24; i++) {
                b_vec[i] = b[k3 * 3 + i];
            }
            
            // Process each row in the r-row block
            for (int row_off = 0; row_off < r; row_off++) {
                int row = row_base + row_off;
                T_out row_sum = 0;
                
                // Process 8 nibbles (24 ternary values)
                for (int g = 0; g < 8; g++) {
                    int nibble_idx = k3 + g;
                    int byte_idx = nibble_idx / 2;
                    uint8_t packed_byte = a_packed[row * (K3 + 1) / 2 + byte_idx];
                    uint8_t nibble = (nibble_idx % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
                    
                    // Get sign
                    int sign_byte_idx = nibble_idx / 8;
                    int sign_bit_idx = nibble_idx % 8;
                    uint8_t sign_byte = a_sign[row * (K3 + 7) / 8 + sign_byte_idx];
                    int8_t sign = ((sign_byte >> sign_bit_idx) & 1) ? -1 : 1;
                    
                    const int8_t *decoded = TERNARY_DECODE_3BIT[nibble];
                    
                    for (int i = 0; i < 3; i++) {
                        int8_t weight = decoded[i] * sign;
                        row_sum += weight * b_vec[g * 3 + i];
                    }
                }
                
                // Accumulate
                acc = aie::add(acc, row_off, row_sum);
            }
        }
        
        // Store results
        aie::vector<T_out, r> c_vec = aie::load_v<r>(c + row_base);
        c_vec = aie::add(c_vec, acc.template to_vector<T_out>());
        aie::store_v(c + row_base, c_vec);
    }
}

//===----------------------------------------------------------------------===//
// Combined Kernel: Handles both 3-bit and 2-bit modes
//===----------------------------------------------------------------------===//
// This is the main kernel that BitNet will call
// Mode is determined by the k3_size parameter

template <typename T_out, int M, int K3, int K2>
void matvec_bitnet_scalar(uint8_t *__restrict a_3bit,
                          uint8_t *__restrict a_sign,
                          uint8_t *__restrict a_2bit,
                          int16_t *__restrict b,
                          T_out *__restrict c) {
    // Process 3-bit portion
    if (K3 > 0) {
        matvec_ternary_3bit_scalar<T_out, M, K3>(a_3bit, a_sign, b, c);
    }
    
    // Process 2-bit portion (remaining K2 values)
    if (K2 > 0) {
        matvec_ternary_2bit_scalar<T_out, M, K2>(a_2bit, b + K3, c);
    }
}

//===----------------------------------------------------------------------===//
// Exported C Functions for AIE Kernel
//===----------------------------------------------------------------------===//

extern "C" {

// Default dimensions - can be overridden at compile time
#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef DIM_K3
#define DIM_K3 0  // 3-bit portion size (must be divisible by 3)
#endif

#ifndef DIM_K2
#define DIM_K2 32  // 2-bit portion size (must be divisible by 2)
#endif

// Zero functions
void zero_scalar_i32(int32_t *c) {
    zero_scalar<int32_t, DIM_M>(c);
}

void zero_vectorized_i32(int32_t *c) {
    zero_vectorized<int32_t, DIM_M>(c);
}

void zero_scalar_f32(float *c) {
    zero_scalar<float, DIM_M>(c);
}

void zero_vectorized_f32(float *c) {
    zero_vectorized<float, DIM_M>(c);
}

// Ternary MV kernels - 3-bit mode
void matvec_ternary_3bit_i16_i32(uint8_t *a_packed, uint8_t *a_sign,
                                  int16_t *b, int32_t *c) {
    matvec_ternary_3bit_scalar<int32_t, DIM_M, DIM_K3>(a_packed, a_sign, b, c);
}

// Ternary MV kernels - 2-bit mode
void matvec_ternary_2bit_i16_i32(uint8_t *a_packed, int16_t *b, int32_t *c) {
    matvec_ternary_2bit_scalar<int32_t, DIM_M, DIM_K2>(a_packed, b, c);
}

// Combined BitNet kernel (handles both 3-bit and 2-bit portions)
void matvec_bitnet_i16_i32(uint8_t *a_3bit, uint8_t *a_sign,
                           uint8_t *a_2bit, int16_t *b, int32_t *c) {
    matvec_bitnet_scalar<int32_t, DIM_M, DIM_K3, DIM_K2>(
        a_3bit, a_sign, a_2bit, b, c);
}

// Simplified interface for LUT-preprocessed activations
// In this mode, activations are already in LUT form and we just do accumulation
void matvec_lut_i8_i32(uint8_t *a_indices, int8_t *lut, int32_t *c) {
    // a_indices: packed weight indices (4 bits each)
    // lut: precomputed activation lookup table (16 entries per group)
    // c: output accumulator
    
    for (int row = 0; row < DIM_M; row++) {
        int32_t sum = 0;
        for (int k = 0; k < DIM_K / 2; k++) {
            int byte_idx = k;
            uint8_t packed = a_indices[row * (DIM_K / 2) + byte_idx];
            
            // Extract two 4-bit indices
            uint8_t idx_lo = packed & 0x0F;
            uint8_t idx_hi = packed >> 4;
            
            // LUT lookup and accumulate
            sum += lut[k * 32 + idx_lo];
            sum += lut[k * 32 + 16 + idx_hi];
        }
        c[row] += sum;
    }
}

} // extern "C"
