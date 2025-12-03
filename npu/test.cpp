//===- test.cpp - BitNet NPU Test ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// Modified for BitNet NPU acceleration testing.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

// Test configuration
#ifndef TEST_M
#define TEST_M 32
#endif

#ifndef TEST_K
#define TEST_K 32
#endif

// Generate random ternary values (-1, 0, +1)
int8_t random_ternary() {
    int r = rand() % 3;
    return (r == 0) ? -1 : (r == 1) ? 0 : 1;
}

// Pack ternary values into nibbles for 2-bit encoding
void pack_ternary_2bit(const int8_t* values, uint8_t* packed, int count) {
    // Encoding: -1 -> 0, 0 -> 1, +1 -> 2
    for (int i = 0; i < count / 2; i++) {
        uint8_t v0 = (values[i * 2] + 1) & 0x03;
        uint8_t v1 = (values[i * 2 + 1] + 1) & 0x03;
        packed[i] = (v1 << 4) | v0;
    }
}

// Reference CPU implementation
void cpu_matvec_ternary(const int8_t* weights, const int16_t* input,
                        int32_t* output, int M, int K) {
    for (int m = 0; m < M; m++) {
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += weights[m * K + k] * input[k];
        }
        output[m] = sum;
    }
}

int main(int argc, const char* argv[]) {
    // Parse arguments
    test_utils::parse_options options;
    options.add_option("xclbin", 'x', "Path to xclbin file", "build/final.xclbin");
    options.add_option("insts", 'i', "Path to instructions file", "build/insts.txt");
    options.add_option("kernel", 'k', "Kernel name", "MLIR_AIE");
    options.add_option("trace", 't', "Trace buffer size", "0");
    options.add_option("verbosity", 'v', "Verbosity level", "0");
    
    if (!options.parse(argc, argv)) {
        return 1;
    }
    
    std::string xclbin_path = options.get_string("xclbin");
    std::string insts_path = options.get_string("insts");
    std::string kernel_name = options.get_string("kernel");
    int trace_size = options.get_int("trace");
    int verbosity = options.get_int("verbosity");
    
    // Test dimensions
    const int M = TEST_M;
    const int K = TEST_K;
    
    std::cout << "BitNet NPU Test" << std::endl;
    std::cout << "  Matrix size: " << M << " x " << K << std::endl;
    std::cout << "  xclbin: " << xclbin_path << std::endl;
    
    // Generate test data
    srand(42);
    
    std::vector<int8_t> weights(M * K);
    std::vector<int16_t> input(K);
    std::vector<int32_t> output_cpu(M);
    std::vector<int32_t> output_npu(M);
    
    // Random ternary weights
    for (int i = 0; i < M * K; i++) {
        weights[i] = random_ternary();
    }
    
    // Random input activations
    for (int i = 0; i < K; i++) {
        input[i] = (rand() % 256) - 128;
    }
    
    // CPU reference
    cpu_matvec_ternary(weights.data(), input.data(), output_cpu.data(), M, K);
    
    // Pack weights for NPU
    std::vector<uint8_t> weights_packed((M * K + 1) / 2);
    pack_ternary_2bit(weights.data(), weights_packed.data(), M * K);
    
    // Load instructions
    std::vector<uint32_t> instr_v = test_utils::load_instr_binary(insts_path);
    if (verbosity >= 1) {
        std::cout << "  Instructions: " << instr_v.size() << " words" << std::endl;
    }
    
    // Initialize XRT
    try {
        xrt::device device(0);
        
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        
        xrt::hw_context context(device, xclbin.get_uuid());
        
        auto xkernels = xclbin.get_kernels();
        auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
            [&kernel_name](xrt::xclbin::kernel& k) {
                return k.get_name().rfind(kernel_name, 0) == 0;
            });
        
        xrt::kernel kernel(context, xkernel.get_name());
        
        if (verbosity >= 1) {
            std::cout << "  Kernel: " << xkernel.get_name() << std::endl;
        }
        
        // Allocate buffers
        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                                XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_weights = xrt::bo(device, weights_packed.size(),
                                  XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_input = xrt::bo(device, K * sizeof(int16_t),
                                XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_output = xrt::bo(device, M * sizeof(int32_t),
                                 XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        
        // Copy data to device
        memcpy(bo_instr.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
        memcpy(bo_weights.map<void*>(), weights_packed.data(), weights_packed.size());
        memcpy(bo_input.map<void*>(), input.data(), K * sizeof(int16_t));
        memset(bo_output.map<void*>(), 0, M * sizeof(int32_t));
        
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Execute kernel
        if (verbosity >= 1) {
            std::cout << "  Executing kernel..." << std::endl;
        }
        
        auto run = kernel(bo_instr, instr_v.size(), bo_weights, bo_input, bo_output);
        run.wait();
        
        // Read results
        bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(output_npu.data(), bo_output.map<int32_t*>(), M * sizeof(int32_t));
        
        // Verify results
        int errors = 0;
        for (int i = 0; i < M; i++) {
            if (output_cpu[i] != output_npu[i]) {
                errors++;
                if (verbosity >= 2 && errors <= 10) {
                    std::cout << "  Mismatch at " << i << ": CPU=" << output_cpu[i]
                              << " NPU=" << output_npu[i] << std::endl;
                }
            }
        }
        
        if (errors == 0) {
            std::cout << "PASS: All " << M << " outputs match!" << std::endl;
            return 0;
        } else {
            std::cout << "FAIL: " << errors << " / " << M << " mismatches" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
