#
# BitNet Matrix-Vector MLIR Design for AMD NPU
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified for BitNet ternary weight support.
#

import numpy as np
import argparse
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_


def bitnet_matvec(M, K, K3, K2, m_tile, k_tile, n_cores, dev_type):
    """
    Generate MLIR design for BitNet matrix-vector multiplication on AMD NPU.
    
    Parameters:
        M: Output dimension (number of rows in weight matrix)
        K: Input dimension (K = K3 + K2)
        K3: Size of 3-bit packed portion (must be divisible by 3)
        K2: Size of 2-bit packed portion (must be divisible by 2)
        m_tile: Tile size for M dimension
        k_tile: Tile size for K dimension
        n_cores: Number of AIE cores to use
        dev_type: Device type ("npu" or "npu2")
    """
    
    # Validate dimensions
    assert K == K3 + K2, f"K ({K}) must equal K3 ({K3}) + K2 ({K2})"
    assert K3 % 3 == 0 or K3 == 0, f"K3 ({K3}) must be divisible by 3"
    assert K2 % 2 == 0 or K2 == 0, f"K2 ({K2}) must be divisible by 2"
    assert M % m_tile == 0, f"M ({M}) must be divisible by m_tile ({m_tile})"
    
    # Calculate packed sizes
    # 3-bit: 3 ternary values per nibble, 2 nibbles per byte
    packed_3bit_size = (K3 // 3 + 1) // 2 if K3 > 0 else 0
    sign_size = (K3 // 3 + 7) // 8 if K3 > 0 else 0
    # 2-bit: 2 ternary values per nibble, 2 nibbles per byte  
    packed_2bit_size = (K2 // 2 + 1) // 2 if K2 > 0 else 0
    
    # Total sizes
    A_3bit_sz = M * packed_3bit_size
    A_sign_sz = M * sign_size
    A_2bit_sz = M * packed_2bit_size
    B_sz = K
    C_sz = M
    
    # Tile calculations
    M_div_m = M // m_tile
    M_div_m_div_cores = M_div_m // n_cores
    K_div_k = K // k_tile
    
    # Data types
    dtype_packed = np.dtype[np.uint8]
    dtype_act = np.dtype[np.int16]  # Quantized activations
    dtype_out = np.dtype[np.int32]  # Accumulator output
    
    with mlir_mod_ctx() as ctx:
        if dev_type == "npu":
            device_ty = AIEDevice.npu1
        else:
            device_ty = AIEDevice.npu2
            
        @device(device_ty)
        def device_body():
            # Tile-level types
            tile_3bit_ty = np.ndarray[(m_tile * packed_3bit_size,), dtype_packed]
            tile_sign_ty = np.ndarray[(m_tile * sign_size,), dtype_packed]
            tile_2bit_ty = np.ndarray[(m_tile * packed_2bit_size,), dtype_packed]
            tile_act_ty = np.ndarray[(k_tile,), dtype_act]
            tile_out_ty = np.ndarray[(m_tile,), dtype_out]
            
            # External kernel functions
            zero_fn = external_func(
                "zero_scalar_i32",
                inputs=[tile_out_ty]
            )
            
            matvec_3bit_fn = external_func(
                "matvec_ternary_3bit_i16_i32",
                inputs=[tile_3bit_ty, tile_sign_ty, tile_act_ty, tile_out_ty]
            ) if K3 > 0 else None
            
            matvec_2bit_fn = external_func(
                "matvec_ternary_2bit_i16_i32",
                inputs=[tile_2bit_ty, tile_act_ty, tile_out_ty]
            ) if K2 > 0 else None
            
            # Tile declarations
            shim_tiles = [tile(i, 0) for i in range(4)]
            mem_tiles = [tile(i, 1) for i in range(4)]
            compute_tiles = [tile(i, 2) for i in range(4)]
            
            # Object FIFOs for data movement
            # Input: Packed weights (3-bit portion)
            mem_3bit_fifos = []
            in_3bit_fifos = []
            # Input: Sign bits
            mem_sign_fifos = []
            in_sign_fifos = []
            # Input: Packed weights (2-bit portion)
            mem_2bit_fifos = []
            in_2bit_fifos = []
            # Output
            out_fifos = []
            
            for i in range(n_cores):
                if K3 > 0:
                    # 3-bit weight FIFOs
                    mem_3bit_fifos.append(
                        object_fifo(f"mem_3bit_{i}", shim_tiles[i], mem_tiles[i], 2, tile_3bit_ty)
                    )
                    in_3bit_fifos.append(
                        object_fifo(f"in_3bit_{i}", mem_tiles[i], compute_tiles[i], 2, tile_3bit_ty)
                    )
                    object_fifo_link(mem_3bit_fifos[i], in_3bit_fifos[i])
                    
                    # Sign FIFOs
                    mem_sign_fifos.append(
                        object_fifo(f"mem_sign_{i}", shim_tiles[i], mem_tiles[i], 2, tile_sign_ty)
                    )
                    in_sign_fifos.append(
                        object_fifo(f"in_sign_{i}", mem_tiles[i], compute_tiles[i], 2, tile_sign_ty)
                    )
                    object_fifo_link(mem_sign_fifos[i], in_sign_fifos[i])
                
                if K2 > 0:
                    # 2-bit weight FIFOs
                    mem_2bit_fifos.append(
                        object_fifo(f"mem_2bit_{i}", shim_tiles[i], mem_tiles[i], 2, tile_2bit_ty)
                    )
                    in_2bit_fifos.append(
                        object_fifo(f"in_2bit_{i}", mem_tiles[i], compute_tiles[i], 2, tile_2bit_ty)
                    )
                    object_fifo_link(mem_2bit_fifos[i], in_2bit_fifos[i])
                
                # Output FIFOs
                out_fifos.append(
                    object_fifo(f"out_{i}", compute_tiles[i], shim_tiles[i], 2, tile_out_ty)
                )
            
            # Activation input FIFO (broadcast to all cores)
            in_act_fifo = object_fifo(
                "in_act", shim_tiles[0], compute_tiles[0:n_cores], 2, tile_act_ty
            )
            
            # Compute tile kernels
            for i in range(n_cores):
                @core(compute_tiles[i], f"mv_ternary_{m_tile}x{k_tile}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        # Acquire output buffer and zero it
                        elem_out = out_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        zero_fn(elem_out)
                        
                        # Process K dimension in tiles
                        k3_tiles = K3 // k_tile if K3 > 0 else 0
                        k2_tiles = K2 // k_tile if K2 > 0 else 0
                        
                        # Process 3-bit portion
                        if K3 > 0:
                            for _ in range_(k3_tiles):
                                elem_3bit = in_3bit_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                                elem_sign = in_sign_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                                elem_act = in_act_fifo.acquire(ObjectFifoPort.Consume, 1)
                                
                                matvec_3bit_fn(elem_3bit, elem_sign, elem_act, elem_out)
                                
                                in_3bit_fifos[i].release(ObjectFifoPort.Consume, 1)
                                in_sign_fifos[i].release(ObjectFifoPort.Consume, 1)
                                in_act_fifo.release(ObjectFifoPort.Consume, 1)
                        
                        # Process 2-bit portion
                        if K2 > 0:
                            for _ in range_(k2_tiles):
                                elem_2bit = in_2bit_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                                elem_act = in_act_fifo.acquire(ObjectFifoPort.Consume, 1)
                                
                                matvec_2bit_fn(elem_2bit, elem_act, elem_out)
                                
                                in_2bit_fifos[i].release(ObjectFifoPort.Consume, 1)
                                in_act_fifo.release(ObjectFifoPort.Consume, 1)
                        
                        # Release output
                        out_fifos[i].release(ObjectFifoPort.Produce, 1)
            
            # Runtime sequence for data movement
            @runtime_sequence(
                np.ndarray[(A_3bit_sz,), dtype_packed],  # Packed 3-bit weights
                np.ndarray[(A_sign_sz,), dtype_packed],  # Sign bits
                np.ndarray[(A_2bit_sz,), dtype_packed],  # Packed 2-bit weights
                np.ndarray[(B_sz,), dtype_act],          # Activations
                np.ndarray[(C_sz,), dtype_out],          # Output
            )
            def sequence(A_3bit, A_sign, A_2bit, B, C):
                # Transfer activations (broadcast)
                npu_dma_memcpy_nd(
                    metadata=in_act_fifo,
                    bd_id=0,
                    mem=B,
                    sizes=[M_div_m_div_cores, K_div_k, 1, k_tile],
                    strides=[0, k_tile, 0, 1],
                )
                
                for i in range(n_cores):
                    row_offset = i * M_div_m_div_cores * m_tile
                    
                    if K3 > 0:
                        # Transfer 3-bit weights
                        npu_dma_memcpy_nd(
                            metadata=mem_3bit_fifos[i],
                            bd_id=1,
                            mem=A_3bit,
                            offsets=[0, 0, 0, row_offset * packed_3bit_size],
                            sizes=[M_div_m_div_cores, 1, 1, m_tile * packed_3bit_size],
                            strides=[m_tile * packed_3bit_size, 0, 0, 1],
                        )
                        
                        # Transfer sign bits
                        npu_dma_memcpy_nd(
                            metadata=mem_sign_fifos[i],
                            bd_id=2,
                            mem=A_sign,
                            offsets=[0, 0, 0, row_offset * sign_size],
                            sizes=[M_div_m_div_cores, 1, 1, m_tile * sign_size],
                            strides=[m_tile * sign_size, 0, 0, 1],
                        )
                    
                    if K2 > 0:
                        # Transfer 2-bit weights
                        npu_dma_memcpy_nd(
                            metadata=mem_2bit_fifos[i],
                            bd_id=3,
                            mem=A_2bit,
                            offsets=[0, 0, 0, row_offset * packed_2bit_size],
                            sizes=[M_div_m_div_cores, 1, 1, m_tile * packed_2bit_size],
                            strides=[m_tile * packed_2bit_size, 0, 0, 1],
                        )
                    
                    # Transfer output
                    npu_dma_memcpy_nd(
                        metadata=out_fifos[i],
                        bd_id=4,
                        mem=C,
                        offsets=[0, 0, 0, row_offset],
                        sizes=[1, 1, 1, M_div_m_div_cores * m_tile],
                        strides=[0, 0, 0, 1],
                    )
                
                dma_wait(*out_fifos)
        
        print(ctx.module)


# BitNet model configurations
BITNET_CONFIGS = {
    "bitnet_b1_58-3B": [
        {"M": 3200, "K": 8640, "K3": 8640, "K2": 0},
        {"M": 3200, "K": 3200, "K3": 3168, "K2": 32},
        {"M": 8640, "K": 3200, "K3": 3168, "K2": 32},
    ],
    "bitnet_b1_58-large": [
        {"M": 1536, "K": 4096, "K3": 4096, "K2": 0},
        {"M": 1536, "K": 1536, "K3": 1536, "K2": 0},
        {"M": 4096, "K": 1536, "K3": 1536, "K2": 0},
    ],
    "Llama3-8B-1.58-100B-tokens": [
        {"M": 14336, "K": 4096, "K3": 4096, "K2": 0},
        {"M": 4096, "K": 14336, "K3": 14336, "K2": 0},
        {"M": 1024, "K": 4096, "K3": 4096, "K2": 0},
        {"M": 4096, "K": 4096, "K3": 4096, "K2": 0},
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BitNet Matrix-Vector MLIR Design for AMD NPU",
        description="Generate MLIR design for BitNet inference on AMD NPU"
    )
    parser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu",
                        help="Target NPU device")
    parser.add_argument("--model", type=str, choices=list(BITNET_CONFIGS.keys()),
                        default="bitnet_b1_58-3B", help="BitNet model configuration")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index in model configuration")
    parser.add_argument("--m-tile", type=int, default=32,
                        help="Tile size for M dimension")
    parser.add_argument("--k-tile", type=int, default=32,
                        help="Tile size for K dimension")
    parser.add_argument("--n-cores", type=int, default=1,
                        help="Number of AIE cores to use")
    parser.add_argument("--M", type=int, default=None,
                        help="Override M dimension")
    parser.add_argument("--K", type=int, default=None,
                        help="Override K dimension")
    parser.add_argument("--K3", type=int, default=None,
                        help="Override K3 (3-bit portion) dimension")
    parser.add_argument("--K2", type=int, default=None,
                        help="Override K2 (2-bit portion) dimension")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.M is not None and args.K is not None:
        # Use override dimensions
        config = {
            "M": args.M,
            "K": args.K,
            "K3": args.K3 if args.K3 is not None else args.K,
            "K2": args.K2 if args.K2 is not None else 0,
        }
    else:
        # Use model configuration
        configs = BITNET_CONFIGS[args.model]
        if args.layer >= len(configs):
            print(f"Error: Layer {args.layer} not found in {args.model} (max: {len(configs)-1})")
            sys.exit(1)
        config = configs[args.layer]
    
    bitnet_matvec(
        M=config["M"],
        K=config["K"],
        K3=config["K3"],
        K2=config["K2"],
        m_tile=args.m_tile,
        k_tile=args.k_tile,
        n_cores=args.n_cores,
        dev_type=args.dev
    )
