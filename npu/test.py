#!/usr/bin/env python3
#
# BitNet NPU Test - Python Version
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified for BitNet NPU acceleration testing.
#

import argparse
import numpy as np
import sys

# Try to import XRT Python bindings
try:
    import pyxrt
    HAS_XRT = True
except ImportError:
    HAS_XRT = False
    print("Warning: pyxrt not found, using simulation mode")


def random_ternary(shape):
    """Generate random ternary values (-1, 0, +1)"""
    return np.random.randint(-1, 2, size=shape, dtype=np.int8)


def pack_ternary_2bit(values):
    """Pack ternary values into nibbles for 2-bit encoding
    Encoding: -1 -> 0, 0 -> 1, +1 -> 2
    """
    flat = values.flatten()
    # Ensure even length
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    
    # Convert to 0,1,2 encoding
    encoded = (flat + 1).astype(np.uint8)
    
    # Pack pairs into bytes
    packed = (encoded[1::2] << 4) | encoded[0::2]
    return packed.astype(np.uint8)


def cpu_matvec_ternary(weights, input_vec):
    """Reference CPU implementation of ternary matrix-vector multiplication"""
    return np.dot(weights.astype(np.int32), input_vec.astype(np.int32))


def run_npu_test(xclbin_path, insts_path, kernel_name, M, K, verbosity=0):
    """Run the NPU test with XRT"""
    
    print(f"BitNet NPU Test (Python)")
    print(f"  Matrix size: {M} x {K}")
    print(f"  xclbin: {xclbin_path}")
    
    # Generate test data
    np.random.seed(42)
    
    weights = random_ternary((M, K))
    input_vec = np.random.randint(-128, 128, size=K, dtype=np.int16)
    
    # CPU reference
    output_cpu = cpu_matvec_ternary(weights, input_vec)
    
    # Pack weights for NPU
    weights_packed = pack_ternary_2bit(weights)
    
    if verbosity >= 2:
        print(f"  Weights shape: {weights.shape}")
        print(f"  Packed weights size: {len(weights_packed)} bytes")
    
    if not HAS_XRT:
        print("  Skipping NPU execution (XRT not available)")
        print(f"  CPU reference output (first 8): {output_cpu[:8]}")
        return True
    
    try:
        # Load instructions
        with open(insts_path, 'rb') as f:
            instr_data = np.frombuffer(f.read(), dtype=np.uint32)
        
        if verbosity >= 1:
            print(f"  Instructions: {len(instr_data)} words")
        
        # Initialize XRT
        device = pyxrt.device(0)
        xclbin = pyxrt.xclbin(xclbin_path)
        device.register_xclbin(xclbin)
        
        context = pyxrt.hw_context(device, xclbin.get_uuid())
        
        # Find kernel
        kernels = xclbin.get_kernels()
        kernel_info = None
        for k in kernels:
            if k.get_name().startswith(kernel_name):
                kernel_info = k
                break
        
        if kernel_info is None:
            raise RuntimeError(f"Kernel {kernel_name} not found")
        
        kernel = pyxrt.kernel(context, kernel_info.get_name())
        
        if verbosity >= 1:
            print(f"  Kernel: {kernel_info.get_name()}")
        
        # Allocate buffers
        bo_instr = pyxrt.bo(device, len(instr_data) * 4, 
                           pyxrt.bo.flags.cacheable, kernel.group_id(1))
        bo_weights = pyxrt.bo(device, len(weights_packed),
                             pyxrt.bo.flags.host_only, kernel.group_id(3))
        bo_input = pyxrt.bo(device, K * 2,  # int16
                           pyxrt.bo.flags.host_only, kernel.group_id(4))
        bo_output = pyxrt.bo(device, M * 4,  # int32
                            pyxrt.bo.flags.host_only, kernel.group_id(5))
        
        # Copy data to device
        bo_instr.write(instr_data.tobytes())
        bo_weights.write(weights_packed.tobytes())
        bo_input.write(input_vec.tobytes())
        bo_output.write(np.zeros(M, dtype=np.int32).tobytes())
        
        bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_weights.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_input.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bo_output.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        
        # Execute kernel
        if verbosity >= 1:
            print("  Executing kernel...")
        
        run = kernel(bo_instr, len(instr_data), bo_weights, bo_input, bo_output)
        run.wait()
        
        # Read results
        bo_output.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        output_npu = np.frombuffer(bo_output.read(M * 4), dtype=np.int32)
        
        # Verify results
        errors = np.sum(output_cpu != output_npu)
        
        if verbosity >= 2:
            print(f"  CPU output (first 8): {output_cpu[:8]}")
            print(f"  NPU output (first 8): {output_npu[:8]}")
        
        if errors == 0:
            print(f"PASS: All {M} outputs match!")
            return True
        else:
            print(f"FAIL: {errors} / {M} mismatches")
            if verbosity >= 1:
                mismatch_idx = np.where(output_cpu != output_npu)[0]
                for idx in mismatch_idx[:10]:
                    print(f"  [{idx}]: CPU={output_cpu[idx]} NPU={output_npu[idx]}")
            return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BitNet NPU Test")
    parser.add_argument("-x", "--xclbin", default="build/final.xclbin",
                        help="Path to xclbin file")
    parser.add_argument("-i", "--insts", default="build/insts.txt",
                        help="Path to instructions file")
    parser.add_argument("-k", "--kernel", default="MLIR_AIE",
                        help="Kernel name prefix")
    parser.add_argument("-m", "--rows", type=int, default=32,
                        help="Number of rows (M)")
    parser.add_argument("-n", "--cols", type=int, default=32,
                        help="Number of columns (K)")
    parser.add_argument("-s", "--size", type=int, default=None,
                        help="Square matrix size (overrides -m and -n)")
    parser.add_argument("-t", "--trace", type=int, default=0,
                        help="Trace buffer size")
    parser.add_argument("-v", "--verbosity", type=int, default=0,
                        help="Verbosity level")
    
    args = parser.parse_args()
    
    M = args.size if args.size else args.rows
    K = args.size if args.size else args.cols
    
    success = run_npu_test(
        args.xclbin,
        args.insts,
        args.kernel,
        M, K,
        args.verbosity
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
