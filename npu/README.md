# BitNet AMD NPU Acceleration

This directory contains the AMD NPU (Neural Processing Unit) acceleration support for BitNet inference. It enables offloading matrix-vector multiplication operations to AMD Ryzen AI NPUs.

## Overview

The NPU acceleration replaces the AVX2-based LUT kernels with AMD AIE (AI Engine) kernels that run on the NPU. This provides hardware acceleration for the ternary matrix-vector multiplications that are the core compute operations in BitNet models.

## Directory Structure

```
npu/
├── kernels/
│   └── mv_ternary.cc      # AIE kernel for ternary MV multiplication
├── bitnet_mv.py           # MLIR design generator for NPU
├── npu_runtime.h          # C API header for NPU runtime
├── npu_runtime.cpp        # NPU runtime implementation using XRT
├── Makefile               # Build system for NPU kernels
└── README.md              # This file
```

## Prerequisites

1. **AMD Ryzen AI NPU** - Laptop or desktop with AMD Ryzen AI processor
2. **XRT (Xilinx Runtime)** - AMD runtime for NPU access
   - Install from: https://www.xilinx.com/products/design-tools/vitis/xrt.html
   - Or use AMD's NPU driver package
3. **MLIR-AIE Toolchain** (for building kernels from source)
   - Required only if you need to rebuild the xclbin
   - Pre-built xclbin files can be used directly

## Quick Start

### Using Pre-built NPU Acceleration

1. **Install XRT**:
   ```bash
   # Ubuntu/Debian
   sudo apt install xrt
   
   # Or download from AMD website
   ```

2. **Build BitNet with NPU support**:
   ```bash
   cd BitNet-2Fast2Furious2Bits
   python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-3B --use-npu
   ```

3. **Run inference**:
   ```bash
   python run_inference.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf \
       -p "Your prompt here"
   ```

### Building NPU Kernels from Source

If you need to rebuild the NPU kernels (e.g., for different tile sizes or custom optimizations):

1. **Install MLIR-AIE toolchain**:
   ```bash
   # Follow MLIR-AIE installation instructions
   # https://github.com/Xilinx/mlir-aie
   ```

2. **Build the kernels**:
   ```bash
   cd npu
   make DEVICE=npu MODEL=bitnet_b1_58-3B
   ```

3. **Available make targets**:
   ```bash
   make help          # Show all options
   make kernels       # Compile AIE kernels only
   make mlir          # Generate MLIR design
   make xclbin        # Generate complete xclbin
   make clean         # Remove build artifacts
   ```

## Configuration

### Model Configurations

The NPU supports the following BitNet models:

| Model | M Dimensions | K Dimensions |
|-------|--------------|--------------|
| bitnet_b1_58-3B | 3200, 3200, 8640 | 8640, 3200, 3200 |
| bitnet_b1_58-large | 1536, 1536, 4096 | 4096, 1536, 1536 |
| Llama3-8B-1.58 | 14336, 4096, 1024, 4096 | 4096, 14336, 4096, 4096 |

### Tile Sizes

Default tile sizes are optimized for common configurations:
- `M_TILE=32` - Output tile size
- `K_TILE=32` - Input tile size

Adjust these based on your NPU's memory constraints and the model dimensions.

## API Reference

### C API (npu_runtime.h)

```c
// Initialize NPU with xclbin
npu_error_t npu_init(const char* xclbin_path);

// Execute BitNet matrix-vector multiplication
npu_error_t npu_bitnet_matvec(
    const npu_matmul_config_t* config,
    const void* weights_3bit,
    const void* weights_sign,
    const void* weights_2bit,
    const void* activations,
    void* output
);

// Cleanup
void npu_shutdown(void);
```

### Integration with BitNet

The NPU kernels maintain the same interface as the CPU kernels:
- `ggml_qgemm_lut()` - Main GEMM entry point
- `ggml_preprocessor()` - Activation preprocessing
- `ggml_bitnet_transform_tensor()` - Tensor transformation

This allows seamless switching between CPU and NPU execution.

## Performance

Expected performance improvements depend on:
- Model size and layer dimensions
- NPU generation (Ryzen AI 300 series vs earlier)
- Memory bandwidth utilization

Typical speedups range from 2-5x for matrix-vector operations compared to AVX2 CPU implementations.

## Troubleshooting

### Common Issues

1. **"NPU device not found"**
   - Ensure XRT is installed correctly
   - Check that the NPU driver is loaded: `lsmod | grep amdnpu`
   - Verify device permissions: `ls -la /dev/dri/`

2. **"Failed to load xclbin"**
   - Ensure the xclbin file matches your NPU hardware
   - Check file permissions
   - Verify XRT version compatibility

3. **"Kernel execution failed"**
   - Check that matrix dimensions are supported
   - Verify input data alignment (64-byte aligned)
   - Check XRT logs: `cat /var/log/syslog | grep xrt`

### Debug Mode

Enable verbose logging:
```c
npu_init_config_t config = {
    .xclbin_path = "bitnet_mv.xclbin",
    .verbosity = 2  // 0=off, 1=info, 2=debug
};
npu_init_with_config(&config);
```

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional NPU generations
- Kernel optimizations for specific model sizes
- Multi-core parallelization
- Mixed-precision support

## License

This code is licensed under the Apache License v2.0 with LLVM Exceptions.
See the LICENSE file for details.
