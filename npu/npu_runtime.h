//===- npu_runtime.h - AMD NPU Runtime for BitNet ----------------*- C++ -*-===//
//
// XRT-based runtime wrapper for AMD NPU acceleration of BitNet inference.
//
//===----------------------------------------------------------------------===//

#ifndef BITNET_NPU_RUNTIME_H
#define BITNET_NPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Error Codes
//===----------------------------------------------------------------------===//

typedef enum {
    NPU_SUCCESS = 0,
    NPU_ERROR_NOT_INITIALIZED = -1,
    NPU_ERROR_DEVICE_NOT_FOUND = -2,
    NPU_ERROR_XCLBIN_LOAD_FAILED = -3,
    NPU_ERROR_KERNEL_NOT_FOUND = -4,
    NPU_ERROR_BUFFER_ALLOC_FAILED = -5,
    NPU_ERROR_EXECUTION_FAILED = -6,
    NPU_ERROR_INVALID_PARAMS = -7,
    NPU_ERROR_OUT_OF_MEMORY = -8,
} npu_error_t;

//===----------------------------------------------------------------------===//
// Configuration Structures
//===----------------------------------------------------------------------===//

typedef struct {
    int M;              // Output dimension
    int K;              // Input dimension
    int K3;             // 3-bit portion size
    int K2;             // 2-bit portion size
    int batch_size;     // Batch size
    float weight_scale; // Weight quantization scale
    float act_scale;    // Activation quantization scale
} npu_matmul_config_t;

typedef struct {
    const char* xclbin_path;    // Path to xclbin file
    int device_index;           // Device index (default: 0)
    int verbosity;              // Verbosity level (0-3)
    size_t max_buffer_size;     // Maximum buffer size in bytes
} npu_init_config_t;

//===----------------------------------------------------------------------===//
// Initialization and Shutdown
//===----------------------------------------------------------------------===//

/**
 * Initialize the NPU runtime with default configuration.
 * @param xclbin_path Path to the compiled xclbin file
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_init(const char* xclbin_path);

/**
 * Initialize the NPU runtime with custom configuration.
 * @param config Initialization configuration
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_init_with_config(const npu_init_config_t* config);

/**
 * Shutdown the NPU runtime and release all resources.
 */
void npu_shutdown(void);

/**
 * Check if NPU is initialized and ready.
 * @return true if initialized, false otherwise
 */
bool npu_is_ready(void);

/**
 * Get the last error message.
 * @return Error message string (valid until next NPU call)
 */
const char* npu_get_error_message(void);

//===----------------------------------------------------------------------===//
// Buffer Management
//===----------------------------------------------------------------------===//

typedef void* npu_buffer_t;

/**
 * Allocate a device buffer.
 * @param size Size in bytes
 * @param host_accessible If true, buffer is accessible from host
 * @return Buffer handle or NULL on failure
 */
npu_buffer_t npu_buffer_alloc(size_t size, bool host_accessible);

/**
 * Free a device buffer.
 * @param buffer Buffer handle
 */
void npu_buffer_free(npu_buffer_t buffer);

/**
 * Copy data from host to device buffer.
 * @param buffer Device buffer handle
 * @param src Host source pointer
 * @param size Size in bytes
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_buffer_write(npu_buffer_t buffer, const void* src, size_t size);

/**
 * Copy data from device buffer to host.
 * @param buffer Device buffer handle
 * @param dst Host destination pointer
 * @param size Size in bytes
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_buffer_read(npu_buffer_t buffer, void* dst, size_t size);

/**
 * Get host pointer to device buffer (for host-accessible buffers).
 * @param buffer Device buffer handle
 * @return Host pointer or NULL if not accessible
 */
void* npu_buffer_map(npu_buffer_t buffer);

/**
 * Synchronize buffer to device.
 * @param buffer Device buffer handle
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_buffer_sync_to_device(npu_buffer_t buffer);

/**
 * Synchronize buffer from device.
 * @param buffer Device buffer handle
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_buffer_sync_from_device(npu_buffer_t buffer);

//===----------------------------------------------------------------------===//
// BitNet Matrix-Vector Operations
//===----------------------------------------------------------------------===//

/**
 * Execute BitNet matrix-vector multiplication on NPU.
 * 
 * Computes: C = scale * (A_ternary @ B_quantized)
 * 
 * @param config Matrix multiplication configuration
 * @param weights_3bit Packed 3-bit ternary weights (M x K3/3 nibbles)
 * @param weights_sign Sign bits for 3-bit weights (M x K3/24 bytes)
 * @param weights_2bit Packed 2-bit ternary weights (M x K2/2 nibbles)
 * @param activations Quantized input activations (K elements, int16)
 * @param output Output vector (M elements, float)
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_bitnet_matvec(
    const npu_matmul_config_t* config,
    const void* weights_3bit,
    const void* weights_sign,
    const void* weights_2bit,
    const void* activations,
    void* output
);

/**
 * Execute BitNet matrix-vector with pre-allocated device buffers.
 * This is more efficient for repeated calls with same dimensions.
 * 
 * @param config Matrix multiplication configuration
 * @param buf_weights_3bit Device buffer for 3-bit weights
 * @param buf_weights_sign Device buffer for sign bits
 * @param buf_weights_2bit Device buffer for 2-bit weights
 * @param buf_activations Device buffer for activations
 * @param buf_output Device buffer for output
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_bitnet_matvec_buffers(
    const npu_matmul_config_t* config,
    npu_buffer_t buf_weights_3bit,
    npu_buffer_t buf_weights_sign,
    npu_buffer_t buf_weights_2bit,
    npu_buffer_t buf_activations,
    npu_buffer_t buf_output
);

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/**
 * Get the number of available NPU devices.
 * @return Number of devices
 */
int npu_get_device_count(void);

/**
 * Get device information string.
 * @param device_index Device index
 * @param info_buf Buffer to store info string
 * @param buf_size Size of buffer
 * @return NPU_SUCCESS on success, error code otherwise
 */
npu_error_t npu_get_device_info(int device_index, char* info_buf, size_t buf_size);

/**
 * Calculate packed weight sizes for given dimensions.
 * @param K3 3-bit portion size
 * @param K2 2-bit portion size  
 * @param M Number of rows
 * @param size_3bit Output: size of 3-bit packed weights
 * @param size_sign Output: size of sign bits
 * @param size_2bit Output: size of 2-bit packed weights
 */
void npu_calc_weight_sizes(int K3, int K2, int M,
                           size_t* size_3bit, size_t* size_sign, size_t* size_2bit);

#ifdef __cplusplus
}
#endif

#endif // BITNET_NPU_RUNTIME_H
