//===- npu_runtime.cpp - AMD NPU Runtime Implementation ---------*- C++ -*-===//
//
// XRT-based runtime implementation for AMD NPU acceleration of BitNet.
//
//===----------------------------------------------------------------------===//

#include "npu_runtime.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

//===----------------------------------------------------------------------===//
// Internal State
//===----------------------------------------------------------------------===//

namespace {

struct BufferWrapper {
    xrt::bo buffer;
    size_t size;
    bool host_accessible;
    void* mapped_ptr;
};

class NPURuntime {
public:
    static NPURuntime& instance() {
        static NPURuntime inst;
        return inst;
    }

    npu_error_t init(const npu_init_config_t* config);
    void shutdown();
    bool isReady() const { return initialized_; }
    const char* getErrorMessage() const { return error_message_.c_str(); }

    // Buffer operations
    npu_buffer_t allocBuffer(size_t size, bool host_accessible);
    void freeBuffer(npu_buffer_t buffer);
    npu_error_t writeBuffer(npu_buffer_t buffer, const void* src, size_t size);
    npu_error_t readBuffer(npu_buffer_t buffer, void* dst, size_t size);
    void* mapBuffer(npu_buffer_t buffer);
    npu_error_t syncToDevice(npu_buffer_t buffer);
    npu_error_t syncFromDevice(npu_buffer_t buffer);

    // Matrix operations
    npu_error_t executeMatVec(const npu_matmul_config_t* config,
                              const void* w3, const void* ws, const void* w2,
                              const void* act, void* out);
    npu_error_t executeMatVecBuffers(const npu_matmul_config_t* config,
                                     npu_buffer_t b3, npu_buffer_t bs,
                                     npu_buffer_t b2, npu_buffer_t ba,
                                     npu_buffer_t bo);

    int getDeviceCount();
    npu_error_t getDeviceInfo(int idx, char* buf, size_t size);

private:
    NPURuntime() : initialized_(false), verbosity_(0) {}
    ~NPURuntime() { shutdown(); }

    void setError(const char* msg) { error_message_ = msg; }
    BufferWrapper* getWrapper(npu_buffer_t buf);
    int getKernelGroupId(int arg_idx);

    bool initialized_;
    int verbosity_;
    std::string error_message_;
    
    std::unique_ptr<xrt::device> device_;
    std::unique_ptr<xrt::kernel> kernel_;
    xrt::xclbin xclbin_;
    xrt::hw_context hw_ctx_;
    
    std::mutex mutex_;
    std::vector<std::unique_ptr<BufferWrapper>> buffers_;
    std::vector<uint32_t> instr_buffer_;
};

npu_error_t NPURuntime::init(const npu_init_config_t* config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
        return NPU_SUCCESS;
    }

    if (!config || !config->xclbin_path) {
        setError("Invalid configuration");
        return NPU_ERROR_INVALID_PARAMS;
    }

    verbosity_ = config->verbosity;

    try {
        // Open device
        int dev_idx = (config->device_index >= 0) ? config->device_index : 0;
        device_ = std::make_unique<xrt::device>(dev_idx);
        
        if (verbosity_ >= 1) {
            fprintf(stderr, "NPU: Opened device %d\n", dev_idx);
        }

        // Load xclbin
        xclbin_ = xrt::xclbin(config->xclbin_path);
        device_->register_xclbin(xclbin_);
        
        if (verbosity_ >= 1) {
            fprintf(stderr, "NPU: Loaded xclbin %s\n", config->xclbin_path);
        }

        // Get hardware context
        hw_ctx_ = xrt::hw_context(*device_, xclbin_.get_uuid());

        // Get kernel
        auto kernels = xclbin_.get_kernels();
        if (kernels.empty()) {
            setError("No kernels found in xclbin");
            return NPU_ERROR_KERNEL_NOT_FOUND;
        }

        std::string kernel_name = kernels[0].get_name();
        kernel_ = std::make_unique<xrt::kernel>(hw_ctx_, kernel_name);
        
        if (verbosity_ >= 1) {
            fprintf(stderr, "NPU: Using kernel %s\n", kernel_name.c_str());
        }

        initialized_ = true;
        return NPU_SUCCESS;

    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_XCLBIN_LOAD_FAILED;
    }
}

void NPURuntime::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    buffers_.clear();
    kernel_.reset();
    device_.reset();
    initialized_ = false;
    
    if (verbosity_ >= 1) {
        fprintf(stderr, "NPU: Shutdown complete\n");
    }
}

int NPURuntime::getKernelGroupId(int arg_idx) {
    // Default group IDs - may need adjustment based on actual kernel
    return kernel_ ? kernel_->group_id(arg_idx) : 0;
}

BufferWrapper* NPURuntime::getWrapper(npu_buffer_t buf) {
    return static_cast<BufferWrapper*>(buf);
}

npu_buffer_t NPURuntime::allocBuffer(size_t size, bool host_accessible) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
        setError("NPU not initialized");
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<BufferWrapper>();
        
        xrt::bo::flags flags = host_accessible ? 
            XRT_BO_FLAGS_HOST_ONLY : XRT_BO_FLAGS_NONE;
        
        wrapper->buffer = xrt::bo(*device_, size, flags, getKernelGroupId(0));
        wrapper->size = size;
        wrapper->host_accessible = host_accessible;
        wrapper->mapped_ptr = host_accessible ? wrapper->buffer.map<void*>() : nullptr;

        BufferWrapper* ptr = wrapper.get();
        buffers_.push_back(std::move(wrapper));
        
        return static_cast<npu_buffer_t>(ptr);

    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    }
}

void NPURuntime::freeBuffer(npu_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(buffers_.begin(), buffers_.end(),
        [buffer](const std::unique_ptr<BufferWrapper>& w) {
            return w.get() == static_cast<BufferWrapper*>(buffer);
        });

    if (it != buffers_.end()) {
        buffers_.erase(it);
    }
}

npu_error_t NPURuntime::writeBuffer(npu_buffer_t buffer, const void* src, size_t size) {
    BufferWrapper* w = getWrapper(buffer);
    if (!w) {
        setError("Invalid buffer");
        return NPU_ERROR_INVALID_PARAMS;
    }

    try {
        void* dst = w->buffer.map<void*>();
        memcpy(dst, src, std::min(size, w->size));
        w->buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return NPU_SUCCESS;
    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

npu_error_t NPURuntime::readBuffer(npu_buffer_t buffer, void* dst, size_t size) {
    BufferWrapper* w = getWrapper(buffer);
    if (!w) {
        setError("Invalid buffer");
        return NPU_ERROR_INVALID_PARAMS;
    }

    try {
        w->buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        void* src = w->buffer.map<void*>();
        memcpy(dst, src, std::min(size, w->size));
        return NPU_SUCCESS;
    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

void* NPURuntime::mapBuffer(npu_buffer_t buffer) {
    BufferWrapper* w = getWrapper(buffer);
    if (!w || !w->host_accessible) {
        return nullptr;
    }
    return w->mapped_ptr;
}

npu_error_t NPURuntime::syncToDevice(npu_buffer_t buffer) {
    BufferWrapper* w = getWrapper(buffer);
    if (!w) {
        setError("Invalid buffer");
        return NPU_ERROR_INVALID_PARAMS;
    }

    try {
        w->buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return NPU_SUCCESS;
    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

npu_error_t NPURuntime::syncFromDevice(npu_buffer_t buffer) {
    BufferWrapper* w = getWrapper(buffer);
    if (!w) {
        setError("Invalid buffer");
        return NPU_ERROR_INVALID_PARAMS;
    }

    try {
        w->buffer.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return NPU_SUCCESS;
    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

npu_error_t NPURuntime::executeMatVec(const npu_matmul_config_t* config,
                                       const void* w3, const void* ws,
                                       const void* w2, const void* act,
                                       void* out) {
    if (!initialized_) {
        setError("NPU not initialized");
        return NPU_ERROR_NOT_INITIALIZED;
    }

    if (!config) {
        setError("Invalid config");
        return NPU_ERROR_INVALID_PARAMS;
    }

    // Calculate sizes
    size_t size_3bit, size_sign, size_2bit;
    npu_calc_weight_sizes(config->K3, config->K2, config->M,
                          &size_3bit, &size_sign, &size_2bit);
    size_t act_size = config->K * sizeof(int16_t);
    size_t out_size = config->M * sizeof(int32_t);

    try {
        // Allocate temporary buffers
        xrt::bo bo_w3(*device_, std::max(size_3bit, (size_t)1), 
                      XRT_BO_FLAGS_HOST_ONLY, getKernelGroupId(3));
        xrt::bo bo_ws(*device_, std::max(size_sign, (size_t)1),
                      XRT_BO_FLAGS_HOST_ONLY, getKernelGroupId(4));
        xrt::bo bo_w2(*device_, std::max(size_2bit, (size_t)1),
                      XRT_BO_FLAGS_HOST_ONLY, getKernelGroupId(5));
        xrt::bo bo_act(*device_, act_size,
                       XRT_BO_FLAGS_HOST_ONLY, getKernelGroupId(6));
        xrt::bo bo_out(*device_, out_size,
                       XRT_BO_FLAGS_HOST_ONLY, getKernelGroupId(7));

        // Copy input data
        if (w3 && size_3bit > 0) {
            memcpy(bo_w3.map<void*>(), w3, size_3bit);
            bo_w3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (ws && size_sign > 0) {
            memcpy(bo_ws.map<void*>(), ws, size_sign);
            bo_ws.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (w2 && size_2bit > 0) {
            memcpy(bo_w2.map<void*>(), w2, size_2bit);
            bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        
        memcpy(bo_act.map<void*>(), act, act_size);
        bo_act.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Zero output
        memset(bo_out.map<void*>(), 0, out_size);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Execute kernel
        auto run = (*kernel_)(bo_w3, bo_ws, bo_w2, bo_act, bo_out);
        run.wait();

        // Read results
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        
        // Apply scaling
        float scale = config->weight_scale / config->act_scale;
        int32_t* result = bo_out.map<int32_t*>();
        float* output = static_cast<float*>(out);
        
        for (int i = 0; i < config->M; i++) {
            output[i] = static_cast<float>(result[i]) * scale;
        }

        return NPU_SUCCESS;

    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

npu_error_t NPURuntime::executeMatVecBuffers(const npu_matmul_config_t* config,
                                              npu_buffer_t b3, npu_buffer_t bs,
                                              npu_buffer_t b2, npu_buffer_t ba,
                                              npu_buffer_t bo) {
    if (!initialized_) {
        setError("NPU not initialized");
        return NPU_ERROR_NOT_INITIALIZED;
    }

    BufferWrapper* wb3 = getWrapper(b3);
    BufferWrapper* wbs = getWrapper(bs);
    BufferWrapper* wb2 = getWrapper(b2);
    BufferWrapper* wba = getWrapper(ba);
    BufferWrapper* wbo = getWrapper(bo);

    if (!wba || !wbo) {
        setError("Invalid buffers");
        return NPU_ERROR_INVALID_PARAMS;
    }

    try {
        // Execute kernel with pre-allocated buffers
        auto run = (*kernel_)(
            wb3 ? wb3->buffer : xrt::bo(),
            wbs ? wbs->buffer : xrt::bo(),
            wb2 ? wb2->buffer : xrt::bo(),
            wba->buffer,
            wbo->buffer
        );
        run.wait();

        return NPU_SUCCESS;

    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_EXECUTION_FAILED;
    }
}

int NPURuntime::getDeviceCount() {
    // XRT doesn't have direct device enumeration in older versions
    // Try to open devices until failure
    int count = 0;
    for (int i = 0; i < 16; i++) {
        try {
            xrt::device dev(i);
            count++;
        } catch (...) {
            break;
        }
    }
    return count;
}

npu_error_t NPURuntime::getDeviceInfo(int idx, char* buf, size_t size) {
    try {
        xrt::device dev(idx);
        std::string info = "AMD NPU Device " + std::to_string(idx);
        strncpy(buf, info.c_str(), size - 1);
        buf[size - 1] = '\0';
        return NPU_SUCCESS;
    } catch (const std::exception& e) {
        setError(e.what());
        return NPU_ERROR_DEVICE_NOT_FOUND;
    }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// C API Implementation
//===----------------------------------------------------------------------===//

extern "C" {

npu_error_t npu_init(const char* xclbin_path) {
    npu_init_config_t config = {};
    config.xclbin_path = xclbin_path;
    config.device_index = 0;
    config.verbosity = 0;
    config.max_buffer_size = 0;
    return npu_init_with_config(&config);
}

npu_error_t npu_init_with_config(const npu_init_config_t* config) {
    return NPURuntime::instance().init(config);
}

void npu_shutdown(void) {
    NPURuntime::instance().shutdown();
}

bool npu_is_ready(void) {
    return NPURuntime::instance().isReady();
}

const char* npu_get_error_message(void) {
    return NPURuntime::instance().getErrorMessage();
}

npu_buffer_t npu_buffer_alloc(size_t size, bool host_accessible) {
    return NPURuntime::instance().allocBuffer(size, host_accessible);
}

void npu_buffer_free(npu_buffer_t buffer) {
    NPURuntime::instance().freeBuffer(buffer);
}

npu_error_t npu_buffer_write(npu_buffer_t buffer, const void* src, size_t size) {
    return NPURuntime::instance().writeBuffer(buffer, src, size);
}

npu_error_t npu_buffer_read(npu_buffer_t buffer, void* dst, size_t size) {
    return NPURuntime::instance().readBuffer(buffer, dst, size);
}

void* npu_buffer_map(npu_buffer_t buffer) {
    return NPURuntime::instance().mapBuffer(buffer);
}

npu_error_t npu_buffer_sync_to_device(npu_buffer_t buffer) {
    return NPURuntime::instance().syncToDevice(buffer);
}

npu_error_t npu_buffer_sync_from_device(npu_buffer_t buffer) {
    return NPURuntime::instance().syncFromDevice(buffer);
}

npu_error_t npu_bitnet_matvec(const npu_matmul_config_t* config,
                              const void* weights_3bit,
                              const void* weights_sign,
                              const void* weights_2bit,
                              const void* activations,
                              void* output) {
    return NPURuntime::instance().executeMatVec(config,
        weights_3bit, weights_sign, weights_2bit, activations, output);
}

npu_error_t npu_bitnet_matvec_buffers(const npu_matmul_config_t* config,
                                       npu_buffer_t buf_weights_3bit,
                                       npu_buffer_t buf_weights_sign,
                                       npu_buffer_t buf_weights_2bit,
                                       npu_buffer_t buf_activations,
                                       npu_buffer_t buf_output) {
    return NPURuntime::instance().executeMatVecBuffers(config,
        buf_weights_3bit, buf_weights_sign, buf_weights_2bit,
        buf_activations, buf_output);
}

int npu_get_device_count(void) {
    return NPURuntime::instance().getDeviceCount();
}

npu_error_t npu_get_device_info(int device_index, char* info_buf, size_t buf_size) {
    return NPURuntime::instance().getDeviceInfo(device_index, info_buf, buf_size);
}

void npu_calc_weight_sizes(int K3, int K2, int M,
                           size_t* size_3bit, size_t* size_sign, size_t* size_2bit) {
    // 3-bit: 3 ternary values per nibble, 2 nibbles per byte
    *size_3bit = (K3 > 0) ? (((K3 / 3) + 1) / 2) * M : 0;
    // Sign: 1 bit per 3-value group, 8 groups per byte
    *size_sign = (K3 > 0) ? (((K3 / 3) + 7) / 8) * M : 0;
    // 2-bit: 2 ternary values per nibble, 2 nibbles per byte
    *size_2bit = (K2 > 0) ? (((K2 / 2) + 1) / 2) * M : 0;
}

} // extern "C"
