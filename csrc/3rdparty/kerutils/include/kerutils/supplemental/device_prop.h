#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

#include <deque>
#include <mutex>
#include <vector>

namespace kerutils {

// Cached device properties (SM count, compute capability, ...) for the current
// device, queried via the CUDA Runtime and cached per device index. ABI-stable
// replacement for at::cuda::getCurrentDeviceProperties().
// Thread-safe: this mirrors vLLM's csrc/libtorch_stable/torch_utils.h
namespace detail {

inline std::deque<std::once_flag> device_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag device_vectors_init_flag;

inline void init_device_vectors() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "cudaGetDeviceCount failed: ", cudaGetErrorString(err));
    device_flags.resize(device_count);
    device_properties.resize(device_count);
}

inline void init_device_property(int device_index) {
    cudaDeviceProp device_prop{};
    cudaError_t err = cudaGetDeviceProperties(&device_prop, device_index);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "cudaGetDeviceProperties failed: ", cudaGetErrorString(err));
    device_properties[device_index] = device_prop;
}

}  // namespace detail

inline const cudaDeviceProp &get_cached_device_prop() {
    std::call_once(detail::device_vectors_init_flag, detail::init_device_vectors);
    int device_index = static_cast<int>(torch::stable::accelerator::getCurrentDeviceIndex());
    STD_TORCH_CHECK(
        device_index >= 0 &&
            static_cast<size_t>(device_index) < detail::device_properties.size(),
        "CUDA device index ", device_index, " out of range [0, ",
        detail::device_properties.size(), ")");
    std::call_once(detail::device_flags[device_index], detail::init_device_property,
                   device_index);
    return detail::device_properties[device_index];
}

}  // namespace kerutils
