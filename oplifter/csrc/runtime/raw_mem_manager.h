#pragma once
#include "oneapi/dnnl/dnnl.hpp"
#include "dnnl_sycl.hpp"
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif  
namespace oplifter {

static inline void* alloc_raw_mem(size_t size){
  void* ptr = nullptr;
  // platform cuda: 0, rocm: 1
#ifdef USE_ROCM
  hipError_t err = hipMalloc(&ptr, size);
  if(err != hipSuccess){
    printf("hipMalloc failed: %s\n", hipGetErrorString(err));
    throw 0;
  }
  // printf("hip malloc raw: %p, size:%ld\n", ptr, size);
#endif
  return ptr;
}

static inline void raw_mem_copy(void* dst, void* src, size_t size, int mode){
  // mode, 0:host2host, 1:host2device, 2:device2host, 3:device2device
#ifdef USE_ROCM
  hipError_t err = hipSuccess;
  switch (mode)
  {
  case 1:
    err = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    break;
  case 2:
    err = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    break;
  case 3:
    err = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
    break;
  default:
    assert("invalid argument" && 0);
    break;
  }
  if(err != hipSuccess){
    printf("hipMemcpy failed: %s\n", hipGetErrorString(err));
    throw 0;
  }
#endif
}

static inline void free_raw_mem(void* ptr){
  // printf("free_raw_mem : %p \n", ptr);
#ifdef USE_ROCM
  hipError_t err = hipFree(ptr);
  if(err != hipSuccess){
    printf("hipFree failed: %s\n", hipGetErrorString(err));
    throw 0;
  }
#endif
}

static inline void set_mem_zero(void* ptr, size_t size){
  // printf("set zero, size:%ld\n", size);
#ifdef USE_ROCM
  hipError_t err = hipMemset(ptr, 0, size);
  if(err != hipSuccess){
    printf("hipMemset failed: %s\n", hipGetErrorString(err));
    throw 0;
  }
#endif
}

static inline void debug_check_device_data(void* ptr){
  std::vector<float> h_buf(2);
  raw_mem_copy(h_buf.data(), ptr, sizeof(float)*2, 2);
  printf("debug check:%f, %f\n", h_buf[0], h_buf[1]);
}

} // namespace oplifter
