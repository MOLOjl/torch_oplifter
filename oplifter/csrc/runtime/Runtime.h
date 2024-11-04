#pragma once

#include <ATen/Config.h>
#include <c10/core/Device.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>

using namespace dnnl;
using DeviceId = c10::DeviceIndex;

namespace oplifter {

static constexpr int kQueuesPerPool = 1;

static inline dnnl::memory dpcpp_onednn_memory(
    dnnl::memory::desc md,
    dnnl::engine& engine,
    void* ptr) {
  return dnnl::memory(md, engine, 0, ptr);
}

// Init queue pool's state.
void dpcppInitQueueStateOnce();
// Inits queue pool on the specified device.
void dpcppInitDeviceQueueOnce(DeviceId device_index);

sycl::queue& dpcppGetRawQueue(DeviceId device_index, int queue_index);

DeviceId dpcppGetCurDevice();

int dpcppGetDeviceCount();

sycl::device& dpcppGetRawDevice(DeviceId device_id);

sycl::context& dpcppGetDeviceContext(DeviceId device_id);

// GpuEngineManager singleton
struct GpuEngineManager {
  static GpuEngineManager& Instance(); // Singleton

  engine& get_engine(const at::Device& device) {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::PrivateUse1);
    TORCH_INTERNAL_ASSERT(device.index() < dpcppGetDeviceCount());
    return *engine_pool[device.index()];
  }

  GpuEngineManager(GpuEngineManager const&) = delete;
  GpuEngineManager& operator=(GpuEngineManager const&) = delete;

 protected:
  GpuEngineManager() {
    int device_count = 1;
    for (int i = 0; i < device_count; i++) {
      // engine_pool.push_back(std::make_shared<dnnl::engine>(
      //   dnnl::engine(dnnl::engine::kind::gpu, i)));
      engine_pool.push_back(
          std::make_shared<dnnl::engine>(dnnl::sycl_interop::make_engine(
              dpcppGetRawDevice(i), dpcppGetDeviceContext(i))));
    }
  }
  ~GpuEngineManager() {}

 private:
  std::vector<std::shared_ptr<dnnl::engine>> engine_pool;
};

// GpuStreamManager singleton
struct GpuStreamManager {
  static GpuStreamManager& Instance(); // Singleton

  // USE_PERSIST_STREAM
  dnnl::stream& get_stream() {
    DeviceId device_index = dpcppGetCurDevice();
    TORCH_INTERNAL_ASSERT(device_index < dpcppGetDeviceCount());
    DeviceId queue_id = 0;
    // int queue_id = getCurrentDPCPPStream(device_index).queue_index();
    if (stream_pool[device_index][queue_id] == nullptr) {
      stream_pool[device_index][queue_id] =
          std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(
              GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, device_index}),
              dpcppGetRawQueue(device_index, queue_id)));
    }
    return *(stream_pool[device_index][queue_id].get());
  }

  GpuStreamManager(GpuStreamManager const&) = delete;
  GpuStreamManager& operator=(GpuStreamManager const&) = delete;

 protected:
  GpuStreamManager() {
    int deviceCount = dpcppGetDeviceCount();
    TORCH_INTERNAL_ASSERT(deviceCount > 0);
    stream_pool.clear();
    stream_pool.resize(deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
      for (int qid = 0; qid < kQueuesPerPool; qid++) {
        stream_pool[dev][qid] = nullptr;
      }
    }
  }

  ~GpuStreamManager() {}

 private:
  // For each device, we have kQueuesPerPool(32) reserved queues.
  std::vector<std::array<std::shared_ptr<dnnl::stream>, kQueuesPerPool>>
      stream_pool;
};

} // namespace oplifter
