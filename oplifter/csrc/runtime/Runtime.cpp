#include "csrc/runtime/Runtime.h"

#include <deque>
// #include "Utils.h"

namespace oplifter {

using namespace dnnl;

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

#pragma region Queue

static std::deque<std::once_flag> device_flags;
static std::once_flag init_queuestate_flag;

static constexpr int kQueueTypeBits = 3;

static std::vector<std::array<std::unique_ptr<sycl::queue>, kQueuesPerPool>> reserved_queues;
static std::vector<std::unordered_map<sycl::queue, uint8_t>> queue_to_idx_map;
static std::deque<std::atomic<uint32_t>> reserved_counters;

// Warning: this function must only be called once!
static void initGlobalQueueState() {
  int num_devices = dpcppGetDeviceCount();
  TORCH_CHECK(num_devices > 0, "Number of SYCL devices should be greater than zero!");

  device_flags.resize(num_devices);
  reserved_queues.resize(num_devices);
  queue_to_idx_map.resize(num_devices);
  reserved_counters.resize(num_devices);
}

// Warning: only call once per device!
static void initDeviceQueue(DeviceId device_index) {
  // Creates the reserved sycl queue pools for the specified device.
  for (auto i = 0; i < kQueuesPerPool; i++) {
    reserved_queues[device_index][i] =
        std::make_unique<sycl::queue>(sycl::queue(
            dpcppGetDeviceContext(device_index),
            dpcppGetRawDevice(device_index)));
    const auto& queue_ptr = reserved_queues[device_index][i];
    queue_to_idx_map[device_index][*queue_ptr] = i;
  }

  reserved_counters[device_index] = 0;
}

// Inits queue pool's size to ensure initialization only occurs once.
void dpcppInitQueueStateOnce() {
  std::call_once(init_queuestate_flag, initGlobalQueueState);
}

void dpcppInitDeviceQueueOnce(DeviceId device_index) {
  std::call_once(device_flags[device_index], initDeviceQueue, device_index);
}

sycl::queue& dpcppGetRawQueue(DeviceId device_index, int queue_index) {
  dpcppInitQueueStateOnce();
  dpcppInitDeviceQueueOnce(device_index);
  return *reserved_queues[device_index][queue_index];
}

#pragma endregion 

// Stream is not a real thing in SYCL. oneDNN's stream is an encapsulation of queue.
#pragma region Stream

// TODO:

#pragma endregion

#pragma region Device

static std::once_flag init_device_flag;
static thread_local int cur_dev_index = 0;

struct DPCPPDevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  // If macro USE_MULTI_CONTEXT is enabled, contexts will be constructed by SYCL
  // runtime API sycl::context. Otherwise, contexts will be initialized by
  // default context that shared by all GPU devices.
  std::vector<std::unique_ptr<sycl::context>> contexts;
  std::mutex devices_mutex;
} gDevPool;

static void enumDevices(std::vector<std::unique_ptr<sycl::device>>& devices) {
  std::vector<sycl::device> root_devices;
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from GPU platform firstly, just HIP for now.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_hip) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        root_devices.push_back(device);
      }
    }
  }

  for (const auto& root_device : root_devices) {
    // Tile partition is disabled, all root-devices are returned.
    devices.push_back(std::make_unique<sycl::device>(root_device));
  }
}

static void initGlobalDevicePoolState() {
  enumDevices(gDevPool.devices);

  auto device_count = gDevPool.devices.size();
  if (device_count <= 0) {
    TORCH_WARN("SYCL HIP Device count is zero!");
    return;
  }

  // USE MULTI CONTEXT
  gDevPool.contexts.resize(device_count);
  for (int i = 0; i < device_count; i++) {
    gDevPool.contexts[i] = std::make_unique<sycl::context>(
        sycl::context({*gDevPool.devices[i]}));
  }
}

static void initDevicePoolCallOnce() {
  std::call_once(init_device_flag, initGlobalDevicePoolState);
}

DeviceId dpcppGetCurDevice() {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  return (DeviceId)cur_dev_index;
}

int dpcppGetDeviceCount() {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  return (int)gDevPool.devices.size();
}

sycl::device& dpcppGetRawDevice(DeviceId device_id) {
  initDevicePoolCallOnce();
  std::lock_guard<std::mutex> lock(gDevPool.devices_mutex);
  if (device_id >= (DeviceId)gDevPool.devices.size()) {
    TORCH_CHECK(0, "dpcppGetRawDevice: device_id is out of range");
  }

  if(device_id == -1)
    return *gDevPool.devices[dpcppGetCurDevice()];
  return *gDevPool.devices[device_id];
}

sycl::context& dpcppGetDeviceContext(DeviceId device_id) {
  initDevicePoolCallOnce();
  if (device_id >= (DeviceId)gDevPool.contexts.size()) {
    TORCH_CHECK(0, "dpcppGetDeviceContext: device_id is out of range");
  }
  if(device_id == -1)
    return *gDevPool.contexts[dpcppGetCurDevice()];
  return *gDevPool.contexts[device_id];
}

#pragma endregion




} // namespace oplifter
