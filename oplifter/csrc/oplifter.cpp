#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <ATen/core/GeneratorForPrivateuseone.h>

#include "csrc/runtime/OplifterCachingAllocator.h"
#include "csrc/runtime/Runtime.h"

#include <chrono>

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

void sync_device(){
  auto strm = oplifter::GpuStreamManager::Instance().get_stream();
  strm.wait();
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // Constructors
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sync_device", &sync_device, "synchronize custom device");
  m.def("custom_device", &get_custom_device, "get custom device object");
  m.def("register_generator", &register_generator, "register generator for custom device");
}
