#include <torch/csrc/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include "csrc/runtime/OplifterCachingAllocator.h"
#include "csrc/runtime/raw_mem_manager.h"
#include "csrc/runtime/Runtime.h"
#include "csrc/operators/_Ops.h"

#include <chrono>

using namespace at;

namespace oplifter {

// > see pytorch-v2.3.0/aten/src/ATen/native/native_functions.yaml:7599
Tensor opl_add_Tensor(const Tensor & self, const Tensor & other, const Scalar & alpha);
Tensor opl_div_Tensor(const Tensor & self, const Tensor & other);
Tensor opl_linear(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt);
Tensor opl_matmul(const Tensor& input, const Tensor& other);
Tensor opl_mul_Tensor(const Tensor & self, const Tensor & other);
Tensor opl_silu(const Tensor& self);
Tensor opl_softmax(const Tensor& self, int64_t dim, bool half_to_float);
Tensor opl_sub_Tensor(const Tensor & self, const Tensor & other, const Scalar & alpha);

Tensor opl_to_device(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<MemoryFormat> memory_format);
Tensor & opl_fill__scalar(Tensor & self, const Scalar & value);
Tensor opl__copy_from(const Tensor& self, const Tensor& dst, bool non_blocking);
Tensor opl_empty_memory_format(IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> memory_format);
Tensor opl_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt);


} // oplifter

