#include "csrc/impl/oplifter_impl.h"
#include "csrc/runtime/OplifterCachingAllocator.h"

using namespace at;

// register guard
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

// Register our dummy allocator
oplifter::OliCachingAllocator global_oli_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_oli_allocator);

namespace oplifter {

// basic dummy add function
at::Tensor opl_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::bin(dnnl::algorithm::binary_add, out, self, other);
  return out;
}

// basic dummy mul function
at::Tensor opl_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::bin(dnnl::algorithm::binary_mul, out, self, other);
  return out;
}

// basic dummy div function
at::Tensor opl_div_Tensor(const at::Tensor & self, const at::Tensor & other) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::bin(dnnl::algorithm::binary_div, out, self, other);
  return out;
}

// basic dummy sub function
at::Tensor opl_sub_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::bin(dnnl::algorithm::binary_sub, out, self, other);
  return out;
}

// basic dummy silu function
at::Tensor opl_silu(const at::Tensor& self) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::eltwise(dnnl::algorithm::eltwise_swish, out, self);
  return out;
}

// _softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor opl_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  at::Tensor out = empty(self.sizes(), self.options());
  oplifter::softmax(self, (int64_t)dim, half_to_float, dnnl::algorithm::softmax_accurate, out);
  return out;
}

// linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
at::Tensor opl_linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt) {
  at::Tensor out;
  oplifter::linear(input, weight, out, bias_opt);
  return out;
}

// matmul(Tensor self, Tensor other) -> Tensor
at::Tensor opl_matmul(const at::Tensor& input, const at::Tensor& other) { 
  // printf("opl_matmul\n");
  Tensor out;
  oplifter::matmul(input, other, out);
  return out;
}

// this function won't be execute.
at::Tensor opl_to_device(
    const at::Tensor & self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<MemoryFormat> memory_format) {
  printf("opl_to_device\n");

  // TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "only allows copy from cpu -> custom device.");
  // TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1, "only allows copy from cpu -> custom device.");
  // // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  // TORCH_CHECK(self.scalar_type() == dtype);
  // TORCH_CHECK(self.is_contiguous());

  // auto options = TensorOptions().dtype(dtype);
  // auto out = from_blob(d_ptr, self.sizes(), options);
  
  // empty(self.sizes(), dtype, self.options().layout(), device, false, memory_format);
  // memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  // Since this custom device is just for testing, not bothering to implement kernels.
  return self;
}

// This function will not be executed.
at::Tensor& opl_fill__scalar(at::Tensor& self, const at::Scalar& value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows dummy device.");
  TORCH_CHECK(self.is_contiguous());
  // TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);
  // printf("opl_fill__scalar, length:%ld\n", self.numel());
  oplifter::fill_tensor(self, value);
  return self;
}

// to.device()
at::Tensor opl__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "only allows copy from cpu -> opl device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "only allows copy from cpu -> opl device.");
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous(), "non-contiguous copy?");
  
  if(self.scalar_type() == dst.scalar_type()) {
    // cast
    
  }

  // dst and src storage nbytes may different when at least one of them be another tensor's view.
  size_t copy_nbytes = std::min<size_t>(self.storage().nbytes(), dst.storage().nbytes());

  if(self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1) {
    // printf("copy %ld bytes to device.\n", copy_nbytes);
    oplifter::raw_mem_copy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), copy_nbytes, 1);
  }
  if(dst.is_cpu() && self.device().type() == c10::DeviceType::PrivateUse1) {
    // printf("copy %ld bytes to host.\n", copy_nbytes);
    oplifter::raw_mem_copy(dst.data_ptr(), self.data_ptr(), copy_nbytes, 2);
  }

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  return dst;
}

at::Tensor opl_empty_memory_format(IntArrayRef size,
                                      c10::optional<ScalarType> dtype,
                                      c10::optional<Layout> layout,
                                      c10::optional<Device> device,
                                      c10::optional<bool> pin_memory,
                                      c10::optional<MemoryFormat> memory_format) {
  printf("opl_empty_memory_format\n");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_oli_allocator, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor opl_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<ScalarType> dtype_opt, 
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_oli_allocator, private_use_ks, dtype);
}

at::Tensor opl_zeros(c10::IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt, 
    c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  auto out = at::detail::empty_generic(size, &global_oli_allocator, private_use_ks, dtype, c10::nullopt);

  auto dtype_meta = c10::scalarTypeToTypeMeta(dtype);
  auto numel = c10::multiply_integers(size);
  auto size_bytes = dtype_meta.itemsize() * numel;
  oplifter::set_mem_zero(out.data_ptr(), size_bytes);

  return out;
}

at::Tensor opl_ones(c10::IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt, 
    c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  at::Tensor out = at::detail::empty_generic(size, &global_oli_allocator, private_use_ks, dtype, c10::nullopt);
  return opl_fill__scalar(out, 1);
}

at::Tensor opl_arange__start_step(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, 
    c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, 
    c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  // TODO: use double to calculate length may cause a bug, do some type check.

  double length_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>());
  int64_t length = static_cast<int64_t>(length_d);

  at::ScalarType dtype = at::ScalarType::Long;
  if(dtype_opt.has_value())
    dtype = c10::dtype_or_default(dtype_opt);

  if(start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint())
    dtype = get_default_dtype_as_scalartype();

  Tensor out = at::detail::empty_generic({length}, &global_oli_allocator, private_use_ks, dtype, c10::nullopt);
  oplifter::arange_step(out, start, end, step);
  return out;
}

at::Tensor opl_arange_(const at::Scalar& end, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, 
    c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  return opl_arange__start_step(0, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}
at::Tensor& opl_arange__start_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out) {
  oplifter::arange_step(out, start, end, step);
  return out;
}
at::Tensor& opl_arange__out(const at::Scalar& end, at::Tensor& out) {
  return opl_arange__start_out(0, end, 1, out);
}

// as_strided
at::Tensor as_strided_tensorimpl_opl(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride, 
    c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.

} // oplifter



TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("to.Device", &oplifter::opl_to_device);
}

// pytorch/aten/src/ATen/native/native_functions.yaml
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &oplifter::opl_add_Tensor);
  m.impl("sub.Tensor", &oplifter::opl_sub_Tensor);
  m.impl("mul.Tensor", &oplifter::opl_mul_Tensor);
  m.impl("div.Tensor", &oplifter::opl_div_Tensor);
  m.impl("silu", &oplifter::opl_silu);
  // m.impl("relu", &opl_relu);
  m.impl("linear", &oplifter::opl_linear);
  m.impl("matmul", &oplifter::opl_matmul);
  m.impl("_softmax", &oplifter::opl_softmax);
  // m.impl("layer_norm", &opl_layer_norm);
  // m.impl("multinomial", &opl_multinomial);
  // m.impl("masked_fill_.Scalar", &opl_masked_fill_scalar);
  // m.impl("masked_fill_.Tensor", &opl_masked_fill_tensor);
  // m.impl("where", &opl_where);

  m.impl("to.device", &oplifter::opl_to_device);
  m.impl("fill_.Scalar", &oplifter::opl_fill__scalar);
  m.impl("_copy_from", &oplifter::opl__copy_from);
  m.impl("empty.memory_format", &oplifter::opl_empty_memory_format);
  m.impl("empty_strided", &oplifter::opl_empty_strided);
  m.impl("zeros", &oplifter::opl_zeros);
  m.impl("ones", &oplifter::opl_ones);

  m.impl("arange.start_step", &oplifter::opl_arange__start_step);
  m.impl("arange", &oplifter::opl_arange_);
  m.impl("arange.start_out", &oplifter::opl_arange__start_out);
  // m.impl("arange.start", &oplifter::opl_arange__start);
  m.impl("arange.out", &oplifter::opl_arange__out);
  
  m.impl("as_strided", &oplifter::as_strided_tensorimpl_opl);
  // empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  // zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  // ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

  // arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
  // arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  // arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  // arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  // arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)

  // m.impl("contiguous.Tensor", &oplifter::opl_contiguous)
}