#include "csrc/operators/_Ops.h"
#include "csrc/runtime/raw_mem_manager.h"

using namespace dnnl;

namespace oplifter {

static inline void get_dnnl_format(
    const at::Tensor& input,
    memory::format_tag& dnnl_format,
    memory::dims& input_tz) {
  auto input_sizes = input.sizes();
  auto input_ndim = input_sizes.size();

  if (input_ndim == 1) {
    dnnl_format = memory::format_tag::x;
    input_tz = {input.size(0)};
  } else if (input_ndim == 2) {
    dnnl_format = memory::format_tag::nc;
    input_tz = {input.size(0), input.size(1)};
  } else if (input_ndim == 3) {
    dnnl_format = memory::format_tag::tnc;
    input_tz = {input.size(0), input.size(1), input.size(2)};
  } else if (input_ndim == 4) {
    dnnl_format = memory::format_tag::nchw;
    input_tz = {
        /*n*/ input.size(0),
        /*c*/ input.size(1),
        /*h*/ input.size(2),
        /*w*/ input.size(3)};
  } else {
    std::stringstream ss;
    ss << "softmax backend got shape=" << input_sizes
       << ", expected input with rank 1 to  rank 4 shape";
    AT_ERROR(ss.str());
  }
}

// _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor softmax(
    const at::Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    dnnl::algorithm softmax_algo,
    at::Tensor& output) {
  TORCH_CHECK(input.dim() <= 4 && input.dim() >= 1, "Input Dims out of range");

  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::format_tag dnnl_format;
  memory::dims input_tz;
  get_dnnl_format(input, dnnl_format, input_tz);

  auto data_type = get_onednn_dtype(input);

  auto input_md = memory::desc({input_tz}, data_type, get_onednn_strides(input));
  auto axis = dim < 0 ? dim + input.dim() : dim;

  // Create primitive descriptor.
  auto softmax_forward_pd = dnnl::softmax_forward::primitive_desc(
      engine,
      prop_kind::forward,
      softmax_algo,
      input_md,
      input_md,
      axis);

  // inplace?
  if (!output.defined()) {
    output = at::empty_like(input);
  }

  auto input_mem = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  auto output_mem = dpcpp_onednn_memory(input_md, engine, output.data_ptr());

  // Create the primitive.
  auto softmax_onednn_forward = dnnl::softmax_forward(softmax_forward_pd);
  
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, input_mem});
  args.insert({DNNL_ARG_DST, output_mem});
  printf("softmax prim execute\n");
  softmax_onednn_forward.execute(strm, args);
  strm.wait();
  printf("softmax prim execute over\n");
  return output;
}

at::Tensor softmax_backward(
    const at::Tensor& grad,
    const at::Tensor& output,
    int64_t dim,
    bool half_to_float,
    dnnl::algorithm softmax_algo,
    at::Tensor gI) {
  TORCH_CHECK(grad.dim() <= 4 && grad.dim() >= 1, "Input Dims out of range");
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();
  
  if (!gI.defined()) {
    gI = at::empty_like(grad);
  }
  memory::format_tag output_dnnl_format;
  memory::format_tag grad_dnnl_format;
  memory::dims output_tz;
  memory::dims grad_tz;

  get_dnnl_format(output, output_dnnl_format, output_tz);
  get_dnnl_format(grad, grad_dnnl_format, grad_tz);

  auto output_type = get_onednn_dtype(output);
  auto grad_type = get_onednn_dtype(grad);

  auto axis = dim < 0 ? dim + grad.dim() : dim;

  auto output_md =  memory::desc({output_tz}, output_type, output_dnnl_format);
  auto output_memory = dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  auto grad_md = memory::desc({grad_tz, grad_type, grad_dnnl_format});
  auto grad_memory = dpcpp_onednn_memory(grad_md, engine, grad.data_ptr());

  auto softmax_forward_pd = softmax_forward::primitive_desc(
      engine,
      prop_kind::forward,
      softmax_algo,
      output_md,
      output_md,
      axis);

  auto softmax_backward_pd = dnnl::softmax_backward::primitive_desc(
      engine,
      softmax_algo,
      grad_md,
      grad_md,
      output_md,
      axis,
      softmax_forward_pd);

  auto gi_md = memory::desc({grad_tz, grad_type, grad_dnnl_format});
  auto gi_memory = dpcpp_onednn_memory(gi_md, engine, gI.data_ptr());

  // Create the primitive.
  auto softmax_onednn_backward = dnnl::softmax_backward(softmax_backward_pd);

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_DST, output_memory});
  args.insert({DNNL_ARG_DIFF_SRC, gi_memory});
  args.insert({DNNL_ARG_DIFF_DST, grad_memory});

  softmax_onednn_backward.execute(strm, args);
  return gI;
}

} // namespace oplifter