#include "csrc/operators/_Ops.h"
#include "csrc/runtime/raw_mem_manager.h"
#include <chrono>

using namespace dnnl;
namespace oplifter {

void check_nd_flatten_dims(int ndims, memory::dims &t_dims, memory::dims &t_strides){
  // check continous 
  bool continous = true;
  continous = continous && t_strides[ndims-1] == 1;
  for (int i = ndims-2; i >= 0; i--)
    continous = continous && t_strides[i+1]*t_dims[i+1];
  
  TORCH_CHECK(continous, "oneDNN backend linear need tensor to be continous.");

  int64_t flatten_dim = 1, keeped_dim = t_dims[ndims-1];
  for(int i=2; i<=ndims; i++) 
    flatten_dim *= t_dims[ndims-i];

  t_dims.resize(2);
  t_strides.resize(2);

  t_dims[0] = flatten_dim;
  t_dims[1] = keeped_dim;
  t_strides[0] = keeped_dim;
  t_strides[1] = 1;
}

at::Tensor linear(
  const at::Tensor& input, 
  const at::Tensor& weight, 
  at::Tensor& output, 
  const c10::optional<at::Tensor>& bias_opt) {
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim > 1 && weight_dim > 1,
              "oneDNN backend linear need tensor to be at least 2D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  memory::dims input_dims = get_onednn_dims(input);
  memory::dims weight_dims = get_onednn_dims(weight);

  memory::dims input_strides = get_onednn_strides(input);
  memory::dims weight_strides = get_onednn_strides(weight);

  check_nd_flatten_dims(input_dim, input_dims, input_strides);
  check_nd_flatten_dims(weight_dim, input_dims, input_strides);

  // Since input shape:(*, H_in), weight shape:(H_out, H_in)
  if(!output.defined()) {
    std::vector<long> out_sizes(input.sizes().begin(), input.sizes().end());
    // c10::ArrayRef<long> out_sizes();
    out_sizes[input_dim-1] = weight_dims[0];
    output = at::empty(out_sizes, input.options());
  }

  at::Tensor bias;
  if(bias_opt.has_value())
    bias = *bias_opt;

  auto input_md = memory::desc(input_dims, get_onednn_dtype(input), input_strides);
  auto weights_md = memory::desc(weight_dims, get_onednn_dtype(weight), weight_strides);

  memory::dims output_dims{input_dims[0], weight_dims[0]};
  memory::dims output_strides{weight_dims[0], 1};
  auto output_md = memory::desc(output_dims, get_onednn_dtype(weight), output_strides);

  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();
  auto input_mem = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  auto weights_mem = dpcpp_onednn_memory(weights_md, engine, weight.data_ptr());
  auto output_mem = dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  memory::desc bias_md;
  dnnl::memory bias_mem;
  if (bias_opt.has_value()) {
    auto bias_dims = get_onednn_dims(bias);
    auto bias_strides = get_onednn_strides(bias);
    if(bias_dims.size() == 1){
      bias_dims.insert(bias_dims.begin(), 1);
      bias_strides.insert(bias_strides.begin(), bias_dims[1]);
    }

    bias_md = memory::desc(bias_dims, get_onednn_dtype(bias), bias_strides);
    bias_mem = dpcpp_onednn_memory(bias_md, engine, bias.data_ptr());
  }

  printf("input:%p, weight:%p, output:%p\n", 
    input.data_ptr(), weight.data_ptr(), output.data_ptr());

  printf("create primitive_desc\n");
  printf("shape src:%ld,%ld, wei:%ld,%ld, dst:%ld,%ld\n",
    input_dims[0], input_dims[1], weight_dims[0], weight_dims[1], output_dims[0], output_dims[1]);

  // primitive_attr attr;
  dnnl::matmul::primitive_desc pd;
  if(bias_opt.has_value())
    pd = dnnl::matmul::primitive_desc(engine, input_md, weights_md, bias_md, output_md, false, true);
  else
    pd = dnnl::matmul::primitive_desc(engine, input_md, weights_md, output_md, false, true);

  printf("create primitive\n");
  auto prim = dnnl::matmul(pd);
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, input_mem});
  args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  args.insert({DNNL_ARG_DST, output_mem});
  if(bias_opt.has_value())
    args.insert({DNNL_ARG_BIAS, bias_mem});
  
  printf("matmul prim execute\n");
  prim.execute(strm, args);
  strm.wait();
  printf("matmul prim execute over\n");
  return output;
}

at::Tensor matmul(
  const at::Tensor& input, 
  const at::Tensor& weight, 
  at::Tensor& output) {
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  // TODO: deal with X-D matmul
  // TORCH_CHECK();

  memory::dims input_dims = get_onednn_dims(input);
  memory::dims weight_dims = get_onednn_dims(weight);

  memory::dims input_strides = get_onednn_strides(input);
  memory::dims weight_strides = get_onednn_strides(weight);

  check_nd_flatten_dims(input_dim, input_dims, input_strides);
  check_nd_flatten_dims(weight_dim, input_dims, input_strides);

  // Since input shape:(*, H_in), weight shape:(H_out, H_in)
  if(!output.defined()) {
    std::vector<long> out_sizes(input.sizes().begin(), input.sizes().end());
    // c10::ArrayRef<long> out_sizes();
    out_sizes[input_dim-1] = weight_dims[0];
    output = at::empty(out_sizes, input.options());
  }

  auto input_md = memory::desc(input_dims, get_onednn_dtype(input), input_strides);
  auto weights_md = memory::desc(weight_dims, get_onednn_dtype(weight), weight_strides);

  memory::dims output_dims{input_dims[0], weight_dims[0]};
  memory::dims output_strides{weight_dims[0], 1};
  auto output_md = memory::desc(output_dims, get_onednn_dtype(weight), output_strides);

  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();
  auto input_mem = dpcpp_onednn_memory(input_md, engine, input.data_ptr());
  auto weights_mem = dpcpp_onednn_memory(weights_md, engine, weight.data_ptr());
  auto output_mem = dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  printf("input:%p, weight:%p, output:%p\n", 
    input.data_ptr(), weight.data_ptr(), output.data_ptr());

  // printf("create primitive_desc\n");
  // printf("shape src:%ld,%ld, wei:%ld,%ld, dst:%ld,%ld\n",
  //   input_dims[0], input_dims[1], weight_dims[0], weight_dims[1], output_dims[0], output_dims[1]);

  // primitive_attr attr;
  dnnl::matmul::primitive_desc pd;
  pd = dnnl::matmul::primitive_desc(engine, input_md, weights_md, output_md, false, false);

  // printf("create primitive\n");
  auto prim = dnnl::matmul(pd);
  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, input_mem});
  args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  args.insert({DNNL_ARG_DST, output_mem});
  
  // printf("matmul prim execute\n");
  // auto start = std::chrono::steady_clock::now();

  prim.execute(strm, args);
  strm.wait();

  // auto end = std::chrono::steady_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
  // float t = float(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
  // printf("time: %.3f ms\n", t);

  // printf("matmul prim execute over\n");
  return output;
}

// linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
// nested_linear_backward
std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
  const at::Tensor& self, 
  const at::Tensor& grad_output,
  const at::Tensor& weight,
  at::Tensor& grad_input, 
  at::Tensor& grad_weight, 
  at::Tensor& grad_bias, 
  bool* output_mask) {

  if (!grad_output.defined()) {
    grad_input = at::Tensor();
    grad_weight = at::Tensor();
    grad_bias = at::Tensor();
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight, grad_bias};
  }
  // since Y = X · W^T
  if (output_mask[0]) {
    // dX = dY · W
    grad_input = oplifter::linear(grad_output, weight, grad_input, c10::optional<at::Tensor>());
  }
  if (output_mask[1]) {
    // dW = dY^T · X
    grad_weight = oplifter::linear(grad_output.t(), self, grad_weight, c10::optional<at::Tensor>());
  }
  if (output_mask[2]) {
    grad_bias = grad_output.sum(0);
  }

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace oplifter
