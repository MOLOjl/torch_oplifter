// #pragma once
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/record_function.h>

#include "csrc/runtime/Runtime.h"
#include "csrc/runtime/Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;

namespace oplifter {

at::Tensor bin(dnnl::algorithm algo, at::Tensor& dst, const at::Tensor& t1, const at::Tensor& t2);

void eltwise(algorithm alg_kind, at::Tensor& dst, const at::Tensor& src, float alpha = 0, float beta = 0);

at::Tensor softmax(
    const at::Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    dnnl::algorithm softmax_algo,
    at::Tensor& output);

at::Tensor linear(
  const at::Tensor& input, 
  const at::Tensor& weight, 
  at::Tensor& output, 
  const c10::optional<at::Tensor>& bias_opt = c10::nullopt);

at::Tensor matmul(
  const at::Tensor& input, 
  const at::Tensor& weight, 
  at::Tensor& output);

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_backward(
  const at::Tensor& self, 
  const at::Tensor& grad_output,
  const at::Tensor& weight,
  at::Tensor& grad_input, 
  at::Tensor& grad_weight, 
  at::Tensor& grad_bias, 
  bool* output_mask);

void fill_tensor(
    at::Tensor& t_out, 
    const at::Scalar & value);

void arange_step(
    at::Tensor& t_out, 
    const at::Scalar & start, 
    const at::Scalar & end, 
    const at::Scalar & step);
}