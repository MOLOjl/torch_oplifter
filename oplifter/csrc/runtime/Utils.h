// #pragma once

#include <ATen/Config.h>
#include <c10/core/Device.h>
#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

#include <iostream>
#include <oneapi/dnnl/dnnl.hpp>

// #include "csrc/runtime/Runtime.h"

using namespace dnnl;

namespace oplifter {

static inline memory::dims get_onednn_dims(const at::Tensor& tensor) {
  memory::dims dims;
  for (int i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

static inline memory::dims get_onednn_strides(const at::Tensor& tensor) {
  memory::dims strides;
  for (int i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

static inline memory::data_type get_onednn_dtype(
    const at::ScalarType scalar_dtype,
    bool allow_undef = false) {
  switch (scalar_dtype) {
    case at::ScalarType::Byte:
      return memory::data_type::u8;
    case at::ScalarType::Char:
      return memory::data_type::s8;
    case at::ScalarType::QInt8:
      return memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return memory::data_type::u8;
    case at::ScalarType::Int:
      return memory::data_type::s32;
    case at::ScalarType::Long:
      return memory::data_type::s64;
    case at::ScalarType::Half:
      return memory::data_type::f16;
    case at::ScalarType::Float:
      return memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return memory::data_type::bf16;
    case at::ScalarType::Float8_e4m3fn:
      return memory::data_type::f8_e4m3;
    case at::ScalarType::Float8_e5m2:
      return memory::data_type::f8_e5m2;
    default:
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(scalar_dtype),
            " is not supported in oneDNN!");
      }
      return memory::data_type::undef;
  };
}

static inline memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,
    bool allow_undef = false) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Byte:
      return memory::data_type::u8;
    case at::ScalarType::Char:
      return memory::data_type::s8;
    case at::ScalarType::QInt8:
      return memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return memory::data_type::u8;
    case at::ScalarType::Int:
      return memory::data_type::s32;
    case at::ScalarType::Long:
      return memory::data_type::s64;
    case at::ScalarType::Half:
      return memory::data_type::f16;
    case at::ScalarType::Float:
      return memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return memory::data_type::bf16;
    case at::ScalarType::Float8_e4m3fn:
      return memory::data_type::f8_e4m3;
    case at::ScalarType::Float8_e5m2:
      return memory::data_type::f8_e5m2;
    default:
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(tensor.scalar_type()),
            " is not supported in oneDNN!");
      }
      return memory::data_type::undef;
  };
}

static inline memory::format_tag get_dnnl_default_format(
    int ndims,
    bool is_channels_last = false,
    bool allow_undef = false) {
  switch (ndims) {
    case 1:
      return memory::format_tag::a;
    case 2:
      return memory::format_tag::ab;
    case 3:
      return is_channels_last ? memory::format_tag::acb
                              : memory::format_tag::abc;
    case 4:
      return is_channels_last ? memory::format_tag::acdb
                              : memory::format_tag::abcd;
    case 5:
      return is_channels_last ? memory::format_tag::acdeb
                              : memory::format_tag::abcde;
    case 6:
      return memory::format_tag::abcdef;
    case 7:
      return memory::format_tag::abcdefg;
    case 8:
      return memory::format_tag::abcdefgh;
    case 9:
      return memory::format_tag::abcdefghi;
    case 10:
      return memory::format_tag::abcdefghij;
    case 11:
      return memory::format_tag::abcdefghijk;
    case 12:
      return memory::format_tag::abcdefghijkl;
    default:
      if (!allow_undef) {
        TORCH_CHECK(false, "oneDNN doesn't support tensor dimension > 12");
      }
      return memory::format_tag::undef;
  }
}

} // namespace oplifter