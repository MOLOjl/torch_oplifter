#include "csrc/operators/_Ops.h"

using namespace dnnl;
using dnnl::algorithm;

namespace oplifter {

void eltwise(
    algorithm alg_kind,
    at::Tensor& dst,
    const at::Tensor& src,
    float alpha,
    float beta) {

  c10::Device curDevice = c10::Device(c10::DeviceType::PrivateUse1, dpcppGetCurDevice());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  std::vector<int64_t> dims;
  for (size_t i = 0; i < src.dim(); i++) {
    dims.push_back(src.size(i));
  }

  memory::dims src_tz = dims;
  auto data_t = get_onednn_dtype(src);
  auto format_data = get_dnnl_default_format(
      src.dim(),
      src.dim() == 4 ? (!src.is_contiguous() &&
                        src.is_contiguous(at::MemoryFormat::ChannelsLast))
                     : (!src.is_contiguous() &&
                        src.is_contiguous(at::MemoryFormat::ChannelsLast3d)));
  auto src_md = memory::desc({src_tz}, data_t, format_data);


  memory src_memory;
  src_memory = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  primitive_attr attr;
  auto eltwise_forward_pd = eltwise_forward::primitive_desc(
      engine, prop_kind::forward, alg_kind, src_md, src_md, alpha, beta, attr);

  memory dst_memory;
  if (!dst.defined()) {
    dst = src.is_contiguous(at::MemoryFormat::ChannelsLast)
        ? at::empty_like(src, at::MemoryFormat::ChannelsLast)
        : at::empty_like(src);
  }
  dst_memory = dpcpp_onednn_memory(
      eltwise_forward_pd.dst_desc(), engine, dst.data_ptr());
  
  auto strm = GpuStreamManager::Instance().get_stream();
  auto eltwise_fwd = dnnl::eltwise_forward(eltwise_forward_pd);

  
  eltwise_fwd.execute(strm, {{DNNL_ARG_SRC, src_memory}, {DNNL_ARG_DST, dst_memory}});
}

template <algorithm alg_kind>
void eltwise_backward(
    at::Tensor& diff_src,
    const at::Tensor& src_dst,
    const at::Tensor& diff_dst_,
    float alpha = 0,
    float beta = 0) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto data_t = get_onednn_dtype(src_dst);
  std::vector<int64_t> src_dst_dims;
  for (size_t i = 0; i < src_dst.dim(); i++) {
    src_dst_dims.push_back(src_dst.size(i));
  }

  memory::dims src_dst_tz = src_dst_dims;
  auto format_data = get_dnnl_default_format(
      src_dst.dim(),
      src_dst.dim() == 4
          ? (!src_dst.is_contiguous() &&
             src_dst.is_contiguous(at::MemoryFormat::ChannelsLast))
          : (!src_dst.is_contiguous() &&
             src_dst.is_contiguous(at::MemoryFormat::ChannelsLast3d)));
  auto src_dst_md = memory::desc({src_dst_tz}, data_t, format_data);
  auto diff_dst_md = memory::desc({src_dst_tz}, data_t, format_data);
  at::Tensor diff_dst;
  if (src_dst.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    diff_dst = diff_dst_.contiguous(at::MemoryFormat::ChannelsLast);
  } else if (src_dst.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
    diff_dst = diff_dst_.contiguous(at::MemoryFormat::ChannelsLast3d);
  } else {
    diff_dst = diff_dst_;
  }

  auto src_dst_memory = dpcpp_onednn_memory(src_dst_md, engine, src_dst.data_ptr());
  auto diff_dst_memory = dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

  primitive_attr attr;

  auto eltwise_forward_pd = eltwise_forward::primitive_desc(
      engine,
      prop_kind::forward_training,
      alg_kind,
      src_dst_md,
      src_dst_md,
      alpha,
      beta,
      attr);

  auto eltwise_backward_pd = eltwise_backward::primitive_desc(
      engine,
      alg_kind,
      diff_dst_md,
      src_dst_md,
      src_dst_md,
      alpha,
      beta,
      eltwise_forward_pd,
      attr);

  memory diff_src_memory;

  if (!diff_src.defined()) {
    if (src_dst.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      diff_src = at::empty_like(src_dst, at::MemoryFormat::ChannelsLast);
    } else {
      diff_src = at::empty_like(src_dst);
    }
  }
  auto diff_src_md = memory::desc({src_dst_tz, data_t, format_data});
  diff_src_memory = dpcpp_onednn_memory(diff_src_md, engine, diff_src.data_ptr());
  
  auto eltwise_bwd = dnnl::eltwise_backward(eltwise_backward_pd);

  if (alg_kind == algorithm::eltwise_logistic_use_dst_for_bwd) {
    eltwise_bwd.execute(
    strm,
    {
        {DNNL_ARG_DST, src_dst_memory},
        {DNNL_ARG_DIFF_DST, diff_dst_memory},
        {DNNL_ARG_DIFF_SRC, diff_src_memory},

    });
  } else {
    eltwise_bwd.execute(
    strm,
    {
        {DNNL_ARG_SRC, src_dst_memory},
        {DNNL_ARG_DIFF_DST, diff_dst_memory},
        {DNNL_ARG_DIFF_SRC, diff_src_memory},
    });
  }
}

} // namespace oplifter
