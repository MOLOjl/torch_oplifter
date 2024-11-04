#include "csrc/operators/_Ops.h"

using namespace dnnl;

namespace oplifter {

std::tuple<Tensor, Tensor, Tensor> layer_norm(
    const Tensor& src,
    const Tensor& wgh,
    const Tensor& bia,
    double epsilon) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  // FP16 Data Type only support forward_inference
  bool training = src.scalar_type() == ScalarType::Half ? false : true;
  auto prop =
      training ? prop_kind::forward_training : prop_kind::forward_inference;
  normalization_flags flags = wgh.defined() && bia.defined()
      ? normalization_flags::use_scale | normalization_flags::use_shift
      : normalization_flags::none;
  bool useScaleShift = (bool)(flags & normalization_flags::use_scale);

  int32_t n, ic, ih;
  memory::dims tz, st, stats_tz;
  memory::format_tag stats_fmt;
  if (src.ndimension() == 3) {
    n = src.size(0);
    ic = src.size(1);
    ih = src.size(2);
    tz = {n, ic, ih};
    st = {src.stride(0), src.stride(1), src.stride(2)};
    stats_tz = {n, ic};
    stats_fmt = memory::format_tag::ab;
  } else {
    ic = src.size(0);
    ih = src.size(1);
    tz = {ic, ih};
    st = {src.stride(0), src.stride(1)};
    stats_tz = {ic};
    stats_fmt = memory::format_tag::a;
  }

  memory::data_type dt = get_onednn_dtype(src);
  memory::data_type stats_dt = memory::data_type::f32;

  auto md = memory::desc({tz}, dt, {st});
  auto stats_md = memory::desc(stats_tz, stats_dt, stats_fmt);
  auto dst = at::empty_like(src, src.options());

  auto src_m = dpcpp_onednn_memory(md, engine, src.data_ptr());
  auto dst_m = dpcpp_onednn_memory(md, engine, dst.data_ptr());

  primitive_attr pattr;

  auto ln_fwd_pd = training
      ? layer_normalization_forward::primitive_desc(
            engine, prop, md, md, stats_md, epsilon, flags, pattr)
      : layer_normalization_forward::primitive_desc(
            engine, prop, md, md, epsilon, flags, pattr);

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DST, dst_m},
  };

  at::Tensor mean, rstd;
  auto stats_exp_md = ln_fwd_pd.mean_desc();
  if (training) {
    auto stats_usr_md = memory::desc(stats_tz, stats_dt, stats_fmt);
    mean = at::empty(stats_tz, src.options().dtype(at::kFloat));
    rstd = at::empty(stats_tz, src.options().dtype(at::kFloat));

    auto mean_memory =
        dpcpp_onednn_memory(stats_exp_md, engine, mean.data_ptr());
    auto var_memory =
        dpcpp_onednn_memory(stats_exp_md, engine, rstd.data_ptr());

    args.insert({DNNL_ARG_MEAN, mean_memory});
    args.insert({DNNL_ARG_VARIANCE, var_memory});
  }

  at::Tensor wgh_f32 = wgh;
  at::Tensor bia_f32 = bia;
  if (useScaleShift) {
    if (wgh.scalar_type() == at::ScalarType::Half ||
        wgh.scalar_type() == at::ScalarType::BFloat16) {
      wgh_f32 = wgh.to(at::kFloat);
    }

    if (bia.scalar_type() == at::ScalarType::Half ||
        bia.scalar_type() == at::ScalarType::BFloat16) {
      bia_f32 = bia.to(at::kFloat);
    }

    auto scl_m = dpcpp_onednn_memory(
        ln_fwd_pd.weights_desc(), engine, wgh_f32.data_ptr());
    auto sft_m = dpcpp_onednn_memory(
        ln_fwd_pd.weights_desc(), engine, bia_f32.data_ptr());
    args.insert({DNNL_ARG_SCALE, scl_m});
    args.insert({DNNL_ARG_SHIFT, sft_m});
  }

  auto ln_fwd = layer_normalization_forward(ln_fwd_pd);

  DPCPP_ONEDNN_EXEC(ln_fwd, strm, args);
  return std::make_tuple(dst, mean, rstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward(
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& wgh,
    double epsilon) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  normalization_flags flags = wgh.defined()
      ? normalization_flags::use_scale | normalization_flags::use_shift
      : normalization_flags::none;
  bool useScaleShift = (bool)(flags & normalization_flags::use_scale);

  int32_t n, ic, ih;
  memory::dims src_tz, src_st, diff_dst_tz, diff_dst_st, stats_tz;
  memory::format_tag stats_fmt;
  if (src.ndimension() == 3) {
    n = src.size(0);
    ic = src.size(1);
    ih = src.size(2);
    src_tz = {n, ic, ih};
    src_st = {src.stride(0), src.stride(1), src.stride(2)};
    diff_dst_tz = {n, ic, ih};
    diff_dst_st = {diff_dst.stride(0), diff_dst.stride(1), diff_dst.stride(2)};
    stats_tz = {n, ic};
    stats_fmt = memory::format_tag::ab;
  } else {
    ic = src.size(0);
    ih = src.size(1);
    src_tz = {ic, ih};
    src_st = {src.stride(0), src.stride(1)};
    diff_dst_tz = {ic, ih};
    diff_dst_st = {diff_dst.stride(0), diff_dst.stride(1)};
    stats_tz = {ic};
    stats_fmt = memory::format_tag::a;
  }

  memory::data_type dt = get_onednn_dtype(src);
  memory::data_type stats_dt = memory::data_type::f32;

  auto src_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(src);
  auto diff_dst_ctx =
      at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(diff_dst);
  auto mean_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(mean);
  auto rstd_ctx = at::AtenIpexTypeXPU::DPCPPTensorContext::get_tensor_ctx(rstd);

  auto src_md = memory::desc({src_tz}, dt, {src_st});
  auto diff_dst_md = memory::desc({diff_dst_tz}, dt, {diff_dst_st});
  auto exp_md = diff_dst_md;
  auto mean_md = memory::desc({stats_tz}, stats_dt, stats_fmt);
  auto rstd_md = memory::desc({stats_tz}, stats_dt, stats_fmt);

  primitive_attr pattr;
  auto ln_fwd_pd = layer_normalization_forward::primitive_desc(
      engine,
      dnnl::prop_kind::forward_training,
      exp_md,
      exp_md,
      epsilon,
      flags,
      pattr);

  auto ln_bwd_pd = layer_normalization_backward::primitive_desc(
      engine,
      dnnl::prop_kind::backward,
      exp_md,
      exp_md,
      exp_md,
      mean_md,
      epsilon,
      flags,
      ln_fwd_pd,
      pattr);

  dnnl::memory src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  dnnl::memory diff_dst_m =
      dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

  at::Tensor diff_src = at::empty_like(src);
  dnnl::memory diff_src_m = dpcpp_onednn_memory(exp_md, engine, diff_src.data_ptr());

  auto stats_exp_md = ln_bwd_pd.mean_desc();
  dnnl::memory mean_m = dpcpp_onednn_memory(mean_md, engine, mean.data_ptr());
  dnnl::memory rstd_m = dpcpp_onednn_memory(rstd_md, engine, rstd.data_ptr());

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DIFF_DST, diff_dst_m},
      {DNNL_ARG_MEAN, mean_m},
      {DNNL_ARG_VARIANCE, rstd_m},
      {DNNL_ARG_DIFF_SRC, diff_src_m},
  };

  at::Tensor wgh_f32, bia_f32, diff_wgh, diff_bia;
  if (useScaleShift) {
    wgh_f32 = wgh.to(at::kFloat);
    auto wgh_m = dpcpp_onednn_memory(
        ln_bwd_pd.weights_desc(), engine, wgh_f32.data_ptr());

    bia_f32 = at::empty_like(wgh_f32);
    auto bia_m = dpcpp_onednn_memory(
        ln_bwd_pd.weights_desc(), engine, bia_f32.data_ptr());

    diff_wgh = at::empty(wgh.sizes(), wgh.options().dtype(ScalarType::Float));
    diff_bia = at::empty(wgh.sizes(), wgh.options().dtype(ScalarType::Float));

    auto diff_wgh_m = dpcpp_onednn_memory(
        ln_bwd_pd.diff_weights_desc(), engine, diff_wgh.data_ptr());

    auto diff_bia_m = dpcpp_onednn_memory(
        ln_bwd_pd.diff_weights_desc(), engine, diff_bia.data_ptr());

    args.insert({DNNL_ARG_SCALE, wgh_m});
    args.insert({DNNL_ARG_SHIFT, bia_m});
    args.insert({DNNL_ARG_DIFF_SCALE, diff_wgh_m});
    args.insert({DNNL_ARG_DIFF_SHIFT, diff_bia_m});
  }

  auto ln_backward = layer_normalization_backward(ln_bwd_pd);

  ln_backward.execute(strm, args);
  return std::make_tuple(diff_src, diff_wgh, diff_bia);
}

} // namespace oplifter
