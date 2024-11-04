#include "csrc/operators/_Ops.h"
#include "csrc/runtime/raw_mem_manager.h"

using namespace dnnl;

namespace oplifter {
  
void arange_step(
    at::Tensor& t_out, 
    const at::Scalar& start, 
    const at::Scalar& end, 
    const at::Scalar& step) {
  TORCH_CHECK(t_out.defined(), "Tensor must be defined.");
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  auto t_out_md = memory::desc(get_onednn_dims(t_out), get_onednn_dtype(t_out), get_onednn_strides(t_out));
  auto t_out_mem = dpcpp_onednn_memory(t_out_md, engine, t_out.data_ptr());

  double vd[3] = {start.toDouble(), end.toDouble(), step.toDouble()};
  int64_t vi[3] = {start.toLong(), end.toLong(), step.toLong()};
  bool vb[3] = {start.toBool(), end.toBool(), step.toBool()};

  std::tuple<double*, int64_t*, bool*> value_3(vd, vi, vb);

  tsop::primitive_desc pd;
  dnnl::algorithm algo = dnnl::algorithm::tsop_arange;
  pd = tsop::primitive_desc(engine, algo, t_out_md, t_out_md, value_3);

  auto prim = tsop(pd);

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, t_out_mem});
  args.insert({DNNL_ARG_DST, t_out_mem});

  prim.execute(strm, args);
  strm.wait();
}

} // namespace oplifter