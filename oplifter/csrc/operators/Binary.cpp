#include "csrc/operators/_Ops.h"
#include "csrc/runtime/raw_mem_manager.h"

using namespace dnnl;

namespace oplifter {
  
at::Tensor bin(
    dnnl::algorithm algo,
    at::Tensor& dst,
    const at::Tensor& t1,
    const at::Tensor& t2) {
  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();
  // TODO: broadcast
  auto tar_md = memory::desc(get_onednn_dims(t1), get_onednn_dtype(t1), get_onednn_strides(t1));

  auto md1 = memory::desc(get_onednn_dims(t1), get_onednn_dtype(t1), get_onednn_strides(t1));
  auto md2 = memory::desc(get_onednn_dims(t2), get_onednn_dtype(t2), get_onednn_strides(t2));

  auto m1 = dpcpp_onednn_memory(md1, engine, t1.data_ptr());
  auto m2 = dpcpp_onednn_memory(md2, engine, t2.data_ptr());

  primitive_attr attr;

  if (t1.is_quantized()) {
    throw "not supported";
    float t1_scale = t1.q_scale();
    float t2_scale = t2.q_scale();
    attr.set_scales_mask(DNNL_ARG_SRC_0, 0);
    attr.set_scales_mask(DNNL_ARG_SRC_1, 0);
  }

  if (!dst.defined()) {
    dst = at::empty_like(t1, t1.suggest_memory_format());
  }
  auto mo = dpcpp_onednn_memory(tar_md, engine, dst.data_ptr());

  binary::primitive_desc pd;
  pd = binary::primitive_desc(engine, algo, md1, md2, tar_md);

  auto prim = binary(pd);

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC_0, m1});
  args.insert({DNNL_ARG_SRC_1, m2});
  args.insert({DNNL_ARG_DST, mo});

  prim.execute(strm, args);
  strm.wait();

  // std::vector<float> dst_data(3*16);
  // oplifter::raw_mem_copy(dst_data.data(), mo.get_data_handle(), dst_data.size()*sizeof(float), 2);
  // printf("\n dst: \n");
  // for(int i=0; i<4; i++)
  //   printf("%.2f, ", dst_data[i]);
  // printf("\n");

  return dst;
}

} // namespace oplifter