#include "csrc/operators/_Ops.h"

using namespace dnnl;
namespace oplifter {

std::pair<int, int> findSwappedIndices(const std::vector<int64_t>& array) {
  int n = array.size();
  int x = -1, y = -1;

  for (int i = 0; i < n - 1; ++i) {
    if (array[i] < array[i + 1]) {
      x = i;
      break;
    }
  }

  for (int i = n - 1; i > 0; --i) {
    if (array[i] > array[i - 1]) {
      y = i;
      break;
    }
  }

  // If no inversion was find, both x and y will be -1.
  return {x, y};
}

// contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
at::Tensor contiguous(
    Tensor& self,
    Tensor& output,
    c10::MemoryFormat memory_format) {    
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous, "Contiguous only for now.");

  auto engine =
      GpuEngineManager::Instance().get_engine({c10::DeviceType::PrivateUse1, dpcppGetCurDevice()});
  auto strm = GpuStreamManager::Instance().get_stream();

  memory::dims self_dims = get_onednn_dims(self);
  memory::dims self_strides = get_onednn_strides(self);

  memory::dim idx1, idx2;
  auto inversion_pair = findSwappedIndices(self_strides);
  if(inversion_pair.first = -1)
    return self;
  
  idx1 = inversion_pair.first;
  idx2 = inversion_pair.second;

  memory::dims input_dims = self_dims;
  memory::dims input_strides(input_dims.size(), 1);
  self_strides[input_dims.size()-1] = 1;
  for(i=input_dims.size()-2; i>-1; i--) {
    self_strides[i] = self_strides[i+1]*self_dims[i+1];
    input_strides[i] = input_strides[i+1]*input_dims[i+1];
  }

  input_dims[idx1] = self_dims[idx2];
  input_dims[idx2] = self_dims[idx1];

  auto input_md = memory::desc(input_dims, get_onednn_dtype(input), input_strides);
  auto output_md = memory::desc(self_dims, get_onednn_dtype(input), self_strides);
  
  if (!output.defined()) {
    output = at::empty(self.sizes(), self.options());
  }

  auto input_mem = dpcpp_onednn_memory(input_md, engine, self.data_ptr());
  auto output_mem = dpcpp_onednn_memory(output_md, engine, output.data_ptr());

  auto pd = dnnl::transpose::primitive_desc(engine, input_md, output_md, idx1, idx2);
  auto prim = dnnl::transpose(pd);

  std::unordered_map<int, memory> args;
  args.insert({DNNL_ARG_SRC, input_mem});
  args.insert({DNNL_ARG_DST, output_mem});

  printf("contiguous execute\n");
  prim.execute(strm, args);
  printf("contiguous execute over\n");

  return output;
}

} // namespace oplifter
