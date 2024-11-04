import os
import shutil
import sys
import unittest

import numpy as np
import torch
import torch._dynamo
import torch.utils.cpp_extension
import torch.nn as nn
from torch.nn.parameter import Parameter

from oplifter.extension_codegen_backend import (
  ExtensionScheduling,
  ExtensionWrapperCodegen,
)

from torch._inductor import metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)

# run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code

# def remove_build_path():
#   if sys.platform == "win32":
#     # Not wiping extensions build folder because Windows
#     return
#   default_build_root = torch.utils.cpp_extension.get_default_build_root()
#   if os.path.exists(default_build_root):
#     shutil.rmtree(default_build_root, ignore_errors=True)

# remove_build_path()

os.environ['CXX'] = 'clang++'
os.environ['CC'] = 'clang'

# source_file_path = os.path.dirname(os.path.abspath(__file__))
# source_file = os.path.join(
#     source_file_path, "oplifter/src/extension_device.cpp"
# )

module = torch.utils.cpp_extension.load(
  name="oplifter_ex",
  sources=[], 
  extra_ldflags=['/home/pangyunfei/sycl_ws/torch-oplifter/oplifter/lib/liboplifter_ex.so'],
  is_python_module=False
)

# 假设动态库是 my_extension.so，并且在当前工作目录下
# extension = load(
#     name='my_extension',           # 模块名称
#     sources=[],                    # 这里可以留空，因为我们已经有了编译好的库
#     extra_ldflags=['./my_extension.so'],  # 指定动态库的路径
#     is_python_module=False         # 如果库是 Python 模块，则为 True，否则为 False
# )

# module = torch.utils.cpp_extension.load(
#   name="extension_device",
#   sources=[
#     str(source_file),
#     "oplifter/src/Linear.cpp",
#     "oplifter/src/Softmax.cpp",
#     "oplifter/src/Binary.cpp",
#     "oplifter/src/Eltwise.cpp",
#     "oplifter/src/Utils.cpp",
#     "oplifter/src/Runtime.cpp"
#   ],
#   build_directory="oplifter/bpath",
#   extra_cflags=["-g", "-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__",
#                 "-I/home/pangyunfei/sycl_ws/torch-oplifter/oplifter/include", 
#                 "-I/home/pangyunfei/sycl_ws/onednn_install/include",
#                 "-I/home/pangyunfei/sycl_ws/llvm_install/release/include",
#                 "-I/home/pangyunfei/sycl_ws/llvm_install/release/include/sycl"],
#   extra_ldflags=["-ldnnl", "-lsycl", "-fPIC", 
#                  "-L/home/pangyunfei/sycl_ws/onednn_install/lib", 
#                  "-L/home/pangyunfei/sycl_ws/llvm_install/release/lib"],
#   verbose=False,
# )

torch.utils.rename_privateuse1_backend("oplifter_ex")
register_backend_for_device(
  "oplifter_ex", ExtensionScheduling, ExtensionWrapperCodegen
)
device = module.custom_device()
print("Hi ya, device ", device)

def test_binary():
  x = torch.empty(3, 16).to(torch.float32).fill_(3.14)
  y = torch.empty(3, 16).to(torch.float32).fill_(2.0)

  x = x.to(device)
  y = y.to(device)

  for i in range(5):
    z = x + y
    if(i == 0):
      z = z.to("cpu")
      print(z)

def test_copy():
  jj = torch.empty(3, 16).to(torch.float32)
  jj = jj.to(device)
  jj.fill_(0.56)
  jj = jj.to("cpu")
  print(jj)

def test_linear():
  m = nn.Linear(3, 4)
  input = torch.randn(18)
  for i in range(18):
    input[i] = 1.0
  input = input.view(6, 3)
  out_cpu = m(input)
  # print("weight : ", m.weight)
  # print("bias : ", m.bias)
  # print("output : ", out_cpu)

  m = m.to(device)
  input = input.to(device)

  output = m(input)
  output = output.to("cpu")
  # print(output)
  print(output.size())

def test_matmul():
  # weight = Parameter(torch.empty(4, 3, device=device, dtype=torch.float32))
  weight = torch.randn(16)
  weight = weight.view(4, 4)
  input = torch.randn(4, 4)
  output = torch.matmul(input, weight)
  output = output.to("cpu")
  input = input.to("cpu")
  weight = weight.to("cpu")

  print(input)
  print(weight)
  print(output)

  weight = weight.to(device)
  input = input.to(device)
  output = torch.matmul(input, weight)

  output = output.to("cpu")
  input = input.to("cpu")
  weight = weight.to("cpu")
  print(input)
  print(weight)
  print(output)

def test_matmul_f16():
  # weight = Parameter(torch.empty(4, 3, device=device, dtype=torch.float32))
  weight = torch.randn(16).to(torch.bfloat16)
  weight = weight.view(4, 4)
  input = torch.randn(4, 4).to(torch.bfloat16)
  output = torch.matmul(input, weight)
  output = output.to("cpu")
  input = input.to("cpu")
  weight = weight.to("cpu")

  print(input)
  print(weight)
  print(output)

  weight = weight.to(device)
  input = input.to(device)
  output = torch.matmul(input, weight)

  output = output.to("cpu")
  input = input.to("cpu")
  weight = weight.to("cpu")
  print(input)
  print(weight)
  print(output)

import time
from datetime import datetime
def test_matmul_f16_time():
  len_n = 16384

  # weight = Parameter(torch.empty(4, 3, device=device, dtype=torch.float32))
  weight = torch.randn(len_n, len_n).to(torch.bfloat16)
  input = torch.randn(len_n, len_n).to(torch.bfloat16)
  
  weight = weight.to(device)
  input = input.to(device)
  output = torch.matmul(input, weight)
  # print("all start ==:", time.monotonic())
  module.sync_device()
  
  res = []

  for i in range(10):
    print("---------------------------------------------------")
    start_time = datetime.now()
    s_t = time.monotonic()
    print(time.monotonic())

    output = torch.matmul(input, weight)

    
    # module.sync_device()
    e_t = time.monotonic()
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() * 1000
    print(f"matmul datetime time: {elapsed_time:.2f} ms")
    el_t = e_t - s_t
    # print(f"matmul time time: {el_t*1000:.3f} ms")

    res.append(elapsed_time)

  print("---------------------------------------------------")
  print("---------------------------------------------------")

  print("max: ", np.max(res), ", min: ", np.min(res), ", mean: ", np.mean(res), ", mid: ", np.median(res))
  # module.sync_device()
  # end_time = datetime.now()
  # elapsed_time = (end_time - start_time).total_seconds() * 1000
  # print(f"matmul datetime time: {elapsed_time:.2f} ms")

  print("---------------------------------------------------")
  print("---------------------------------------------------")
  output = output.to("cpu")
  input = input.to("cpu")
  weight = weight.to("cpu")
  # print(input)
  # print(weight)
  # print(output)

def test_softmax():
  m = nn.Softmax(dim=1)
  input = torch.randn(4, 4)
  output = m(input)
  print(input)
  print(output)

  input = input.to(device)
  m = m.to(device)
  output = m(input)
  output = output.to("cpu")
  print(output)

def test_softmax_backward():
  m = nn.Softmax(dim=1)
  input = torch.randn(4, 4, requires_grad=True)
  output = m(input)
  fake_dy = torch.ones_like(output)
  print(input)
  print(output)
  output.backward(fake_dy)
  print(input.grad)

  # input = input.to(device)
  # m = m.to(device)
  # output = m(input)
  # output = output.to("cpu")
  # print(output)

# test_copy()
test_matmul_f16_time()
print("over")

