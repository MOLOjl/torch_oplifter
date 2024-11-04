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

import oplifter_c

device = oplifter_c.custom_device()
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
  kk = torch.zeros(2, 3, device=device)
  kk = kk.to("cpu")
  print(kk)
  zz = torch.ones(2, 3, device=device)
  zz = zz.to("cpu")
  print(zz)

def test_arange():
  aa = torch.arange(0, 26, 2)
  bb = torch.arange(13, device=device, dtype=torch.float32)

  aa = aa.to("cpu")
  bb = bb.to("cpu")
  print(aa)
  print(aa.dtype)
  print(bb)

# freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
def test_as_strided():
  n = 25
  print("run test_as_strided")
  aa = torch.arange(0, n, 2.01)
  dd = aa.to("cpu")
  print(dd.dtype)
  print(dd)

  bb = aa[: (n // 2)]
  cc = bb.float()
  ee = cc.to("cpu")
  print(ee.dtype)
  print(ee)

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
  oplifter_c.sync_device()
  
  res = []

  for i in range(4):
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
# test_matmul_f16_time()
torch.set_default_device(device)
# torch.set_default_dtype(torch.half)
torch.set_default_dtype(torch.bfloat16)
test_as_strided()

print("over")

