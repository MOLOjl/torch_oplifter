from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

lib_dirs = []
inc_dirs = []

sycl_dir = os.getenv('SYCL_PATH')
onednn_dir = os.getenv('DNNL_PATH')

if sycl_dir is None:
  raise ValueError("argument 'sycl_dir' is needed.")
if onednn_dir is None:
  raise ValueError("argument 'onednn_dir' is needed.")

print(f"using sycl: {sycl_dir}")
print(f"using onednn: {onednn_dir}")

lib_dirs.append(sycl_dir + '/lib')
inc_dirs.append(sycl_dir + '/include')
inc_dirs.append(sycl_dir + '/include/sycl')
lib_dirs.append(onednn_dir + '/lib')
inc_dirs.append(onednn_dir + '/include')
inc_dirs.append(BASE_DIR)

opllib_dir = os.path.split(os.path.realpath(__file__))[0] + '/build'
lib_dirs.append(opllib_dir)

extra_compile_args = [
  '-std=c++17',
  '-DUSE_ROCM', 
  '-D__HIP_PLATFORM_AMD__'
]

extra_link_args = ['-g']

setup(name='oplifter_c',
  ext_modules=[
    cpp_extension.CppExtension(
      'oplifter_c', 
      sources = ['csrc/oplifter.cpp'],
      library_dirs = lib_dirs,
      libraries = ['sycl', 'dnnl', 'oplifter_C'],
      extra_compile_args = extra_compile_args,
      extra_link_args = extra_link_args,
      # build_directory = os.path.relpath(os.path.join(BASE_DIR, "build/pybuild")),
    )
  ],
  include_dirs = inc_dirs,
  # package_dir={'': os.path.relpath(os.path.join(BASE_DIR, "build/packages"))},
  cmdclass={
    'build_ext': cpp_extension.BuildExtension,
  }
)