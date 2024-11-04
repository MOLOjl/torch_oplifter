cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-w)

# 项目名称
project(torch_oplifter)

# 查找 Python 解释器、库和头文件
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# 添加 Python 头文件的搜索路径
include_directories(${Python3_INCLUDE_DIRS})
# message(STATUS "python include:${Python3_INCLUDE_DIRS}")

# 尝试使用 Python 查找 PyTorch 的 CMake 配置文件路径
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 将 PyTorch 的 CMake 配置文件路径添加到 CMake 的搜索路径
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
message(STATUS "PyTorch include directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch library directories: ${TORCH_LIBRARY_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 枚举源文件
set(CSRC_ROOT "${PROJECT_SOURCE_DIR}/csrc")

set(OPS_SRCS)
add_subdirectory(${CSRC_ROOT}/operators)

set(RT_SRCS)
add_subdirectory(${CSRC_ROOT}/runtime)

set(OPLIFTER_SRCS
  ${CSRC_ROOT}/impl/oplifter_impl.cpp
  ${OPS_SRCS}
  ${RT_SRCS}
)

# 源代码中的头文件
include_directories(${PROJECT_SOURCE_DIR})

# 添加库
add_library(oplifter_C SHARED ${OPLIFTER_SRCS})

# 编译选项
target_compile_definitions(oplifter_C PUBLIC 
  USE_ROCM __HIP_PLATFORM_AMD__)

if(NOT SYCL_PATH)
  message(WARNING "SYCL_PATH not provided, will use '~/sycl_ws/llvm_install/release' by default, you can define it like -DSYCL_PATH_PATH=xxx")
  set(SYCL_PATH "~/sycl_ws/llvm_install/release")
endif()

if(NOT ONEDNN_PATH)
  message(WARNING "ONEDNN_PATH not provided, will use '~/sycl_ws/llvm_install' by default, you can define it like -DONEDNN_PATH=xxx")
  set(ONEDNN_PATH "~/sycl_ws/onednn_install")
endif()

# message(FATAL_ERROR "${SYCL_PATH}/include")

# third_party 头文件目录
target_include_directories(
  oplifter_C PUBLIC
  ${SYCL_PATH}/include 
  ${SYCL_PATH}/include/sycl 
  ${ONEDNN_PATH}/include 
)

# 添加动态链接库
target_link_libraries(
  oplifter_C PUBLIC
  ${Python3_LIBRARIES} 
  ${TORCH_LIBRARIES} 
  ${SYCL_PATH}/lib/libsycl.so 
  ${ONEDNN_PATH}/lib/libdnnl.so
)