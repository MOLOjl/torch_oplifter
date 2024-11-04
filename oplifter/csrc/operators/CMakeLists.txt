# FILE(GLOB _OPS_SRCS *.cpp)
FILE(GLOB _OPS_SRCS 
  Binary.cpp
  Eltwise.cpp
  Linear.cpp
  Softmax.cpp
  FillTensor.cpp
  Arange.cpp
  # Transpose.cpp
)


LIST(APPEND OPS_SRCS ${_OPS_SRCS})

# Pass to parent
set(OPS_SRCS ${OPS_SRCS} PARENT_SCOPE)