add_mlir_conversion_library(MLIROpenACCToGPU
  ConvertOpenACCToGPU.cpp
  ConvertOpenACCToTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToGPU
)
add_dependencies(
  MLIROpenACCToGPU

  MLIROpenACC
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIROpenACCToGPU

  MLIROpenACC
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
