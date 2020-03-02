add_mlir_conversion_library(MLIROpenACCToStandard
  ConvertOpenACCToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MLIROpenACCToStandard
)
add_dependencies(
  MLIROpenACCToStandard

  MLIROpenACC
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIROpenACCToStandard

  MLIROpenACC
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
