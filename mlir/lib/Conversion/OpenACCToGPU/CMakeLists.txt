add_mlir_conversion_library(MLIROpenACCToGPU
  ConvertOpenACCToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToGPU

  DEPENDS
  MLIROpenACCPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIROpenACC
  MLIRTransforms
  )