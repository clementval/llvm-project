add_mlir_conversion_library(MLIROpenACCToStandard
  ConvertOpenACCToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MLIROpenACCToStandard


  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIROpenACC
  MLIRTransforms
)
