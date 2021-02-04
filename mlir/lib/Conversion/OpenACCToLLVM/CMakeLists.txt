add_mlir_conversion_library(MLIROpenACCToLLVM
  OpenACCToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToLLVM

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIROpenACC
  MLIRStandardToLLVM
  MLIRTransforms
  )
