add_mlir_dialect_library(MLIROpenACC
  IR/OpenACC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC

  DEPENDS
  MLIROpenACCIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)