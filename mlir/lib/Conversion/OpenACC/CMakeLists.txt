set(LLVM_TARGET_DEFINITIONS OpenACCToGPU.td)
mlir_tablegen(OpenACCToGPU.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROpenACCToGPUIncGen)

add_llvm_library(MLIROpenACCPasses
  OpenACCToGPUPass.cpp
  OpenACCToSeqPass.cpp
)

add_dependencies(MLIROpenACCPasses MLIROpenACCToGPUIncGen)
target_link_libraries(MLIROpenACCPasses
  MLIRGPU
  MLIROpenACC
  MLIRLLVMIR
  MLIRPass
)

