set(LLVM_TARGET_DEFINITIONS OpenACC.td)
mlir_tablegen(OpenACC.h.inc -gen-op-decls)
mlir_tablegen(OpenACC.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIROpenACCIncGen)

