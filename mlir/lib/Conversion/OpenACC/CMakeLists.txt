set(LLVM_TARGET_DEFINITIONS OpenACCToGPU.td)
mlir_tablegen(OpenACCToGPU.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROpenACCToGPUIncGen)

set(LIBS
    MLIROpenACC
    MLIRAffineOps
    MLIRGPU
    MLIRIR
    MLIRLinalg
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    LLVMSupport
    )

add_llvm_library(MLIROpenACCPasses
    OpenACCToGPU.cpp
    OpenACCToGPUPass.cpp
    OpenACCToSeqPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC/
    )

add_dependencies(MLIROpenACCPasses MLIROpenACCToGPUIncGen)
add_dependencies(MLIROpenACCPasses ${LIBS})
target_link_libraries(MLIROpenACCPasses ${LIBS})

