add_llvm_library(MLIROpenACC
        OpenACC.cpp
        DialectRegistration.cpp
        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC
        )
add_dependencies(MLIRAffineOps
        MLIROpenACCIncGen
        MLIRIR
        MLIRStandardOps)
target_link_libraries(MLIROpenACC MLIRIR MLIRStandardOps)

