//===- Passes.h - OpenACC pass entry points ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_PASSES_H
#define MLIR_DIALECT_OPENACC_PASSES_H

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenACCToGPUPass();

std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenACCToStandardPass();

} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_PASSES_H
