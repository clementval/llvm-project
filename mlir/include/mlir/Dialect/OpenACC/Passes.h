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

#include "mlir/Pass/Pass.h"

namespace mlir {
    std::unique_ptr<OpPassBase<mlir::ModuleOp>> createConvertOpenACCToGPUPass();
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_PASSES_H
