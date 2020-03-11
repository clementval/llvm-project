//===- OpenACC.h - MLIR OpenACC Dialect -------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ============================================================================
//
// This file defines the OpenACC related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_DIALECT_H
#define MLIR_DIALECT_OPENACC_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace acc {

class OpenACCDialect : public Dialect {
public:
  explicit OpenACCDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "acc"; }

  static StringRef getCollapseAttrName() { return "collapse"; }

  static StringRef getAsyncAttrName() { return "async"; }
};

enum OpenACCExecMapping { NONE = 0, VECTOR = 1, WORKER = 2, GANG = 4, GANG_VECTOR = 5 };

#define GET_OP_CLASSES

#include "mlir/Dialect/OpenACC/OpenACC.h.inc"

} // end namespace acc
} // end namespace mlir

#endif // MLIR_DIALECT_OPENACC_DIALECT_H
