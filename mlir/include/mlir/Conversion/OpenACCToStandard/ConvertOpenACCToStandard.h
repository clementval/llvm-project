//===- ConvertOpenACCToSeq.h - Convert OpenACC to Standard ops --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Provides patterns to convert from OpenACC structure ops to standard ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H
#define MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
template <typename T>
class OpPassBase;

std::unique_ptr<OpPassBase<ModuleOp>> createConvertOpenACCToSeqPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOSEQ_H
