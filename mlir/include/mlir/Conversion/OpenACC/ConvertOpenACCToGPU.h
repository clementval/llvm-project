//===- ConvertOpenACCToGPU.h - Convert OpenACC to GPU ops -------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Provides patterns to convert from Stencil structure ops to affine ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOGPU_H
#define MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOGPU_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
    template <typename T> class OpPassBase;
/// Collect a set of patterns to lower from Stencil structure
/// operations (stencil.stage, stencil.do_method etc.) to loop
/// operations within the Affine dialect; in particular, convert
/// abstract stencil descriptions into affine loop nests.
    void populateOpenACCToGPUConversionPatterns(OwningRewritePatternList &patterns,
                                                MLIRContext *ctx);

    std::unique_ptr<OpPassBase<FuncOp>> createOpenACCToGPUPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACC_CONVERTOPENACCTOGPU_H
