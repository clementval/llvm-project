//===- OpenACC.cpp - OpenACC to GPU dialect conversion --------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// This file implements a pass to convert the MLIR OpenACC dialect into the
// GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACC/ConvertOpenACCToGPU.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "llvm/ADT/StringMap.h"

#include <llvm/Support/raw_ostream.h>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <iostream>

using namespace mlir;

struct OpenACCLoopEmptyConstructFolder : public OpRewritePattern<acc::LoopOp> {
    using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
                                       PatternRewriter &rewriter) const override {
        // Check that the body only contains a terminator.
        if (!has_single_element(loopOp.getBody()))
            return matchFailure();
        rewriter.eraseOp(loopOp);
        return matchSuccess();
    }
};

void
acc::LoopOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context) {
    patterns.insert<OpenACCLoopEmptyConstructFolder>(context);
}
