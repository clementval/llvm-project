//===- OpenACCToSeqPass.cpp - Convert OpenACC Dialect to Standard Ops -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// This file implements a pass to convert MLIR OpenACC Dialect into the Standard
// ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#include "mlir/Pass/Pass.h"
#define GEN_PASS_CLASSES
#include "mlir/Dialect/OpenACC/Passes.h.inc"



struct OpenACCToStandardPass
    : public OpenACCToStandardBase<OpenACCToStandardPass> {

  void runOnOperation() override;
};

template <typename TerminatorOp>
struct TerminatorOpLowering final : public OpRewritePattern<TerminatorOp> {
  using OpRewritePattern<TerminatorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TerminatorOp terminatorOp,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(terminatorOp);
    return success();
  }
};

template <typename StructureOp>
static void extractOperationsForSequential(StructureOp baseOp) {
  SmallVector<Operation *, 8> toHoist;
  for (Operation &op : baseOp.getOperation()
                           ->getRegion(0)
                           .getBlocks()
                           .front()
                           .getOperations()) {
    if (&op == baseOp.getOperation()) {
      continue;
    } else {
      toHoist.push_back(&op);
    }
  }

  for (auto *op : toHoist) {
    op->moveBefore(baseOp.getOperation());
  }
}

/// Convert the OpenACC construct to run program in a sequential manner.
static void convertToSequential(ModuleOp m) {
  m.walk([&](acc::ParallelOp parallelOp) {
    parallelOp.walk([&](acc::LoopOp loopOp) {
      extractOperationsForSequential(loopOp);
      loopOp.erase();
    });
    extractOperationsForSequential(parallelOp);
    parallelOp.erase();
  });
}

void OpenACCToStandardPass::runOnOperation() {

  ConversionTarget target(getContext());
  target.addIllegalDialect<acc::OpenACCDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  // If operation is considered legal the rewrite pattern in not called.
  OwningRewritePatternList patterns;
  patterns.insert<TerminatorOpLowering<acc::ParallelEndOp>>(&getContext());
  patterns.insert<TerminatorOpLowering<acc::YieldOp>>(&getContext());

  ModuleOp m = getOperation();
  convertToSequential(m);

  if (failed(applyPartialConversion(m, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
    mlir::createConvertOpenACCToStandardPass() {
  return std::make_unique<OpenACCToStandardPass>();
}
