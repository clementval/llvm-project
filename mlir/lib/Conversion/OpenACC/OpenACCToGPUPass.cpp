//===- OpenACCToGPUPass.cpp - Convert OpenACC Ops to GPU Ops --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// This file implements a pass to convert MLIR OpenACC ops into the GPU ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/Conversion/OpenACC/ConvertOpenACCToGPU.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

struct OpenACCToGPULoweringPass : public ModulePass<OpenACCToGPULoweringPass> {
  void runOnModule() override;
};

struct ParallelOpLowering final : public OpRewritePattern<acc::ParallelOp> {
  using OpRewritePattern<acc::ParallelOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(acc::ParallelOp parallelOp,
                                     PatternRewriter &rewriter) const override;
};

struct LoopOpLowering final : public OpRewritePattern<acc::LoopOp> {
  using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
                                     PatternRewriter &rewriter) const override;
};

template <typename TerminatorOp>
struct TerminatorOpLowering final : public OpRewritePattern<TerminatorOp> {
  using OpRewritePattern<TerminatorOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TerminatorOp terminatorOp,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(terminatorOp);
    return Pattern::matchSuccess();
  }
};

PatternMatchResult
ParallelOpLowering::matchAndRewrite(acc::ParallelOp parallelOp,
                                    PatternRewriter &rewriter) const {
  // Not used now
  return matchSuccess();
}

PatternMatchResult
LoopOpLowering::matchAndRewrite(acc::LoopOp loopOp,
                                PatternRewriter &rewriter) const {
  // Not used now
  return matchSuccess();
}

template <typename StructureOp>
static void extractOperationsOutsideOfConstruct(StructureOp baseOp) {
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

static FuncOp outlineParallelKernel(acc::ParallelOp parallelOp) {
  auto loc = parallelOp.getLoc();
  std::string parallelKernelName =
      Twine(parallelOp.getParentOfType<FuncOp>().getName(), "_kernel").str();
  Builder builder(parallelOp.getContext());
  FuncOp outlinedFunc = FuncOp::create(
      loc, parallelKernelName, builder.getFunctionType(llvm::None, llvm::None));
  outlinedFunc.getBody().takeBody(parallelOp.getOperation()->getRegion(0));

  OpBuilder opBuilder(parallelOp.getOperation());
  outlinedFunc.walk([&](acc::LoopOp loopOp) {});

  // Add a terminator at then end of the new func
  opBuilder.setInsertionPointToEnd(&outlinedFunc.getBody().back());
  opBuilder.create<mlir::ReturnOp>(loc);

  return outlinedFunc;
}




void OpenACCToGPULoweringPass::runOnModule() {

  ConversionTarget target(getContext());
  target.addIllegalDialect<acc::OpenACCDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  target.addLegalOp<acc::ParallelOp>();
  target.addLegalOp<acc::LoopOp>();

  // If operation is considered legal the rewrite pattern in not called.
  OwningRewritePatternList patterns;
  patterns.insert<TerminatorOpLowering<acc::ParallelEndOp>>(&getContext());
  patterns.insert<TerminatorOpLowering<acc::LoopEndOp>>(&getContext());

  auto m = getModule();
  m.walk([&](acc::ParallelOp parallelOp) {
    auto numGangs = parallelOp.getNumGangs();
    auto numWorkers = parallelOp.getNumWorkers();
    parallelOp.walk([&](acc::LoopOp loopOp) {

      for (auto &op : loopOp.getBody().getOperations()) {
        if (auto forOp = dyn_cast<loop::ForOp>(&op)) {
          
          OpBuilder builder(parallelOp.getOperation()->getRegion(0));
          SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;
          auto constOp1 = builder.create<ConstantOp>(parallelOp.getLoc(), 
              builder.getIntegerAttr(builder.getIndexType(), numGangs));
          numWorkGroupsVal.push_back(constOp1);
          
          auto constOp2 = builder.create<ConstantOp>(parallelOp.getLoc(), 
              builder.getIntegerAttr(builder.getIndexType(), numWorkers));
          workGroupSizeVal.push_back(constOp2);
          
          if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
                                            workGroupSizeVal))) {
            loopOp.emitError(
              "Unable to map loop to accelerator.");
            signalPassFailure();  
          }
          extractOperationsOutsideOfConstruct(loopOp);
          loopOp.erase();
          
          break;
        } else {
          loopOp.emitError(
              "First operation in acc.loop region must be a loop.");
          signalPassFailure();
        }
      }
    }); // Walk over LoopOp within ParallelOp
    extractOperationsOutsideOfConstruct(parallelOp);
    parallelOp.erase();
  }); // Walk over ParallelOp within the module

  if (failed(applyPartialConversion(m, target, patterns)))
    signalPassFailure();
}

static PassRegistration<OpenACCToGPULoweringPass>
    pass("convert-openacc-to-gpu", "Convert OpenACC Ops to GPU dialect");
