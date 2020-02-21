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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

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

template <typename OpTy>
static void extractRegionBeforeItself(OpTy baseOp) {
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

template <typename StructureOp, typename OpTy>
static void extractOperationsOutsideOfRegion(StructureOp baseOp,
                                             OpTy insPosition) {
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
    op->moveBefore(insPosition);
  }
}

// Collapse `n` perfectly nested loops starting at `rootForOp`
// New opeartion may be placed between in the accLoopOp region before
// the forOp.
static LogicalResult collapseNestedLoops(loop::ForOp rootForOp, 
                                         acc::LoopOp accLoopOp) {
  unsigned n = accLoopOp.getCollapse();
  // Looks for perfectly nested loops
  SmallVector<loop::ForOp, 4> loops;
  getPerfectlyNestedLoops(loops, rootForOp);
  if (loops.size() < n) {
    rootForOp.emitError("Not enough nested loops to collapse");
    return failure();
  }

  // Collapse the n first loops of the nest
  auto nest = llvm::makeMutableArrayRef(loops.data(), n);
  coalesceLoops(nest);
  return success();
}

static LogicalResult applyCollapseClause(acc::LoopOp accLoopOp) {
  if (!accLoopOp.hasCollapseAttr())
    return success();
  if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
    if(failed(collapseNestedLoops(forOp, accLoopOp)))
      return failure();
    accLoopOp.removeAttr(acc::LoopOp::getCollapseAttrName());
  }
  return success();
}

static LogicalResult mapParallerLoopToGangWorker(acc::LoopOp accLoopOp,
                                                 gpu::LaunchOp launchOp) {
  if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
    OpBuilder builder(forOp);
    Location loc(forOp.getLoc());

    if (accLoopOp.hasSeqAttr()) { // Loop has to be exectued sequentially
      // Create if (blockId.x == 0 && threadId.x == 0) then { do work }
      Value const0 = builder.create<ConstantOp>(
          loc, builder.getIntegerAttr(builder.getIndexType(), 0));
      Value isBlock0 = builder.create<CmpIOp>(
          loc, CmpIPredicate::eq, launchOp.getBlockIds().x, const0);
      Value isThread0 = builder.create<CmpIOp>(
          loc, CmpIPredicate::eq, launchOp.getThreadIds().x, const0);
      Value isSeqId = builder.create<AndOp>(loc, isBlock0, isThread0);
      auto ifOp = builder.create<loop::IfOp>(loc, isSeqId, false);

      // Add a barrier to sync all worker after the seq loop
      builder.create<gpu::BarrierOp>(loc);

      // Move the seq loop in the if-then region
      auto &thenRegion = ifOp.thenRegion();
      forOp.getOperation()->moveBefore(thenRegion.back().getTerminator());
    } else {
      if (accLoopOp.isGangVector() ||
          accLoopOp.getExecutionMappingAttr() ==
              acc::OpenACCExecMapping::NONE) { // Map to gang vector
        // lb = blockIdx.x * blockDim.x + threadIdx.x
        Value tmp = builder.create<MulIOp>(loc, launchOp.getBlockIds().x,
                                           launchOp.getBlockSize().x);
        Value lb =
            builder.create<AddIOp>(loc, tmp, launchOp.getThreadIds().x);
        forOp.setLowerBound(lb);

        // step = gridDim.x * blockDim.x
        Value step = builder.create<MulIOp>(loc, launchOp.getGridSize().x,
                                            launchOp.getBlockSize().x);
        forOp.setStep(step);
      } else if (accLoopOp.isGang()) { // Map to gang only
        forOp.setLowerBound(launchOp.getBlockIds().x);
        forOp.setStep(launchOp.getGridSize().x);
      } else if (accLoopOp.isVector()) { // Map to vector only
        forOp.setLowerBound(launchOp.getThreadIds().x);
        forOp.setStep(launchOp.getBlockSize().x);
      }
    }

    extractRegionBeforeItself(accLoopOp);
    accLoopOp.erase();
  } else if (dyn_cast<AffineForOp>(accLoopOp.getBody().front().front())) {
    accLoopOp.emitError(
        "affine.for operation in acc.loop not supported yet.");
    return failure();
  } else {
    accLoopOp.emitError("First operation in acc.loop region must be a loop.");
    return failure();
  }
  return success();
}

static void gatherForOp(acc::LoopOp accLoopOp, SmallVector<loop::ForOp, 2> &forOps) {
  if(auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
    forOps.push_back(forOp);
  }
}

static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             int64_t divisor) {
  assert(divisor > 0 && "expected positive divisor");
  assert(dividend.getType().isIndex() && "expected index-typed value");
  Value divisorMinusOneCst = builder.create<ConstantIndexOp>(loc, divisor - 1);
  Value divisorCst = builder.create<ConstantIndexOp>(loc, divisor);
  Value sum = builder.create<AddIOp>(loc, dividend, divisorMinusOneCst);
  return builder.create<SignedDivIOp>(loc, sum, divisorCst);
}


static Value generateGangValue(OpBuilder &builder, Location loc, 
                               loop::ForOp forOp) {
  Value diffUpLb = builder.create<SubIOp>(loc, forOp.upperBound(), 
      forOp.lowerBound());
  Value numberOfIterations = ceilDivPositive(builder, loc, diffUpLb, 
      forOp.step());
  Value nbOfGangs = ceilDivPositive(builder, loc, forOp.upperBound(), 128);
  return nbOfGangs;
}

template <typename StructureOp>
static void hoistOpBeforeOperation(StructureOp &parentOp, 
                                   acc::LoopOp &accLoopOp) {
  // Hoist operation created by coalesceLoops out of acc.loop
  SmallVector<Operation *, 5> toHoist;
  for (auto &op : accLoopOp.getBody().front().getOperations()) {
    if (isa<loop::ForOp>(op))
      break;
    toHoist.push_back(&op);
  }

  for (auto *op : toHoist) {
    op->moveBefore(parentOp);
  }
}
 
static LogicalResult
createGPULaunchForParallelRegion(acc::ParallelOp parallelOp) {
  OpBuilder builder(parallelOp.getOperation());
  auto loc = parallelOp.getLoc();

  bool dynamicNumGangs = parallelOp.getNumGangs() == 1;

  Value one = builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), 1));
  Value numGangs = one;
  Value numWorkers = one;

  // Count number of parallel loops to deduce gang/worker count
  SmallVector<loop::ForOp, 2> forOps;
  parallelOp.walk([&](acc::LoopOp accLoopOp) {
    applyCollapseClause(accLoopOp);
    if(dynamicNumGangs)
      hoistOpBeforeOperation(parallelOp, accLoopOp);
    else
      hoistOpBeforeOperation(accLoopOp, accLoopOp);
    if(!accLoopOp.hasSeqAttr()) {
      gatherForOp(accLoopOp, forOps);
    }
  });

  if(dynamicNumGangs && forOps.size() != 0) {
    assert(forOps.size() > 0 && "At least one forOp needed to compute the gang size");
    numGangs = generateGangValue(builder, loc, forOps.front()); // TODO: when there is more than one loop in the parallel region
  } else if(!dynamicNumGangs) {
    numGangs = builder.create<ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getIndexType(),
        parallelOp.getNumGangs()));
  }

  // Create workers constant if different than 1
  if(dynamicNumGangs && forOps.size() != 0) {
    numWorkers = builder.create<ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getIndexType(), 128));
  } else if(parallelOp.getNumWorkers() != 1) {
    numWorkers = builder.create<ConstantOp>(loc, 
        builder.getIntegerAttr(builder.getIndexType(),
        parallelOp.getNumWorkers()));
  }

  auto launchOp = builder.create<gpu::LaunchOp>(loc, numGangs, one, one,
                                                numWorkers, one, one);

  builder.setInsertionPointToEnd(&launchOp.body().front());
  auto gpuTerminatorOp = builder.create<gpu::TerminatorOp>(launchOp.getLoc());

  // Move parallel operation into launchOp body
  parallelOp.getOperation()->moveBefore(gpuTerminatorOp);

  // Adapt acc loop in the parallel region
  parallelOp.walk([&](acc::LoopOp accLoopOp) {
    mapParallerLoopToGangWorker(accLoopOp, launchOp);
  });

  extractRegionBeforeItself(parallelOp);
  parallelOp.erase();

  return success();
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
    // Convert the parallel region into a GPU kernel
    if (failed(createGPULaunchForParallelRegion(parallelOp)))
      signalPassFailure();
  }); // Walk over ParallelOp within the module

  if (failed(applyPartialConversion(m, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertOpenACCToGPUPass() {
  return std::make_unique<OpenACCToGPULoweringPass>();
}

static PassRegistration<OpenACCToGPULoweringPass>
    pass("convert-openacc-to-gpu", "Convert OpenACC Ops to GPU dialect");
