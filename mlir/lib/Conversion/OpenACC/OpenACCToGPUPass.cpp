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
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
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
  for (Operation &op : baseOp.getOperation()->getRegion(0).getBlocks()
                           .front().getOperations()) {
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

/**
 * 
 */
// static FuncOp outlineParallelRegion(acc::ParallelOp parallelOp) {
//   Location loc = parallelOp.getLoc();

//   OpBuilder builder(parallelOp.getContext());

//   std::string parallelRegionKernelName =
//       Twine(parallelOp.getParentOfType<FuncOp>().getName(), "_kernel").str();
//   // FunctionType type =
//   //     FunctionType::get(kernelOperandTypes, {}, parallelOp.getContext());

//   // auto outlinedFunc = builder.create<gpu::GPUFuncOp>(loc, kernelFuncName, type);
//   //   outlinedFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
//   //                      builder.getUnitAttr());

//   // outlinedFunc.body().takeBody(parallelOp.body());
  
  
//   FuncOp outlinedFunc = FuncOp::create(loc, parallelRegionKernelName, 
//       builder.getFunctionType(llvm::None, llvm::None));

//   outlinedFunc.getBody().takeBody(parallelOp.getOperation()->getRegion(0));

//   //outlinedFunc.walk([&](acc::LoopOp loopOp) {});

//   extractOperationsOutsideOfConstruct(parallelOp);

//   // Add a terminator at then end of the new func
//   builder.setInsertionPointToEnd(&outlinedFunc.getBody().back());
//   builder.create<mlir::ReturnOp>(loc);

//   parallelOp.erase();

//   return outlinedFunc;
// }

static void
replaceAllUsesExcept(Value orig, Value replacement,
                     const SmallPtrSetImpl<Operation *> &exceptions) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (exceptions.count(use.getOwner()) == 0)
      use.set(replacement);
  }
}

// Collapse `n` perfectly nested loops starting at `rootForOp` 
static loop::ForOp& collapseNestedLoops(loop::ForOp rootForOp, unsigned n) {

  // Looks for perfectly nested loops
  SmallVector<loop::ForOp, 4> loops;
  getPerfectlyNestedLoops(loops, rootForOp);
  if(loops.size() < n) {
    rootForOp.emitError("Not enough nested loops to collapse");
    return loops[0];
  }

  // Collapse the n first loops of the nest
  auto nest = llvm::makeMutableArrayRef(loops.data(), n);
  coalesceLoops(nest);
  return loops.front();

  // OpBuilder builder(rootForOp);
  // Location loc(rootForOp.getLoc());
  // builder.setInsertionPointToStart(rootForOp.getBody());

  // loop::ForOp outermost = loops.front();
  // loop::ForOp innermost = loops.back();

  // // for(auto loop : loops)
  // //   normalizeLoops(loops, outermost, innermost);

  // // Define new loop bounds
  // Value upperBound1 = outermost.upperBound();
  // Value upperBound2 = innermost.upperBound();

  // Value upperBound = outermost.upperBound();
  // for(unsigned i = 1; i < loops.size(); ++i) {
  //   auto mul = builder.create<MulIOp>(loc, upperBound, loops[i].upperBound());
  //   mul.getOperation()->moveBefore(rootForOp);
  //   upperBound = mul;
  // }

  // builder.setInsertionPointToStart(loops[1].getBody());
  // Value innerLoopInductionVar = loops[1].getInductionVar();



  // Value index1 = builder.create<UnsignedDivIOp>(loc, innerLoopInductionVar, upperBound1);
  // Value index2 = builder.create<UnsignedRemIOp>(loc, innerLoopInductionVar, upperBound2);
  

  // SmallPtrSet<Operation *, 2> preserve{index1.getDefiningOp(),
  //                                      index2.getDefiningOp()};


  // for (auto pair :
  //      llvm::zip_first(valuesToForward, launchOp.getKernelArguments())) {
  //   Value from = std::get<0>(pair);
  //   Value to = std::get<1>(pair);
  //   replaceAllUsesInRegionWith(from, to, launchOp.body());
  // }


  // for(auto loop : loops) {
  //   replaceAllUsesExcept(loop.getInductionVar(), index1, preserve);
  // }

  // innermost.setUpperBound(upperBound);

  // // replaceAllUsesExcept(loops[0].getInductionVar(), index1, preserve);
  // // replaceAllUsesExcept(loops[1].getInductionVar(), index2, preserve);

  // innermost.getOperation()->moveBefore(rootForOp);
  // outermost.erase();

  // return loops[1];
}

static LogicalResult mapParallerLoopToGangWorker(acc::LoopOp accLoopOp, 
    gpu::LaunchOp launchOp) {
  for (auto &op : accLoopOp.getBody().getOperations()) {
    if (auto forOp = dyn_cast<loop::ForOp>(&op)) {
      

      if(accLoopOp.hasCollapseAttr())
        forOp = collapseNestedLoops(forOp, accLoopOp.getCollapse());
    
      OpBuilder builder(forOp);
      Location loc(forOp.getLoc());
      if(accLoopOp.isGang()) { // Map to gang only
        forOp.setLowerBound(launchOp.getBlockIds().x);
        forOp.setStep(launchOp.getGridSize().x);
      } else if (accLoopOp.isVector()) {
        forOp.setLowerBound(launchOp.getThreadIds().x);
        forOp.setStep(launchOp.getBlockSize().x);
      } else {
        // lb = blockIdx.x * blockDim.x + threadIdx.x
        Value tmp = builder.create<MulIOp>(loc, launchOp.getBlockIds().x, 
            launchOp.getBlockSize().x);
        Value lb = builder.create<AddIOp>(loc, tmp, launchOp.getThreadIds().x);
        forOp.setLowerBound(lb);

        // step = gridDim.x * blockDim.x
        Value step = builder.create<MulIOp>(loc, launchOp.getGridSize().x, 
            launchOp.getBlockSize().x);
        forOp.setStep(step);
      }

      extractRegionBeforeItself(accLoopOp);
      accLoopOp.erase();
      break;
    } else if (dyn_cast<AffineForOp>(&op)) {
      accLoopOp.emitError("affine.for operation in acc.loop not supported yet.");
      return failure();
    } else {
      accLoopOp.emitError("First operation in acc.loop region must be a loop.");
      return failure();
    }
  }
  return success();
}

static LogicalResult createGPULaunchForParallelRegion(acc::ParallelOp 
    parallelOp) {
  OpBuilder builder(parallelOp.getOperation());
  auto loc = parallelOp.getLoc();

  Value one = builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), 1));

  // Create gangs constant if different than 1
  Value numGangs = (parallelOp.getNumGangs() != 1) ? 
    builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), 
      parallelOp.getNumGangs())) : one;

  // Create workers constant if different than 1
  Value numWorkers = (parallelOp.getNumWorkers() != 1) ? 
    builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), 
      parallelOp.getNumWorkers())) : one;

  llvm::SetVector<Value> valuesToForwardSet;
  getUsedValuesDefinedAbove(parallelOp.region(), parallelOp.region(), 
                            valuesToForwardSet);
  auto valuesToForward = valuesToForwardSet.takeVector();

  auto launchOp = builder.create<gpu::LaunchOp>(loc, 
      numGangs, one, one, numWorkers, one, one, valuesToForward);

  builder.setInsertionPointToEnd(&launchOp.body().front());
  auto gpuTerminatorOp = builder.create<gpu::TerminatorOp>(launchOp.getLoc());


  // Move parallel body into launchOp
  parallelOp.getOperation()->moveBefore(gpuTerminatorOp);
  
  // Replace values that are used within the region of the launchOp but are
  // defined outside. They all are replaced with kernel arguments.
  for (auto pair :
       llvm::zip_first(valuesToForward, launchOp.getKernelArguments())) {
    Value from = std::get<0>(pair);
    Value to = std::get<1>(pair);
    replaceAllUsesInRegionWith(from, to, launchOp.body());
  }

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
    if(failed(createGPULaunchForParallelRegion(parallelOp)))
      signalPassFailure();

    // parallelOp.walk([&](acc::LoopOp loopOp) {

      

      // for (auto &op : loopOp.getBody().getOperations()) {
      //   if (auto forOp = dyn_cast<loop::ForOp>(&op)) {

      //     OpBuilder builder(parallelOp.getOperation()->getRegion(0));
      //     SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;
      //     auto constOp1 = builder.create<ConstantOp>(parallelOp.getLoc(),
      //         builder.getIntegerAttr(builder.getIndexType(), numGangs));
      //     numWorkGroupsVal.push_back(constOp1);

      //     auto constOp2 = builder.create<ConstantOp>(parallelOp.getLoc(),
      //         builder.getIntegerAttr(builder.getIndexType(), numWorkers));
      //     workGroupSizeVal.push_back(constOp2);

      //     if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
      //                                       workGroupSizeVal))) {
      //       loopOp.emitError(
      //         "Unable to map loop to accelerator.");
      //       signalPassFailure();
      //     }
      //     extractOperationsOutsideOfConstruct(loopOp);
      //     loopOp.erase();

      //     break;
      //   } else if(auto affineForOp = dyn_cast<AffineForOp>(&op)) {
      //     loopOp.emitError("affine.for operation in acc.loop not supported yet.");
      //     signalPassFailure();
      //     break;
      //   } else {
      //     loopOp.emitError(
      //         "First operation in acc.loop region must be a loop.");
      //     signalPassFailure();
      //   }
      // }
    // }); // Walk over LoopOp within ParallelOp
  }); // Walk over ParallelOp within the module

  if (failed(applyPartialConversion(m, target, patterns)))
    signalPassFailure();
}

static PassRegistration<OpenACCToGPULoweringPass>
    pass("convert-openacc-to-gpu", "Convert OpenACC Ops to GPU dialect");
