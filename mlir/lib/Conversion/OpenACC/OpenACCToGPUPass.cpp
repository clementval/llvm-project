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
static void extractOperationsBeforeRegion(OpTy baseOp) {
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


static LogicalResult mapParallerLoopToGangWorker(acc::LoopOp accLoopOp, 
    ArrayRef<Value> numGangs, ArrayRef<Value> numWorkers) {
  for (auto &op : accLoopOp.getBody().getOperations()) {
    if (auto forOp = dyn_cast<loop::ForOp>(&op)) {
      mapLoopToProcessorIds(forOp, numGangs, numWorkers);
      extractOperationsBeforeRegion(accLoopOp);
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
  // auto barrierOp = builder.create<gpu::BarrierOp>(launchOp.getLoc());
  auto returnOp = builder.create<gpu::ReturnOp>(launchOp.getLoc());

  // extractOperationsOutsideOfRegion(op, barrierOp);
  extractOperationsOutsideOfRegion(parallelOp, returnOp);

  // Replace values that are used within the region of the launchOp but are
  // defined outside. They all are replaced with kernel arguments.
  for (auto pair :
       llvm::zip_first(valuesToForward, launchOp.getKernelArguments())) {
    Value from = std::get<0>(pair);
    Value to = std::get<1>(pair);
    replaceAllUsesInRegionWith(from, to, launchOp.body());
  }

  SmallVector<Value, 3> workgroupID = {launchOp.getBlockIds().z, launchOp.getBlockIds().y, launchOp.getBlockIds().x};
  SmallVector<Value, 3> numWorkGroups = {launchOp.getGridSize().z, launchOp.getGridSize().y, launchOp.getGridSize().x};
  
  // Adapt acc loop in the parallel region
  launchOp.walk([&](acc::LoopOp loopOp) {
    mapParallerLoopToGangWorker(loopOp, workgroupID[0], numWorkGroups[0]);
  });
  
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
