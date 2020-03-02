//===- OpenACCToTargetPass.cpp - Convert OpenACC to Target runtime calls --===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// This file implements a pass to convert MLIR OpenACC ops into the target
// runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/Conversion/OpenACCToGPU/ConvertOpenACCToTarget.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

struct OpenACCToTargetLoweringPass
    : public ModulePass<OpenACCToTargetLoweringPass> {
  void runOnModule() override;

private:
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable);
};

struct ParallelOpOutling final : public OpRewritePattern<acc::ParallelOp> {
  using OpRewritePattern<acc::ParallelOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(acc::ParallelOp parallelOp,
                                     PatternRewriter &rewriter) const override;
};

// struct LoopOpLowering final : public OpRewritePattern<acc::LoopOp> {
//   using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

//   PatternMatchResult matchAndRewrite(acc::LoopOp loopOp,
//                                      PatternRewriter &rewriter) const
//                                      override;
// };

template <typename TerminatorOp>
struct TerminatorOpLowering final : public OpRewritePattern<TerminatorOp> {
  using OpRewritePattern<TerminatorOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TerminatorOp terminatorOp,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(terminatorOp);
    return Pattern::matchSuccess();
  }
};

static FuncOp outlineParallelRegion(acc::ParallelOp parallelOp,
                                    llvm::SetVector<Value> &operands, 
                                    llvm::SetVector<Value> &privates) {
  Location loc = parallelOp.getLoc();
  OpBuilder builder(parallelOp.getContext());

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  llvm::SetVector<Value> allOperands;
  getUsedValuesDefinedAbove(parallelOp.getOperation()->getRegion(0), 
      allOperands);

  for(Value v : allOperands) {
    bool isInPrivate = false;
    for(Value p : parallelOp.getGangPrivates()) {
      if(v == p) {
        isInPrivate = true;
        privates.insert(v);
        break;
      }
    }
    if(!isInPrivate) {
      operands.insert(v);
    }
  }

  SmallVector<Type, 4> regionOperandTypes;
  regionOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    regionOperandTypes.push_back(operand.getType());
  }
  std::string parallelRegionFuncName =
      Twine(parallelOp.getParentOfType<FuncOp>().getName(), "_acc_parallel")
          .str();
  auto funcType = builder.getFunctionType(regionOperandTypes, llvm::None);
  auto outlinedRegion = FuncOp::create(loc, parallelRegionFuncName, funcType);
  outlinedRegion.getBody().takeBody(parallelOp.getOperation()->getRegion(0));

  Block &entryBlock = outlinedRegion.getBody().front();
  for (Value operand : operands) {
    BlockArgument newArg = entryBlock.addArgument(operand.getType());
    replaceAllUsesInRegionWith(operand, newArg, outlinedRegion.getBody());
  }

  outlinedRegion.walk([](acc::ParallelEndOp op) {
    OpBuilder replacer(op);
    replacer.create<ReturnOp>(op.getLoc());
    op.erase();
  });

  return outlinedRegion;
}

PatternMatchResult
ParallelOpOutling::matchAndRewrite(acc::ParallelOp parallelOp,
                                   PatternRewriter &rewriter) const {
  auto module = parallelOp.getParentOfType<ModuleOp>();
  SymbolTable symbolTable(module);
  llvm::SetVector<Value> operands;
  llvm::SetVector<Value> privates;
  auto outlinedParallelRegion = outlineParallelRegion(parallelOp, operands, 
      privates);
  symbolTable.insert(outlinedParallelRegion);

  // replace region with newly outlined function call
  // rewriter.setInsertionPoint(parallelOp);
  OpBuilder builder(parallelOp.getContext());
  builder.create<CallOp>(parallelOp.getLoc(), outlinedParallelRegion,
                         operands.getArrayRef());
  // inlineBeneficiaryOps(kernelFunc, launchFuncOp);

  rewriter.eraseOp(parallelOp);
  return matchSuccess();
}

// PatternMatchResult
// LoopOpLowering::matchAndRewrite(acc::LoopOp loopOp,
//                                 PatternRewriter &rewriter) const {
//   // Not used now
//   return matchSuccess();
// }

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

// template <typename StructureOp, typename OpTy>
// static void extractOperationsOutsideOfRegion(StructureOp baseOp,
//                                              OpTy insPosition) {
//   SmallVector<Operation *, 8> toHoist;
//   for (Operation &op : baseOp.getOperation()
//                            ->getRegion(0)
//                            .getBlocks()
//                            .front()
//                            .getOperations()) {
//     if (&op == baseOp.getOperation()) {
//       continue;
//     } else {
//       toHoist.push_back(&op);
//     }
//   }

//   for (auto *op : toHoist) {
//     op->moveBefore(insPosition);
//   }
// }

// // Collapse `n` perfectly nested loops starting at `rootForOp`
// // New opeartion may be placed between in the accLoopOp region before
// // the forOp.
// static LogicalResult collapseNestedLoops(loop::ForOp rootForOp,
//                                          acc::LoopOp accLoopOp) {
//   unsigned n = accLoopOp.getCollapse();
//   // Looks for perfectly nested loops
//   SmallVector<loop::ForOp, 4> loops;
//   getPerfectlyNestedLoops(loops, rootForOp);
//   if (loops.size() < n) {
//     rootForOp.emitError("Not enough nested loops to collapse");
//     return failure();
//   }

//   // Collapse the n first loops of the nest
//   auto nest = llvm::makeMutableArrayRef(loops.data(), n);
//   coalesceLoops(nest);
//   return success();
// }

// static LogicalResult applyCollapseClause(acc::LoopOp accLoopOp) {
//   if (!accLoopOp.hasCollapseAttr())
//     return success();
//   if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front())) {
//     if(failed(collapseNestedLoops(forOp, accLoopOp)))
//       return failure();
//     accLoopOp.removeAttr(acc::LoopOp::getCollapseAttrName());
//   }
//   return success();
// }

// static LogicalResult mapParallerLoopToGangWorker(acc::LoopOp accLoopOp,
//                                                  gpu::LaunchOp launchOp) {
//   if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front())) {
//     OpBuilder builder(forOp);
//     Location loc(forOp.getLoc());

//     if (accLoopOp.hasSeqAttr()) { // Loop has to be exectued sequentially
//       // Create if (blockId.x == 0 && threadId.x == 0) then { do work }
//       Value const0 = builder.create<ConstantOp>(
//           loc, builder.getIntegerAttr(builder.getIndexType(), 0));
//       Value isBlock0 = builder.create<CmpIOp>(
//           loc, CmpIPredicate::eq, launchOp.getBlockIds().x, const0);
//       Value isThread0 = builder.create<CmpIOp>(
//           loc, CmpIPredicate::eq, launchOp.getThreadIds().x, const0);
//       Value isSeqId = builder.create<AndOp>(loc, isBlock0, isThread0);
//       auto ifOp = builder.create<loop::IfOp>(loc, isSeqId, false);

//       // Add a barrier to sync all worker after the seq loop
//       builder.create<gpu::BarrierOp>(loc);

//       // Move the seq loop in the if-then region
//       auto &thenRegion = ifOp.thenRegion();
//       forOp.getOperation()->moveBefore(thenRegion.back().getTerminator());
//     } else {
//       if (accLoopOp.isGangVector() ||
//           accLoopOp.getExecutionMappingAttr() ==
//               acc::OpenACCExecMapping::NONE) { // Map to gang vector
//         // lb = blockIdx.x * blockDim.x + threadIdx.x
//         Value tmp = builder.create<MulIOp>(loc, launchOp.getBlockIds().x,
//                                            launchOp.getBlockSize().x);
//         Value lb =
//             builder.create<AddIOp>(loc, tmp, launchOp.getThreadIds().x);
//         forOp.setLowerBound(lb);

//         // step = gridDim.x * blockDim.x
//         Value step = builder.create<MulIOp>(loc, launchOp.getGridSize().x,
//                                             launchOp.getBlockSize().x);
//         forOp.setStep(step);
//       } else if (accLoopOp.isGang()) { // Map to gang only
//         forOp.setLowerBound(launchOp.getBlockIds().x);
//         forOp.setStep(launchOp.getGridSize().x);
//       } else if (accLoopOp.isVector()) { // Map to vector only
//         forOp.setLowerBound(launchOp.getThreadIds().x);
//         forOp.setStep(launchOp.getBlockSize().x);
//       }
//     }

//     extractRegionBeforeItself(accLoopOp);
//     accLoopOp.erase();
//   } else if (dyn_cast<AffineForOp>(accLoopOp.getBody().front())) {
//     accLoopOp.emitError(
//         "affine.for operation in acc.loop not supported yet.");
//     return failure();
//   } else {
//     accLoopOp.emitError("First operation in acc.loop region must be a
//     loop."); return failure();
//   }
//   return success();
// }

// static void gatherForOp(acc::LoopOp accLoopOp, SmallVector<loop::ForOp, 2>
// &forOps) {
//   if(auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front())) {
//     forOps.push_back(forOp);
//   }
// }

// static Value ceilDivPositive(OpBuilder &builder, Location loc, Value
// dividend,
//                              int64_t divisor) {
//   assert(divisor > 0 && "expected positive divisor");
//   assert(dividend.getType().isIndex() && "expected index-typed value");
//   Value divisorMinusOneCst = builder.create<ConstantIndexOp>(loc, divisor -
//   1); Value divisorCst = builder.create<ConstantIndexOp>(loc, divisor); Value
//   sum = builder.create<AddIOp>(loc, dividend, divisorMinusOneCst); return
//   builder.create<SignedDivIOp>(loc, sum, divisorCst);
// }

// static Value generateGangValue(OpBuilder &builder, Location loc,
//                                loop::ForOp forOp) {
//   Value diffUpLb = builder.create<SubIOp>(loc, forOp.upperBound(),
//       forOp.lowerBound());
//   Value numberOfIterations = ceilDivPositive(builder, loc, diffUpLb,
//       forOp.step());
//   Value nbOfGangs = ceilDivPositive(builder, loc, forOp.upperBound(), 128);
//   return nbOfGangs;
// }

// template <typename StructureOp>
// static void hoistOpBeforeOperation(StructureOp &parentOp,
//                                    acc::LoopOp &accLoopOp) {
//   // Hoist operation created by coalesceLoops out of acc.loop
//   SmallVector<Operation *, 5> toHoist;
//   for (auto &op : accLoopOp.getBody().getOperations()) {
//     if (dyn_cast<loop::ForOp>(&op))
//       break;
//     toHoist.push_back(&op);
//   }

//   for (auto *op : toHoist) {
//     op->moveBefore(parentOp);
//   }
// }

// static LogicalResult
// createGPULaunchForParallelRegion(acc::ParallelOp parallelOp) {
//   OpBuilder builder(parallelOp.getOperation());
//   auto loc = parallelOp.getLoc();

//   bool dynamicNumGangs = parallelOp.getNumGangs() == 1;

//   Value one = builder.create<ConstantOp>(
//       loc, builder.getIntegerAttr(builder.getIndexType(), 1));
//   Value numGangs = one;
//   Value numWorkers = one;

//   // Count number of parallel loops to deduce gang/worker count
//   SmallVector<loop::ForOp, 2> forOps;
//   parallelOp.walk([&](acc::LoopOp accLoopOp) {
//     applyCollapseClause(accLoopOp);
//     if(dynamicNumGangs)
//       hoistOpBeforeOperation(parallelOp, accLoopOp);
//     else
//       hoistOpBeforeOperation(accLoopOp, accLoopOp);
//     if(!accLoopOp.hasSeqAttr()) {
//       gatherForOp(accLoopOp, forOps);
//     }
//   });

//   if(dynamicNumGangs && forOps.size() != 0) {
//     assert(forOps.size() > 0 && "At least one forOp needed to compute the
//     gang size"); numGangs = generateGangValue(builder, loc, forOps.front());
//     // TODO: when there is more than one loop in the parallel region
//   } else if(!dynamicNumGangs) {
//     numGangs = builder.create<ConstantOp>(loc,
//         builder.getIntegerAttr(builder.getIndexType(),
//         parallelOp.getNumGangs()));
//   }

//   // Create workers constant if different than 1
//   if(dynamicNumGangs && forOps.size() != 0) {
//     numWorkers = builder.create<ConstantOp>(loc,
//         builder.getIntegerAttr(builder.getIndexType(), 128));
//   } else if(parallelOp.getNumWorkers() != 1) {
//     numWorkers = builder.create<ConstantOp>(loc,
//         builder.getIntegerAttr(builder.getIndexType(),
//         parallelOp.getNumWorkers()));
//   }

//   auto launchOp = builder.create<gpu::LaunchOp>(loc, numGangs, one, one,
//                                                 numWorkers, one, one);

//   builder.setInsertionPointToEnd(&launchOp.body().front());
//   auto gpuTerminatorOp =
//   builder.create<gpu::TerminatorOp>(launchOp.getLoc());

//   // Move parallel operation into launchOp body
//   parallelOp.getOperation()->moveBefore(gpuTerminatorOp);

//   // Adapt acc loop in the parallel region
//   parallelOp.walk([&](acc::LoopOp accLoopOp) {
//     mapParallerLoopToGangWorker(accLoopOp, launchOp);
//   });

//   extractRegionBeforeItself(parallelOp);
//   parallelOp.erase();

//   return success();
// }

// TODO remove after inlineBeneficiaryOps is called in KernelOutlining
static bool isInliningBeneficiary(Operation *op) {
  return isa<ConstantOp>(op) || isa<DimOp>(op);
}

// TODO make the one in GPU - KernelOutling accessible so we can call it
// Move arguments of the given kernel function into the function if this reduces
// the number of kernel arguments.
static gpu::LaunchFuncOp inlineBeneficiaryOps(gpu::GPUFuncOp kernelFunc,
                                              gpu::LaunchFuncOp launch) {
  OpBuilder kernelBuilder(kernelFunc.getBody());
  auto &firstBlock = kernelFunc.getBody().front();
  SmallVector<Value, 8> newLaunchArgs;
  BlockAndValueMapping map;
  for (int i = 0, e = launch.getNumKernelOperands(); i < e; ++i) {
    map.map(launch.getKernelOperand(i), kernelFunc.getArgument(i));
  }
  for (int i = launch.getNumKernelOperands() - 1; i >= 0; --i) {
    auto operandOp = launch.getKernelOperand(i).getDefiningOp();
    if (!operandOp || !isInliningBeneficiary(operandOp)) {
      newLaunchArgs.push_back(launch.getKernelOperand(i));
      continue;
    }
    // Only inline operations that do not create new arguments.
    if (!llvm::all_of(operandOp->getOperands(),
                      [map](Value value) { return map.contains(value); })) {
      continue;
    }
    auto clone = kernelBuilder.clone(*operandOp, map);
    firstBlock.getArgument(i).replaceAllUsesWith(clone->getResult(0));
    firstBlock.eraseArgument(i);
  }
  if (newLaunchArgs.size() == launch.getNumKernelOperands())
    return launch;

  std::reverse(newLaunchArgs.begin(), newLaunchArgs.end());
  OpBuilder LaunchBuilder(launch);
  SmallVector<Type, 8> newArgumentTypes;
  newArgumentTypes.reserve(firstBlock.getNumArguments());
  for (auto value : firstBlock.getArguments()) {
    newArgumentTypes.push_back(value.getType());
  }
  kernelFunc.setType(LaunchBuilder.getFunctionType(newArgumentTypes, {}));
  auto newLaunch = LaunchBuilder.create<gpu::LaunchFuncOp>(
      launch.getLoc(), kernelFunc, launch.getGridSizeOperandValues(),
      launch.getBlockSizeOperandValues(), newLaunchArgs);
  launch.erase();
  return newLaunch;
}


static gpu::GPUFuncOp
convertOutlinedParallelRegionToKernel(FuncOp outlinedParallelRegion) {
  OpBuilder builder(outlinedParallelRegion);
  auto loc = outlinedParallelRegion.getLoc();
  FunctionType type =
      FunctionType::get(outlinedParallelRegion.getType().getInputs(), {},
                        outlinedParallelRegion.getContext());
  auto kernelFunc = builder.create<gpu::GPUFuncOp>(
      loc, outlinedParallelRegion.getName(), type);
  kernelFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     builder.getUnitAttr());
  kernelFunc.body().takeBody(outlinedParallelRegion.getBody());
  // injectGpuIndexOperations(loc, kernelFunc.body());
  kernelFunc.walk([](ReturnOp op) {
    OpBuilder replacer(op);
    replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return kernelFunc;
}

///
///
static gpu::LaunchFuncOp
createLaunchParallelRegion(acc::ParallelOp parallelOp,
                           gpu::GPUFuncOp outlinedParallelRegionKernel,
                           ValueRange operands) {
  OpBuilder builder(parallelOp);
  Value constOne = builder.create<ConstantOp>(
      parallelOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(), 1));

  Value numGangs = constOne;
  if (parallelOp.getNumGangs() != 1)
    numGangs = builder.create<ConstantOp>(
        parallelOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(),
                                                    parallelOp.getNumGangs()));

  auto callOutlinedParallelRegionKernel = builder.create<gpu::LaunchFuncOp>(
      parallelOp.getLoc(), outlinedParallelRegionKernel, numGangs, constOne,
      constOne, constOne, constOne, constOne, operands);
  // TODO FRIDAY look here       
  // inlineBeneficiaryOps(outlinedParallelRegionKernel,
  //                      callOutlinedParallelRegionKernel);
  return callOutlinedParallelRegionKernel;
}

// TODO grab the one from KernelOutining if possible
gpu::GPUModuleOp OpenACCToTargetLoweringPass::createKernelModule(
    gpu::GPUFuncOp kernelFunc, const SymbolTable &parentSymbolTable) {
  // TODO: This code cannot use an OpBuilder because it must be inserted into
  // a SymbolTable by the caller. SymbolTable needs to be refactored to
  // prevent manual building of Ops with symbols in code using SymbolTables
  // and then this needs to use the OpBuilder.
  auto context = getModule().getContext();
  Builder builder(context);
  OperationState state(kernelFunc.getLoc(),
                       gpu::GPUModuleOp::getOperationName());
  gpu::GPUModuleOp::build(&builder, state, kernelFunc.getName());
  auto kernelModule = cast<gpu::GPUModuleOp>(Operation::create(state));
  SymbolTable symbolTable(kernelModule);
  symbolTable.insert(kernelFunc);

  SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
  while (!symbolDefWorklist.empty()) {
    if (Optional<SymbolTable::UseRange> symbolUses =
            SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        StringRef symbolName =
            symbolUse.getSymbolRef().cast<FlatSymbolRefAttr>().getValue();
        if (symbolTable.lookup(symbolName))
          continue;

        Operation *symbolDefClone =
            parentSymbolTable.lookup(symbolName)->clone();
        symbolDefWorklist.push_back(symbolDefClone);
        symbolTable.insert(symbolDefClone);
      }
    }
  }

  return kernelModule;
}

static constexpr unsigned BLOCK_ID_X = 0;
static constexpr unsigned THREAD_ID_X = 1;
static constexpr unsigned GRID_DIM_X = 2;
static constexpr unsigned BLOCK_DIM_X = 3;

static void createSeqBlock(OpBuilder &builder, Location loc, 
                           SmallVector<Value, 4> &indexOps, Operation* op) {
  assert(indexOps.size() == 4 && "Expecting 4 indexes");

  // if blockIdx.x == 0 && threadIdx.x == 0 then
  Value const0 = builder.create<ConstantOp>(
    loc, builder.getIntegerAttr(builder.getIndexType(), 0));
  Value isBlock0 = builder.create<CmpIOp>(
    loc, CmpIPredicate::eq, indexOps[BLOCK_ID_X], const0);
  Value isThread0 = builder.create<CmpIOp>(
   loc, CmpIPredicate::eq, indexOps[THREAD_ID_X], const0);
  Value isSeqId = builder.create<AndOp>(loc, isBlock0, isThread0);
  auto ifOp = builder.create<loop::IfOp>(loc, isSeqId, false);

  // Move operation inside the newly created if-then region
  auto &thenRegion = ifOp.thenRegion();
  op->moveBefore(thenRegion.back().getTerminator());

  // Add a barrier to sync all worker after the seq loop
  builder.setInsertionPointAfter(ifOp);
  builder.create<gpu::BarrierOp>(loc);
}

static void mapLoopToGrid(acc::LoopOp accLoopOp, 
                          SmallVector<Value, 4> &indexOps) {
  if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
    OpBuilder builder(forOp);
    Location loc(accLoopOp.getLoc());
    if(accLoopOp.isSeq()) {
      // Nothing done now. 
    } else if(accLoopOp.isGangVector() || accLoopOp.getExecutionMappingAttr() ==
        acc::OpenACCExecMapping::NONE) {

      // lb = blockIdx.x * blockDim.x + threadIdx.x          
      Value tmp = builder.create<MulIOp>(loc, indexOps[BLOCK_ID_X],
                                         indexOps[BLOCK_DIM_X]);
      Value lb = builder.create<AddIOp>(loc, tmp, indexOps[THREAD_ID_X]);
      forOp.setLowerBound(lb);
      
      // step = gridDim.x * blockDim.x
      Value step = builder.create<MulIOp>(loc, indexOps[GRID_DIM_X],
                                          indexOps[BLOCK_DIM_X]);
      forOp.setStep(step);
    } else if (accLoopOp.isGang()) {
      forOp.setLowerBound(indexOps[BLOCK_ID_X]);
      forOp.setStep(indexOps[GRID_DIM_X]);
    } else if (accLoopOp.isVector() || accLoopOp.isWorker()) {
      forOp.setLowerBound(indexOps[THREAD_ID_X]);
      forOp.setStep(indexOps[BLOCK_DIM_X]);
    } 
  } else {

  }

  accLoopOp.walk([](acc::LoopEndOp op) {
    op.erase();
  });
  extractRegionBeforeItself(accLoopOp);
  accLoopOp.erase();
}

static void transformGangRedundant(acc::GangRedundantOp accGangRedundantOp, 
                                   SmallVector<Value, 4> &indexOps) {

  OpBuilder builder(accGangRedundantOp);
  Location loc = accGangRedundantOp.getLoc();
  
  
  // if threadIdx.x == 0 then
  Value const0 = builder.create<ConstantOp>(
    loc, builder.getIntegerAttr(builder.getIndexType(), 0));
  Value isThread0 = builder.create<CmpIOp>(
   loc, CmpIPredicate::eq, indexOps[THREAD_ID_X], const0);
  
  auto ifOp = builder.create<loop::IfOp>(loc, isThread0, false);

  // Move operation inside the newly created if-then region
  auto &thenRegion = ifOp.thenRegion();
  
  // accGangRedundantOp.walk([](acc::GangRedundantEndOp op) {
  //   op.erase();
  // });

  thenRegion.takeBody(accGangRedundantOp.getBody());
  // Add a barrier to sync all worker after the seq loop
  builder.setInsertionPointAfter(ifOp);
  accGangRedundantOp.erase();
}


static void applyGangPrivateList(gpu::GPUFuncOp outlinedParallelRegion,
                                 acc::ParallelOp accParallelOp, 
                                 llvm::SetVector<Value> &privates) {
  if(accParallelOp.getNumGangPrivates() == 0)
    return;
  
  // assert(accParallelOp.getNumGangPrivates() == privates.size() 
  //     && "Number of private variable doesn't match");
  // OpBuilder builder(outlinedParallelRegion.getBody());
  // for(auto p : accParallelOp.getGangPrivates()) {
  //   auto type = p.getType().dyn_cast<MemRefType>();
  //   assert(type && type.hasStaticShape() && "can only privatize memrefs");
  //   auto newPrivate = builder.create<AllocOp>(accParallelOp.getLoc(),
  //       MemRefType::Builder(type).setMemorySpace(
  //         gpu::GPUDialect::getWorkgroupAddressSpace()));
  //   replaceAllUsesInRegionWith(p, newPrivate,
  //       outlinedParallelRegion.getBody());
  // }

  assert(accParallelOp.getNumGangPrivates() == privates.size() 
      && "Number of private variable doesn't match");

  for(auto p : accParallelOp.getGangPrivates()) {
    auto type = p.getType().dyn_cast<MemRefType>();
    assert(type && type.hasStaticShape() && "can only privatize memrefs");

    Value newPrivate = outlinedParallelRegion.addWorkgroupAttribution(
        type.getShape(), type.getElementType());
    
    replaceAllUsesInRegionWith(p, newPrivate, outlinedParallelRegion.getBody());
  } 
}

static void removeUslessArguments(gpu::GPUFuncOp outlinedParallelRegionKernel,
                                  llvm::SetVector<Value> &operands) {
  assert(outlinedParallelRegionKernel.getNumArguments() == operands.size() 
      && "operands size must be the same as number of arguments");                                   
  auto &firstBlock = outlinedParallelRegionKernel.getBody().front();
  for(int i = firstBlock.getNumArguments() - 1; i >= 0; --i) {
    Value arg = firstBlock.getArgument(i);
    if(arg.use_empty()) {
      firstBlock.eraseArgument(i);
      operands.remove(operands[i]);
    }
  }
}

template <typename OpTy>
static void createForX(OpBuilder &builder, Location loc,
                       SmallVectorImpl<Value> &values) {
  Value v = builder.create<OpTy>(loc, builder.getIndexType(),
                                 builder.getStringAttr("x"));
  values.push_back(v);
}


static void injectGpuIndexOperations(Location loc, Region &body,
                                     SmallVector<Value, 4> &indexOps) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = body.front();
  builder.setInsertionPointToStart(&firstBlock);
  createForX<gpu::BlockIdOp>(builder, loc, indexOps);
  createForX<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForX<gpu::GridDimOp>(builder, loc, indexOps);
  createForX<gpu::BlockDimOp>(builder, loc, indexOps);
}

void OpenACCToTargetLoweringPass::runOnModule() {

  ConversionTarget target(getContext());
  target.addIllegalDialect<acc::OpenACCDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  // target.addLegalOp<acc::ParallelOp>();
  // target.addLegalOp<acc::ParallelEndOp>();
  target.addLegalOp<acc::LoopOp>();
  target.addLegalOp<acc::LoopEndOp>();
  target.addLegalOp<acc::GangRedundantEndOp>();

  // If operation is considered legal the rewrite pattern in not called.
  OwningRewritePatternList patterns;
  // patterns.insert<ParallelOpOutling>(&getContext());
  // patterns.insert<TerminatorOpLowering<acc::ParallelEndOp>>(&getContext());
  // patterns.insert<TerminatorOpLowering<acc::LoopEndOp>>(&getContext());

  auto m = getModule();
  SymbolTable symbolTable(m);
  bool modified = false;

  for (auto func : getModule().getOps<FuncOp>()) {
    Block::iterator insertPt(func.getOperation()->getNextNode());

    // Walk over ParallelOp operation to outline the parallel region
    m.walk([&](acc::ParallelOp parallelOp) {
      OpBuilder builder(parallelOp);
      llvm::SetVector<Value> operands;
      llvm::SetVector<Value> gangPrivates;
      auto outlinedParallelRegion = outlineParallelRegion(parallelOp, operands, 
          gangPrivates);
      auto outlinedParallelRegionKernel =
          convertOutlinedParallelRegionToKernel(outlinedParallelRegion);

      SmallVector<Value, 4> indexOps;
      injectGpuIndexOperations(parallelOp.getLoc(),
          outlinedParallelRegionKernel.getBody(), indexOps);

      outlinedParallelRegionKernel.walk([&](acc::GangRedundantOp accGangRedundantOp) {
        transformGangRedundant(accGangRedundantOp, indexOps);
      });   

      outlinedParallelRegionKernel.walk([&](acc::LoopOp accLoopOp) {
        mapLoopToGrid(accLoopOp, indexOps);
      });

      // parallel private
      applyGangPrivateList(outlinedParallelRegionKernel, parallelOp, 
          gangPrivates);
      //removeUslessArguments(outlinedParallelRegionKernel, operands);

      auto kernelModule =
          createKernelModule(outlinedParallelRegionKernel, symbolTable);
      symbolTable.insert(kernelModule, insertPt);

      // for(Value p : gangPrivates) {
      //   operands.insert(p); 
      // }
      auto callOutlinedParallelRegion = createLaunchParallelRegion(
          parallelOp, outlinedParallelRegionKernel, operands.getArrayRef());

      modified = true;

      parallelOp.erase();
    }); // Walk over ParallelOp within the module
  }

  if (modified)
    getModule().setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                        UnitAttr::get(&getContext()));

  if (failed(applyPartialConversion(m, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertOpenACCToTargetPass() {
  return std::make_unique<OpenACCToTargetLoweringPass>();
}

static PassRegistration<OpenACCToTargetLoweringPass>
    pass("convert-openacc-to-target", "Convert OpenACC to target runtime");
