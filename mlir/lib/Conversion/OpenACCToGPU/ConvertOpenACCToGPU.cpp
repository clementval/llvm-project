//===- ConvertOpenACCToGPU.cpp - Convert OpenACC to GPU dialect -----------===//
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

#include "mlir/Conversion/OpenACCToGPU/ConvertOpenACCToGPU.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
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

struct OpenACCToGPULoweringPass : public ModulePass<OpenACCToGPULoweringPass> {
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

struct SequentialLoopOpNesting final : public OpRewritePattern<acc::LoopOp> {
  using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(acc::LoopOp accLoopOp,
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

  for (Value v : allOperands) {
    bool isInPrivate = false;
    for (Value p : parallelOp.getGangPrivates()) {
      if (v == p) {
        isInPrivate = true;
        privates.insert(v);
        break;
      }
    }
    if (!isInPrivate) {
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
  auto outlinedParallelRegion =
      outlineParallelRegion(parallelOp, operands, privates);
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
// SequentialLoopOpNesting::matchAndRewrite(acc::LoopOp accLoopOp,
//                                          PatternRewriter &rewriter) const {
//   if(accLoopOp.isSeq()) {
//     auto parentOp = accLoopOp.getParentOfType<acc::LoopOp>();
//     if(!parentOp)
//       return matchFailure();

//     auto gangRedundantOp = accLoopOp.getParentOfType<acc::GangRedundantOp>();
//     if(gangRedundantOp)
//       return matchFailure();

//     Location loc = accLoopOp.getLoc();
//     auto gangRedOp = rewriter.create<acc::GangRedundantOp>(loc);
//     rewriter.setInsertionPointToStart(&gangRedOp.getBody().front());
//     auto gangRedundantTerminator =
//     rewriter.create<acc::GangRedundantEndOp>(loc);
//     accLoopOp.getOperation()->moveBefore(gangRedundantTerminator);
//     return matchSuccess();
//   }

//   return matchFailure();
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
    if (failed(collapseNestedLoops(forOp, accLoopOp)))
      return failure();
    accLoopOp.removeAttr(acc::LoopOp::getCollapseAttrName());
  }
  return success();
}

// Hoist opeartion inserted at the beginning of the LoopOp region until the
// ForOp outside of the LoopOp region.
template <typename StructureOp>
static void hoistOpBeforeOperation(StructureOp &parentOp,
                                   acc::LoopOp &accLoopOp) {
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
  kernelFunc.walk([](ReturnOp op) {
    OpBuilder replacer(op);
    replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return kernelFunc;
}

///
///
static gpu::LaunchFuncOp createLaunchParallelRegion(
    acc::ParallelOp parallelOp, gpu::GPUFuncOp outlinedParallelRegionKernel,
    Value numGangs, Value numWorkers, ValueRange operands) {
  OpBuilder builder(parallelOp);
  Value constOne = builder.create<ConstantOp>(
      parallelOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(), 1));

  auto callOutlinedParallelRegionKernel = builder.create<gpu::LaunchFuncOp>(
      parallelOp.getLoc(), outlinedParallelRegionKernel, numGangs, constOne,
      constOne, numWorkers, constOne, constOne, operands);
  // TODO this seems to be broken
  // inlineBeneficiaryOps(outlinedParallelRegionKernel,
  //                      callOutlinedParallelRegionKernel);
  return callOutlinedParallelRegionKernel;
}

// TODO grab the one from KernelOutining if possible
gpu::GPUModuleOp OpenACCToGPULoweringPass::createKernelModule(
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
// static constexpr unsigned THREAD_ID_Y = 4;
// static constexpr unsigned BLOCK_DIM_Y = 5;

static loop::IfOp createGangRedundantWrapper(OpBuilder &builder, Location loc,
                                             SmallVector<Value, 4> &indexOps) {
  assert(indexOps.size() == 4 && "Expecting 4 indexes to map loop");

  // if threadIdx.x == 0 then
  Value const0 = builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), 0));
  Value isThread0 = builder.create<CmpIOp>(loc, CmpIPredicate::eq,
                                           indexOps[THREAD_ID_X], const0);
  auto ifOp = builder.create<loop::IfOp>(loc, isThread0, false);

  // Add a barrier to sync all worker after the seq loop
  builder.setInsertionPointAfter(ifOp);
  builder.create<gpu::BarrierOp>(loc);

  return ifOp;
}

static void mapLoopToGrid(acc::LoopOp accLoopOp,
                          SmallVector<Value, 4> &indexOps,
                          SmallVector<loop::ForOp, 4> &forOps) {
  assert(indexOps.size() == 4 && "Expecting 4 indexes to map loop");

  if (auto forOp = dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
    OpBuilder builder(forOp);
    Location loc(accLoopOp.getLoc());
    if (accLoopOp.isSeq()) {
      if (accLoopOp.isGangRedundant()) {
        loop::IfOp wrapper = createGangRedundantWrapper(builder, loc, indexOps);
        forOp.getOperation()->moveBefore(
            wrapper.thenRegion().back().getTerminator());
      }
    } else if (accLoopOp.isGangVector()) {

      // lb = blockIdx.x * blockDim.x + threadIdx.x
      Value tmp = builder.create<MulIOp>(loc, indexOps[BLOCK_ID_X],
                                         indexOps[BLOCK_DIM_X]);
      Value lb = builder.create<AddIOp>(loc, tmp, indexOps[THREAD_ID_X]);
      forOp.setLowerBound(lb);

      // step = gridDim.x * blockDim.x
      Value step = builder.create<MulIOp>(loc, indexOps[GRID_DIM_X],
                                          indexOps[BLOCK_DIM_X]);
      forOp.setStep(step);
      forOps.push_back(forOp);
    } else if (accLoopOp.isGang()) {
      forOp.setLowerBound(indexOps[BLOCK_ID_X]);
      forOp.setStep(indexOps[GRID_DIM_X]);
      forOps.push_back(forOp);
    } else if (accLoopOp.isVector() || accLoopOp.isWorker()) {
      forOp.setLowerBound(indexOps[THREAD_ID_X]);
      forOp.setStep(indexOps[BLOCK_DIM_X]);
      forOps.push_back(forOp);
    }
  } else {
  }

  accLoopOp.walk([](acc::LoopEndOp op) { op.erase(); });
  extractRegionBeforeItself(accLoopOp);
  accLoopOp.erase();
}

static void transformGangRedundant(acc::GangRedundantOp accGangRedundantOp,
                                   SmallVector<Value, 4> &indexOps) {

  OpBuilder builder(accGangRedundantOp);
  Location loc = accGangRedundantOp.getLoc();

  loop::IfOp wrapper = createGangRedundantWrapper(builder, loc, indexOps);

  wrapper.thenRegion().takeBody(accGangRedundantOp.getBody());

  builder.setInsertionPointAfter(wrapper);
  accGangRedundantOp.erase();
}

static void applyGangPrivateList(gpu::GPUFuncOp outlinedParallelRegion,
                                 acc::ParallelOp accParallelOp,
                                 llvm::SetVector<Value> &privates) {
  if (accParallelOp.getNumGangPrivates() == 0)
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

  assert(accParallelOp.getNumGangPrivates() == privates.size() &&
         "Number of private variable doesn't match");

  for (auto p : accParallelOp.getGangPrivates()) {
    auto type = p.getType().dyn_cast<MemRefType>();
    assert(type && type.hasStaticShape() && "can only privatize memrefs");

    Value newPrivate = outlinedParallelRegion.addWorkgroupAttribution(
        type.getShape(), type.getElementType());

    replaceAllUsesInRegionWith(p, newPrivate, outlinedParallelRegion.getBody());
  }
}

static void removeUslessArguments(gpu::GPUFuncOp outlinedParallelRegionKernel,
                                  llvm::SetVector<Value> &operands) {
  assert(outlinedParallelRegionKernel.getNumArguments() == operands.size() &&
         "operands size must be the same as number of arguments");
  auto &firstBlock = outlinedParallelRegionKernel.getBody().front();
  for (int i = firstBlock.getNumArguments() - 1; i >= 0; --i) {
    Value arg = firstBlock.getArgument(i);
    if (arg.use_empty()) {
      firstBlock.eraseArgument(i);
      operands.remove(operands[i]);
    }
  }
}

template <typename OpTy>
static void createGPUIndexOperationForX(OpBuilder &builder, Location loc,
                                        SmallVectorImpl<Value> &values) {
  Value v = builder.create<OpTy>(loc, builder.getIndexType(),
                                 builder.getStringAttr("x"));
  values.push_back(v);
}

static void injectGpuIndexOperationsForX(Location loc, Region &body,
                                         SmallVector<Value, 4> &indexOps) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = body.front();
  builder.setInsertionPointToStart(&firstBlock);
  createGPUIndexOperationForX<gpu::BlockIdOp>(builder, loc, indexOps);
  createGPUIndexOperationForX<gpu::ThreadIdOp>(builder, loc, indexOps);
  createGPUIndexOperationForX<gpu::GridDimOp>(builder, loc, indexOps);
  createGPUIndexOperationForX<gpu::BlockDimOp>(builder, loc, indexOps);
}

/// Perform analysis on the LoopOp to determine its execution mode.
static void analysisLoopOp(acc::LoopOp accLoopOp,
                           SmallVector<Value, 4> &lowerBounds,
                           SmallVector<Value, 4> &upperBounds,
                           SmallVector<Value, 4> &steps) {
  if (accLoopOp.isSeq()) {
    auto parentOp = accLoopOp.getParentOfType<acc::LoopOp>();
    if (!parentOp)
      accLoopOp.setAttr(acc::LoopOp::getGangRedundantAttrName(),
                        UnitAttr::get(accLoopOp.getOperation()->getContext()));
  } else {
    if (auto forOp =
            dyn_cast<loop::ForOp>(accLoopOp.getBody().front().front())) {
      lowerBounds.push_back(forOp.lowerBound());
      upperBounds.push_back(forOp.upperBound());
      steps.push_back(forOp.step());
    }
    // TODO handle more cases
  }
}

///
static Value ceilDivPositive(OpBuilder &builder, Location loc, Value dividend,
                             int64_t divisor) {
  assert(divisor > 0 && "expected positive divisor");
  assert(dividend.getType().isIndex() && "expected index-typed value");
  Value divisorMinusOneCst = builder.create<ConstantIndexOp>(loc, divisor - 1);
  Value divisorCst = builder.create<ConstantIndexOp>(loc, divisor);
  Value sum = builder.create<AddIOp>(loc, dividend, divisorMinusOneCst);
  return builder.create<SignedDivIOp>(loc, sum, divisorCst);
}

///
///
static Value generateGangValue(OpBuilder &builder, Location loc,
                               Value lowerBound, Value upperBound, Value step,
                               int64_t vectorLength) {
  // upper bound - lower bound
  Value diffUpLb = builder.create<SubIOp>(loc, upperBound, lowerBound);
  // ceildiv(diffUpLb, step)
  Value numberOfIterations = ceilDivPositive(builder, loc, diffUpLb, step);
  // ceildiv(numberOfIterations, vectorLength)
  Value nbOfGangs =
      ceilDivPositive(builder, loc, numberOfIterations, vectorLength);
  return nbOfGangs;
}

/// Generate the number of gang value used to launch kernel
/// Take value defined as attribute or dereived avlue from loops inside the
/// parallel region.
static Value generateNumGangs(acc::ParallelOp parallelOp,
                              SmallVector<Value, 4> lowerBounds,
                              SmallVector<Value, 4> upperBounds,
                              SmallVector<Value, 4> steps,
                              int64_t vectorLength) {
  OpBuilder builder(parallelOp);
  Location loc = parallelOp.getLoc();
  if (parallelOp.hasNumGangs())
    return builder.create<ConstantOp>(
        parallelOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(),
                                                    parallelOp.getNumGangs()));

  if (upperBounds.size() == 0)
    return builder.create<ConstantOp>(
        parallelOp.getLoc(), builder.getIntegerAttr(builder.getIndexType(), 1));

  // TODO handle more than one loop
  return generateGangValue(builder, loc, lowerBounds.front(),
                           upperBounds.front(), steps.front(), vectorLength);
}

static Value generateNumWorkers(acc::ParallelOp parallelOp,
                                SmallVector<Value, 4> lowerBounds,
                                SmallVector<Value, 4> upperBounds,
                                SmallVector<Value, 4> steps,
                                int64_t vectorLength) {
  OpBuilder builder(parallelOp);
  Location loc = parallelOp.getLoc();
  if (parallelOp.hasNumWorkers())
    return builder.create<ConstantOp>(
        loc, builder.getIntegerAttr(builder.getIndexType(),
                                    parallelOp.getNumWorkers()));

  if (lowerBounds.size() == 0)
    return builder.create<ConstantOp>(
        loc, builder.getIntegerAttr(builder.getIndexType(), 1));

  return builder.create<ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), vectorLength));
}

void OpenACCToGPULoweringPass::runOnModule() {

  ConversionTarget target(getContext());
  target.addIllegalDialect<acc::OpenACCDialect>();
  target.addLegalDialect<gpu::GPUDialect>();

  // target.addLegalOp<acc::ParallelOp>();
  // target.addLegalOp<acc::ParallelEndOp>();
  // target.addLegalOp<acc::LoopOp>();
  target.addLegalOp<acc::LoopEndOp>();
  target.addLegalOp<acc::GangRedundantEndOp>();

  // If operation is considered legal the rewrite pattern in not called.
  OwningRewritePatternList patterns;
  // patterns.insert<SequentialLoopOpNesting>(&getContext());
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

      // Perform some analysis on loops before applying transformation
      SmallVector<Value, 4> lowerBounds;
      SmallVector<Value, 4> upperBounds;
      SmallVector<Value, 4> steps;
      parallelOp.walk([&](acc::LoopOp accLoopOp) {
        analysisLoopOp(accLoopOp, lowerBounds, upperBounds, steps);
      });

      llvm::SetVector<Value> operands;
      llvm::SetVector<Value> gangPrivates;
      auto outlinedParallelRegion =
          outlineParallelRegion(parallelOp, operands, gangPrivates);
      auto outlinedParallelRegionKernel =
          convertOutlinedParallelRegionToKernel(outlinedParallelRegion);

      SmallVector<Value, 4> indexOps;
      injectGpuIndexOperationsForX(parallelOp.getLoc(),
                                   outlinedParallelRegionKernel.getBody(),
                                   indexOps);

      outlinedParallelRegionKernel.walk(
          [&](acc::GangRedundantOp accGangRedundantOp) {
            transformGangRedundant(accGangRedundantOp, indexOps);
          });

      // Collapse acc.loop if necessary
      outlinedParallelRegionKernel.walk([&](acc::LoopOp accLoopOp) {
        applyCollapseClause(accLoopOp);
        hoistOpBeforeOperation(accLoopOp, accLoopOp);
      });

      SmallVector<loop::ForOp, 4> forOps;
      outlinedParallelRegionKernel.walk([&](acc::LoopOp accLoopOp) {
        mapLoopToGrid(accLoopOp, indexOps, forOps);
      });

      // parallel private
      applyGangPrivateList(outlinedParallelRegionKernel, parallelOp,
                           gangPrivates);
      // removeUslessArguments(outlinedParallelRegionKernel, operands);

      auto kernelModule =
          createKernelModule(outlinedParallelRegionKernel, symbolTable);
      symbolTable.insert(kernelModule, insertPt);

      Value numGangs =
          generateNumGangs(parallelOp, lowerBounds, upperBounds, steps, 128);
      Value numWorkers =
          generateNumWorkers(parallelOp, lowerBounds, upperBounds, steps, 128);
      createLaunchParallelRegion(parallelOp, outlinedParallelRegionKernel,
                                 numGangs, numWorkers, operands.getArrayRef());

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

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertOpenACCToGPUPass() {
  return std::make_unique<OpenACCToGPULoweringPass>();
}

static PassRegistration<OpenACCToGPULoweringPass>
    pass("convert-openacc-to-gpu", "Convert OpenACC to GPU dialect");
