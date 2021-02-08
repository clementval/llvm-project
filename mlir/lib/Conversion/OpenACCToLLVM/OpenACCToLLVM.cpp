//===- OpenACCToLLVM.cpp - conversion from OpenACC to LLVM dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ConvertOpenACCToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

#include "llvm/IR/GlobalValue.h"

using namespace mlir;
using namespace mlir::acc;


static LLVM::LLVMStructType getKmpIdentType(MLIRContext *ctx) {
  // The ident structure that describes a source location from kmp.h. with
  // source location string data as ";filename;function;line;column;;\0".
  // struct ident_t {
  //   // Ident_t flags described in kmp.h.
  //   int32_t reserved_1;
  //   int32_t flags;
  //   int32_t reserved_2;
  //   int32_t reserved_3;
  //   char const *psource;
  // };
  auto i8Ty = IntegerType::get(ctx, 8);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i8Ptr = LLVM::LLVMPointerType::get(i8Ty);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty, i32Ty, i32Ty, i8Ptr});
}

static LLVM::LLVMFunctionType getTgtTargetDataBeginMapperType(MLIRContext *ctx) {
  auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
  auto identType = getKmpIdentType(ctx);
  auto identTypePtr = LLVM::LLVMPointerType::get(identType);
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  // return LLVM::LLVMFunctionType::get(llvmVoidType, {identTypePtr, i64Ty, i32Ty});
  return LLVM::LLVMFunctionType::get(llvmVoidType, {i64Ty, i32Ty});
}


struct EnterDataOpConversion : public ConvertOpToLLVMPattern<EnterDataOp> {
  using ConvertOpToLLVMPattern<EnterDataOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(EnterDataOp enterDataOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO handle async and wait info

    // __tgt_target_data_begin_mapper(ident_t *loc, int64_t device_id,
    //                                int32_t arg_num, void **args_base,
    //                                void **args, int64_t *arg_sizes,
    //                                int64_t *arg_types,
    //                                map_var_info_t *arg_names,
    //                                void **arg_mappers);
    auto loc = enterDataOp->getLoc();
    auto module = enterDataOp->getParentOfType<ModuleOp>();
    MLIRContext *ctx = module.getContext();

    auto mapperFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        "__tgt_target_data_begin_mapper");
    if (!mapperFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto mapperFuncTy = getTgtTargetDataBeginMapperType(ctx);
      mapperFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
          "__tgt_target_data_begin_mapper", mapperFuncTy);
    }

    auto deviceId = rewriter.create<LLVM::ConstantOp>(enterDataOp->getLoc(),
        rewriter.getI64Type(), rewriter.getI64IntegerAttr(-1));

    auto argNum = rewriter.create<LLVM::ConstantOp>(enterDataOp->getLoc(),
        rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(enterDataOp.createOperands().size()));

    auto structType = getKmpIdentType(ctx);
    auto structPtrType = LLVM::LLVMPointerType::get(structType);

    //
    OpBuilder moduleBuilder(module.getBodyRegion(), rewriter.getListener());
    std::string name(";unknown;unknown;0;0;;");
    auto globalStringType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), name.size());
    auto global = moduleBuilder.create<LLVM::GlobalOp>(
      loc, globalStringType, /*isConstant=*/true, LLVM::Linkage::Private, "dummy",
      rewriter.getStringAttr(name));
    global.unnamed_addrAttr(rewriter.getI64IntegerAttr(2)); // UnnamedAddr::Global


// Constant *I32Null = ConstantInt::getNullValue(Int32);
//     Constant *IdentData[] = {
//         I32Null, ConstantInt::get(Int32, uint32_t(LocFlags)),
//         ConstantInt::get(Int32, Reserve2Flags), I32Null, SrcLocStr};



    // auto callOp = rewriter.create<LLVM::CallOp>(loc, mapperFunc, ValueRange{structPtr, deviceId, argNum});
    auto callOp = rewriter.create<LLVM::CallOp>(loc, mapperFunc, ValueRange{deviceId, argNum});

    rewriter.eraseOp(enterDataOp);

    return success();
  }
};


void mlir::populateOpenACCToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<EnterDataOpConversion>(converter);
}

namespace {
struct ConvertOpenACCToLLVMPass
    : public ConvertOpenACCToLLVMBase<ConvertOpenACCToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to OpenMP operations with LLVM IR dialect
  OwningRewritePatternList patterns;
  LLVMTypeConverter converter(&getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateOpenACCToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  // target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
  //     [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
  // target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
  //                   omp::BarrierOp, omp::TaskwaitOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenACCToLLVMPass() {
  return std::make_unique<ConvertOpenACCToLLVMPass>();
}
