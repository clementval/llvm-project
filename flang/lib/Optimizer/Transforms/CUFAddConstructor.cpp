//===-- CUFAddConstructor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFADDCONSTRUCTOR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static constexpr llvm::StringRef cudaFortranCtorName{
    "__cudaFortranConstructor"};

struct CUFAddConstructor
    : public fir::impl::CUFAddConstructorBase<CUFAddConstructor> {

  std::string getBinaryIdentifier(mlir::gpu::BinaryOp binary) {
    return binary.getSymName().str() + "_cubin_cst";
  }

  // Embed the object as a global string so we can manipulate it in the
  // constructor.
  mlir::LLVM::GlobalOp embedBinary(mlir::OpBuilder &builder,
                                   mlir::ModuleOp mod,
                                   mlir::gpu::BinaryOp binary) {
    mlir::gpu::ObjectAttr object =
        mlir::dyn_cast<mlir::gpu::ObjectAttr>(binary.getObjectsAttr().getValue()[0]);
    if (!object)
      return {};

    std::string binaryCst = getBinaryIdentifier(binary);
    
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(mod.getBody());
    auto type = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), object.getObject().getValue().size());
    return builder.create<mlir::LLVM::GlobalOp>(binary.getLoc(), type, /*isConstant=*/true,
                                         mlir::LLVM::Linkage::Internal, binaryCst,
                                         builder.getStringAttr(object.getObject().getValue()),
                                         /*alignment=*/8);
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             llvm::StringRef name,
                                             llvm::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              mlir::LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::SymbolTable symTab(getOperation());
    mlir::OpBuilder builder{mod.getBodyRegion()};
    builder.setInsertionPointToEnd(mod.getBody());
    mlir::Location loc = mod.getLoc();
    auto *ctx = mod.getContext();
    auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto funcTy =
        mlir::LLVM::LLVMFunctionType::get(voidTy, {}, /*isVarArg=*/false);

    // Symbol reference to CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    auto registerFuncOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        loc, RTNAME_STRING(CUFRegisterAllocator), funcTy);
    registerFuncOp.setVisibility(mlir::SymbolTable::Visibility::Private);

    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    auto registerModuleFuncTy =
        mlir::LLVM::LLVMFunctionType::get(voidTy, {llvmPtrTy, llvmPtrTy}, /*isVarArg=*/false);
    auto registerModuleFuncOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        loc, RTNAME_STRING(CUFRegisterModule), registerModuleFuncTy);
    registerModuleFuncOp.setVisibility(mlir::SymbolTable::Visibility::Private);

    auto cufRegisterAllocatorRef = mlir::SymbolRefAttr::get(
        mod.getContext(), RTNAME_STRING(CUFRegisterAllocator));
    builder.setInsertionPointToEnd(mod.getBody());

    // Create the constructor function that call CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, cudaFortranCtorName,
                                                       funcTy);
    func.setLinkage(mlir::LLVM::Linkage::Internal);
    builder.setInsertionPointToStart(func.addEntryBlock(builder));

    // Call to register the allocators.
    builder.create<mlir::LLVM::CallOp>(loc, funcTy, cufRegisterAllocatorRef);

    // Register the CUDA binary module if any.
    if (auto gpuBin = symTab.lookup<mlir::gpu::BinaryOp>("cuda_device_mod")) {
      std::string binaryIdentifier = getBinaryIdentifier(gpuBin);
      mlir::LLVM::GlobalOp global = embedBinary(builder, mod, gpuBin);
      if (!global)
        signalPassFailure();

      auto cufRegisterModule = mlir::SymbolRefAttr::get(
        mod.getContext(), RTNAME_STRING(CUFRegisterModule));

      auto deviceModName = getOrCreateGlobalString(loc, builder, "cuda_device_mod_name_cst", "cuda_device_mod", mod);

      mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
      mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
      mlir::Value gep = builder.create<mlir::LLVM::GEPOp>(
          loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
          globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));


      builder.create<mlir::LLVM::CallOp>(loc, funcTy, cufRegisterModule, llvm::ArrayRef<mlir::Value>({deviceModName, gep}));
    } else {
      llvm::errs() << "NO GPU.BINARY\n";
    }
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});

    // Create the llvm.global_ctor with the function.
    // TODO: We might want to have a utility that retrieve it if already created
    // and adds new functions.
    builder.setInsertionPointToEnd(mod.getBody());
    llvm::SmallVector<mlir::Attribute> funcs;
    funcs.push_back(
        mlir::FlatSymbolRefAttr::get(mod.getContext(), func.getSymName()));
    llvm::SmallVector<int> priorities;
    priorities.push_back(0);
    builder.create<mlir::LLVM::GlobalCtorsOp>(
        mod.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities));
  }
};

} // end anonymous namespace
