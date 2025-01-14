//===-- CUFLegalizeRegion.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/allocatable.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"

namespace fir {
#define GEN_PASS_DEF_CUFLEGALIZEREGION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

template <typename OpTy>
static void legalizeGlobalDescriptors(OpTy op, mlir::SymbolTable &symTab,
                                      fir::FirOpBuilder &builder) {
  mlir::Region &cufRegion = op->getRegion(0);
  llvm::SetVector<mlir::Value> liveInValues;
  mlir::getUsedValuesDefinedAbove(cufRegion, liveInValues);
  for (mlir::Value val : liveInValues) {
    if (auto declareOp = val.getDefiningOp<fir::DeclareOp>()) {
      if (auto addrOfOp =
              declareOp.getMemref().getDefiningOp<fir::AddrOfOp>()) {
        if (auto global = symTab.lookup<fir::GlobalOp>(
                addrOfOp.getSymbol().getRootReference().getValue())) {
          if (cuf::isRegisteredDeviceGlobal(global)) {
            builder.setInsertionPoint(op);
            mlir::Value devAddr =
                builder
                    .create<cuf::DeviceAddressOp>(
                        op.getLoc(), addrOfOp.getType(), addrOfOp.getSymbol())
                    .getResult();
            replaceAllUsesInRegionWith(val, devAddr, cufRegion);
          }
        }
      }
    }
  }
}

class CUFLegalizeRegion
    : public fir::impl::CUFLegalizeRegionBase<CUFLegalizeRegion> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder opBuilder{mod.getBodyRegion()};
    fir::FirOpBuilder builder(opBuilder, mod);
    if (!mod)
      return signalPassFailure();
    mlir::SymbolTable symTab(mod);
    op->walk([&](cuf::KernelOp kernelOp) {
      legalizeGlobalDescriptors(kernelOp, symTab, builder);
      return mlir::WalkResult::advance();
    });
  }
};
} // namespace
