//===- ConvertOpenACCToLLVM.h - OpenACC conversion pass entrypoint --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVMPASS_H_
#define MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVMPASS_H_

#include<memory>

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename T>
class OperationPass;
class OwningRewritePatternList;

/// Collect the patterns to convert from the OpenACC dialect LLVMIR dialect.
void populateOpenACCToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                             OwningRewritePatternList &patterns);

/// Create a pass to convert the OpenACC dialect into the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenACCToLLVMPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVMPASS_H_
