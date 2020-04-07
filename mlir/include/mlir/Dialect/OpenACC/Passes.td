//===-- Passes.td - OpenACC pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LOOP_PASSES
#define MLIR_DIALECT_LOOP_PASSES

include "mlir/Pass/PassBase.td"

def OpenACCToStandard : Pass<"convert-openacc-to-gpu"> {
  let summary = "OpenACC to GPU";
  let constructor = "mlir::createConvertOpenACCToGPUPass()";
}

def OpenACCToGpu : Pass<"convert-openacc-to-standard"> {
  let summary = "OpenACC to Standard";
  let constructor = "mlir::createConvertOpenACCToStandardPass()";
}

#endif // MLIR_DIALECT_LOOP_PASSES
