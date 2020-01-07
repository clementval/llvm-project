//==- OpenACCToGPU.td - OpenACC Ops to GPU Patterns ----------*- tablegen -*==//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines Patterns to lower OpenACC ops to the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOGPU_TD
#define MLIR_CONVERSION_OPENACCTOGPU_TD

#ifndef OPENACC_TD
include "mlir/Dialect/OpenACC/OpenACC.td"
#endif // OPENACC_TD

#endif // MLIR_CONVERSION_OPENACCTOGPU_TD
