//===- DialectRegistration.cpp - Register OpenACC dialect -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
using namespace mlir;
using namespace acc;

// Static initialization for OpenACC dialect registration.
static DialectRegistration<OpenACCDialect> OpenACC;
