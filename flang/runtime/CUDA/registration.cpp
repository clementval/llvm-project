//===-- runtime/CUDA/registration.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/registration.h"
#include "flang/Runtime/CUDA/module-registry.h"
#include <cstdio>

namespace Fortran::runtime::cuda {

extern "C" {

void RTDEF(CUFRegisterModule)(const char *name, void* data) {
  moduleRegistry.Load(name, data);
}

void RTDEF(CUFUnregisterModule)(const char *name) {
  moduleRegistry.Unload(name);
}

}
} // namespace Fortran::runtime::cuda
