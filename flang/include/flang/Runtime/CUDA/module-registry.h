//===-- runtime/CUDA/module-registry.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_CUDA_MODULE_REGISTRY_H_
#define FORTRAN_RUNTIME_CUDA_MODULE_REGISTRY_H_

#include "flang/Common/api-attrs.h"
#include <cuda.h>
#include <map>

#define MAX_MODULES 100

static constexpr int kModuleNotFound = -1;

namespace Fortran::runtime::cuda {

typedef struct {
  const char *name;
  CUmodule mod;
} ModuleRegistryEntry;

class ModuleRegistry {
public:
  void Load(const char *name, void* data);
  void Unload(const char *name);
  CUmodule GetModule(const char *name);

private:
  ModuleRegistryEntry modules_[MAX_MODULES];
  int FindModuleIndex(const char *name);
  unsigned moduleCount_ = 0;
};

extern ModuleRegistry moduleRegistry;

} // namespace Fortran::runtime::cuda

#endif // FORTRAN_RUNTIME_CUDA_MODULE_REGISTRY_H_
