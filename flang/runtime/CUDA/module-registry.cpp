//===-- runtime/CUDA/module-registry.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/module-registry.h"
#include "flang/Runtime/CUDA/common.h"
#include "../terminator.h"

namespace Fortran::runtime::cuda {

ModuleRegistry moduleRegistry;

thread_local static int32_t defaultDevice = 0;

CUdevice getDefaultCuDevice() {
  CUdevice device;
  CU_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  return device;
}

// This is a temporary solution until we have a way to correctly initialize
// the context.
class ScopedContext {
public:
  ScopedContext() {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [] {
      CU_REPORT_IF_ERROR(cuInit(/*flags=*/0));
      CUcontext ctx;
      // Note: this does not affect the current context.
      CU_REPORT_IF_ERROR(
          cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();

    CU_REPORT_IF_ERROR(cuCtxPushCurrent(context));
  }

  ~ScopedContext() { CU_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)); }
};

void ModuleRegistry::Load(const char *name, void *data) {
  ScopedContext scopedContext;
  CUmodule mod = nullptr;
  CU_REPORT_IF_ERROR(cuModuleLoadData(&mod, data));
  if (mod) {
    modules_[moduleCount_] = ModuleRegistryEntry{name, mod};
    ++moduleCount_;
  } else {
    Terminator terminator;
    terminator.Crash("Unable to load device module: %s", name);
  }
}

void ModuleRegistry::Unload(const char *name) {
  ScopedContext scopedContext;
  int idx = FindModuleIndex(name);
  if (idx != kModuleNotFound) {
    CU_REPORT_IF_ERROR(cuModuleUnload(modules_[idx].mod));
    modules_[idx].mod = nullptr;
  }
}

int ModuleRegistry::FindModuleIndex(const char *name) {
  // TODO: Use binary search
  for (unsigned i = 0; i < moduleCount_; ++i) {
    if (strcmp(modules_[i].name, name) == 0) {
      return i;
    }
  }
  return kModuleNotFound;
}

CUmodule ModuleRegistry::GetModule(const char *name) {
  int idx = FindModuleIndex(name);
  if (idx != kModuleNotFound) {
    return modules_[idx].mod;
  }
  Terminator terminator;
  terminator.Crash("Unable to retrieve device module: %s", name);
}


// int binarySearch(char *arr[], int size, const char *target) {
//     int left = 0;
//     int right = size - 1;

//     while (left <= right) {
//         int mid = left + (right - left) / 2;

//         int comparison = strcmp(arr[mid], target);
//         if (comparison == 0) {
//             return mid; // Target found
//         }
//         else if (comparison < 0) {
//             left = mid + 1; // Search right half
//         }
//         else {
//             right = mid - 1; // Search left half
//         }
//     }
//     return -1; // Target not found
// }

} // namespace Fortran::runtime::cuda
