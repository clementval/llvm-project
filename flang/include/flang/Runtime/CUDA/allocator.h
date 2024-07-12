//===-- include/flang/Runtime/allocatable-cuf.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for Fortran runtime library support of code generated
// to manipulate and query allocatable with CUDA Fortran attribute.
#ifndef FORTRAN_RUNTIME_ALLOCATABLE_CUF_H_
#define FORTRAN_RUNTIME_ALLOCATABLE_CUF_H_

#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/entry-names.h"

static constexpr unsigned kPinnedAllocatorPos = 1;
static constexpr unsigned kDeviceAllocatorPos = 2;
static constexpr unsigned kManagedAllocatorPos = 3;

#define CUDA_REPORT_IF_ERROR(expr) \
  [](CUresult result) { \
    if (!result) \
      return; \
    const char *name = nullptr; \
    cuGetErrorName(result, &name); \
    if (!name) \
      name = "<unknown>"; \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name); \
  }(expr)

namespace Fortran::runtime {

extern "C" {

void CUFRegisterAllocator();

void *CUFAllocPinned(std::size_t);
void CUFFreePinned(void*);

void *CUFAllocDevice(std::size_t);
void CUFFreeDevice(void*);

void *CUFAllocManaged(std::size_t);
void CUFFreeManaged(void*);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ALLOCATABLE_CUF_H_
