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

#define CUDA_REPORT_IF_ERROR(expr, sourceFile, sourceLine) \
  [sourceFile, sourceLine](CUresult result) { \
    if (!result) \
      return; \
    const char *name = nullptr; \
    cuGetErrorName(result, &name); \
    if (!name) \
      name = "<unknown>"; \
    fprintf(stderr, "'%s:%d:' '%s' failed with '%s'\n", sourceFile, \
        sourceLine, #expr, name); \
  }(expr)

namespace Fortran::runtime {

extern "C" {

int RTDECL(CUFAllocatableAllocate)(Descriptor &, int cudaAttr,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

int RTDECL(CUFAllocatableDeallocate)(Descriptor &, int cudaAttr,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_ALLOCATABLE_CUF_H_
