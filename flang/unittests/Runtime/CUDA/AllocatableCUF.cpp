//===-- flang/unittests/Runtime/AllocatableCUF.cpp ---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/Fortran.h"
#include "flang/Runtime/allocatable-cuf.h"
#include "flang/Runtime/allocatable.h"

#include "cuda.h"

using namespace Fortran::runtime;

static OwningPtr<Descriptor> createAllocatable(
    Fortran::common::TypeCategory tc, int kind, int rank = 1) {
  return Descriptor::Create(TypeCode{tc, kind}, kind, nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

thread_local static int32_t defaultDevice = 0;

CUdevice getDefaultCuDevice() {
  CUdevice device;
  const char *sourceFile = "";
  int sourceLine = 0;
  CUDA_REPORT_IF_ERROR(
      cuDeviceGet(&device, /*ordinal=*/defaultDevice), sourceFile, sourceLine);
  return device;
}

class ScopedContext {
public:
  ScopedContext(const char *sourceFile, int sourceLine) {
    // Static reference to CUDA primary context for device ordinal
    // defaultDevice.
    static CUcontext context = [sourceFile, sourceLine] {
      CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0), sourceFile, sourceLine);
      CUcontext ctx;
      // Note: this does not affect the current context.
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()),
          sourceFile, sourceLine);
      return ctx;
    }();

    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(context), sourceFile, sourceLine);
  }

  ~ScopedContext() {
    const char *sourceFile = "";
    int sourceLine = 0;
    CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr), sourceFile, sourceLine);
  }
};

TEST(AllocatableCUFTest, SimpleDeviceAllocate) {
  using Fortran::common::TypeCategory;
  ScopedContext ctx(__FILE__, __LINE__);
  // REAL(4), DEVICE, ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Real, 4)};
  RTNAME(AllocatableSetBounds)(*a, 0, 1, 10);
  RTNAME(CUFAllocatableAllocate)
  (*a, (int)Fortran::common::CUDADataAttr::Device, /*hasStat=*/false,
      /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  RTNAME(CUFAllocatableDeallocate)
  (*a, (int)Fortran::common::CUDADataAttr::Device, /*hasStat=*/false,
      /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
}

TEST(AllocatableCUFTest, SimplePinnedAllocate) {
  using Fortran::common::TypeCategory;
  ScopedContext ctx(__FILE__, __LINE__);
  // INTEGER(4), PINNED, ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Integer, 4)};
  RTNAME(AllocatableSetBounds)(*a, 0, 1, 10);
  RTNAME(CUFAllocatableAllocate)
  (*a, (int)Fortran::common::CUDADataAttr::Pinned, /*hasStat=*/false,
      /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  RTNAME(CUFAllocatableDeallocate)
  (*a, (int)Fortran::common::CUDADataAttr::Pinned, /*hasStat=*/false,
      /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
}
