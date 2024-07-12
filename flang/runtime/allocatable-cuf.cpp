//===-- runtime/allocatable.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/allocatable-cuf.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"
#include "flang/Common/Fortran.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/descriptor.h"

#include "cuda.h"

namespace Fortran::runtime {
extern "C" {

static int AllocateCuda(Fortran::runtime::Descriptor &desc,
    Fortran::common::CUDADataAttr attr, const char *sourceFile,
    int sourceLine) {
  std::size_t elementBytes{desc.ElementBytes()};
  if (static_cast<std::int64_t>(elementBytes) < 0) {
    // F'2023 7.4.4.2 p5: "If the character length parameter value evaluates
    // to a negative value, the length of character entities declared is zero."
    elementBytes = desc.raw().elem_len = 0;
  }
  std::size_t byteSize{desc.Elements() * elementBytes};
  // Zero size allocation is possible in Fortran and the resulting
  // descriptor must be allocated/associated. Since std::malloc(0)
  // result is implementation defined, always allocate at least one byte.
  if (byteSize == 0) {
    byteSize = 1;
  }
  if (attr == Fortran::common::CUDADataAttr::Pinned) {
    void *p;
    CUDA_REPORT_IF_ERROR(cuMemAllocHost(&p, byteSize), sourceFile, sourceLine);
    if (!p) {
      return CFI_ERROR_MEM_ALLOCATION;
    }
    desc.raw().base_addr = reinterpret_cast<void *>(p);
  } else {
    CUdeviceptr p = 0;
    if (attr == Fortran::common::CUDADataAttr::Device) {
      CUDA_REPORT_IF_ERROR(cuMemAlloc(&p, byteSize), sourceFile, sourceLine);
    } else if (attr == Fortran::common::CUDADataAttr::Managed ||
        attr == Fortran::common::CUDADataAttr::Unified) {
      CUDA_REPORT_IF_ERROR(
          cuMemAllocManaged(&p, byteSize, CU_MEM_ATTACH_GLOBAL), sourceFile,
          sourceLine);
    }
    if (!p) {
      return CFI_ERROR_MEM_ALLOCATION;
    }
    desc.raw().base_addr = reinterpret_cast<void *>(p);
  }
  desc.SetByteStrides();
  return 0;
}

int RTDEF(CUFAllocatableAllocate)(Descriptor &descriptor, int cudaAttr,
    bool hasStat, const Descriptor *errMsg, const char *sourceFile,
    int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  } else if (descriptor.IsAllocated()) {
    return ReturnError(terminator, StatBaseNotNull, errMsg, hasStat);
  } else {
    int stat{ReturnError(terminator,
        AllocateCuda(descriptor, (Fortran::common::CUDADataAttr)cudaAttr,
            sourceFile, sourceLine),
        errMsg, hasStat)};
    if (stat == StatOk) {
      if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
        if (const auto *derived{addendum->derivedType()}) {
          if (!derived->noInitializationNeeded()) {
            if ((Fortran::common::CUDADataAttr)cudaAttr ==
                Fortran::common::CUDADataAttr::Pinned) {
              stat =
                  Initialize(descriptor, *derived, terminator, hasStat, errMsg);
            } else {
              // TODO
              terminator.Crash("initialization of derived type on device not "
                               "yet implemented");
            }
          }
        }
      }
    }
    return stat;
  }
}

static int deallocate(Fortran::ISO::CFI_cdesc_t *descriptor,
    const char *sourceFile, int sourceLine) {
  if (!descriptor) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->version != CFI_VERSION) {
    return CFI_INVALID_DESCRIPTOR;
  }
  if (descriptor->attribute != CFI_attribute_allocatable &&
      descriptor->attribute != CFI_attribute_pointer) {
    // Non-interoperable object
    return CFI_INVALID_DESCRIPTOR;
  }
  if (!descriptor->base_addr) {
    return CFI_ERROR_BASE_ADDR_NULL;
  }
  CUDA_REPORT_IF_ERROR(
      cuMemFree(reinterpret_cast<CUdeviceptr>(descriptor->base_addr)),
      sourceFile, sourceLine);
  descriptor->base_addr = nullptr;
  return CFI_SUCCESS;
}

static int DeallocateCuda(Fortran::runtime::Descriptor &descriptor,
    Fortran::common::CUDADataAttr attr,
    Fortran::runtime::Terminator &terminator, const char *sourceFile,
    int sourceLine, bool finalize, bool destroyPointers) {
  if (!destroyPointers && descriptor.raw().attribute == CFI_attribute_pointer) {
    return Fortran::runtime::StatOk;
  } else {
    if (auto *addendum{descriptor.Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noDestructionNeeded()) {
          if (attr == Fortran::common::CUDADataAttr::Pinned) {
            Fortran::runtime::Destroy(
                descriptor, finalize, *derived, &terminator);
          } else {
            // TODO
            terminator.Crash(
                "destruction of derived type on device not yet implemented");
          }
        }
      }
    }
    return deallocate(&descriptor.raw(), sourceFile, sourceLine);
  }
}

int RTNAME(CUFAllocatableDeallocate)(Fortran::runtime::Descriptor &descriptor,
    int cudaAttr, bool hasStat, const Fortran::runtime::Descriptor *errMsg,
    const char *sourceFile, int sourceLine) {
  Fortran::runtime::Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(
        terminator, Fortran::runtime::StatInvalidDescriptor, errMsg, hasStat);
  } else if (!descriptor.IsAllocated()) {
    return ReturnError(
        terminator, Fortran::runtime::StatBaseNull, errMsg, hasStat);
  } else {
    return ReturnError(terminator,
        DeallocateCuda(descriptor, (Fortran::common::CUDADataAttr)cudaAttr,
            terminator, sourceFile, sourceLine,
            /*finalize=*/true, /*destroyPointers=*/false),
        errMsg, hasStat);
  }
}
}
} // namespace Fortran::runtime
