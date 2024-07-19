//===-- runtime/allocator.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ALLOCATOR_H_
#define FORTRAN_RUNTIME_ALLOCATOR_H_

#include "flang/Common/api-attrs.h"
#include <cstdlib>
#include <vector>

#define MAX_ALLOCATOR 5

namespace Fortran::runtime {

using AllocFct = void *(*)(std::size_t);
using FreeFct = void (*)(void *);

typedef struct Allocator_t {
  AllocFct alloc = nullptr;
  FreeFct free = nullptr;
} Allocator_t;

struct AllocatorRegistry {
  constexpr AllocatorRegistry() { allocators[0] = {&std::malloc, &std::free}; };
  void Register(int, Allocator_t);
  AllocFct GetAllocator(int pos);
  FreeFct GetDeallocator(int pos);

  Allocator_t allocators[MAX_ALLOCATOR];
};

RT_OFFLOAD_VAR_GROUP_BEGIN
extern RT_VAR_ATTRS AllocatorRegistry allocatorRegistry;
RT_OFFLOAD_VAR_GROUP_END

} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_ALLOCATOR_H_
