// RUN: mlir-opt -convert-openacc-to-llvm %s  -split-input-file | FileCheck %s

func @testenterdataop(%a: memref<10xf32>) -> () {
  acc.enter_data create(%a : memref<10xf32>)
  return
}

// CHECK: llvm.call @__tgt_target_data_begin_mapper(%{{.*}}, %{{.*}}) : (i64, i32) -> ()
// CHECK-NOT: acc.enter_data