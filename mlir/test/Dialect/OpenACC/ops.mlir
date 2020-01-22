// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

  acc.parallel {
    acc.loop {
      loop.for %arg3 = %c0 to %c10 step %c1 {
        loop.for %arg4 = %c0 to %c10 step %c1 {
          loop.for %arg5 = %c0 to %c10 step %c1 {
            %a = load %A[%arg3, %arg5] : memref<10x10xf32>
            %b = load %B[%arg5, %arg4] : memref<10x10xf32>
            %cij = load %C[%arg3, %arg4] : memref<10x10xf32>
            %p = mulf %a, %b : f32
            %co = addf %cij, %p : f32
            store %co, %C[%arg3, %arg4] : memref<10x10xf32>
          }
        }
      }
    } attributes { collapse = 3 }
  } attributes { async = 1 }

  return %C : memref<10x10xf32>
}

// CHECK-LABEL: func @compute(
//  CHECK-NEXT:   %{{.*}} = constant 0 : index
//  CHECK-NEXT:   %{{.*}} = constant 10 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   acc.parallel {
//  CHECK-NEXT:     acc.loop {
//  CHECK-NEXT:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:         loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:           loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//  CHECK-NEXT:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:           }
//  CHECK-NEXT:         }
//  CHECK-NEXT:       }
//  CHECK-NEXT: } attributes {collapse = 3 : i64}
