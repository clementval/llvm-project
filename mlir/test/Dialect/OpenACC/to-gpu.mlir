// RUN: mlir-opt %s --convert-openacc-to-gpu | FileCheck %s

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
//  CHECK-NEXT:   %{{.*}} = subi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:   %{{.*}} = subi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   gpu.launch blocks
//  CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     %{{.*}} = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     loop.for
//  CHECK-NEXT:       %{{.*}} = load
//  CHECK-NEXT:       %{{.*}} = load
//  CHECK-NEXT:       %{{.*}} = load
//  CHECK-NEXT:       %{{.*}} = mulf
//  CHECK-NEXT:       %{{.*}} = addf
//  CHECK-NEXT:       store %{{.*}}
//  CHECK-NEXT:     }
//  CHECK-NEXT:     gpu.return
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %{{.*}}

