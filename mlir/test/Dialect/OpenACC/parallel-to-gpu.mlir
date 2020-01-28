// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index

  acc.parallel {
    %tmp = load %A[%c0,%c0] : memref<10x10xf32> 
    store %tmp, %B[%c0,%c0] : memref<10x10xf32>
  }

  return %C : memref<10x10xf32>
}

// CHECK-LABEL: func @compute(
//  CHECK-NEXT:   %{{.*}} = constant 0 : index
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : memref<10x10xf32>, index, memref<10x10xf32> {
//  CHECK-NEXT:      %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:      store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
//  CHECK-NEXT:      gpu.return
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return %{{.*}} : memref<10x10xf32>
//  CHECK-NEXT: }
