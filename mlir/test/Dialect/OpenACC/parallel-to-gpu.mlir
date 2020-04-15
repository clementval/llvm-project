// RUN: mlir-opt --convert-openacc-to-gpu %s | FileCheck %s

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index

  acc.parallel {
    %tmp = load %A[%c0,%c0] : memref<10x10xf32>
    store %tmp, %B[%c0,%c0] : memref<10x10xf32>
  }

  return %C : memref<10x10xf32>
}

// CHECK-LABEL: module attributes {gpu.container_module} {
// CHECK-NEXT:   func @compute(%{{.*}}: memref<10x10xf32>, %{{.*}}: memref<10x10xf32>, %{{.*}}: memref<10x10xf32>) -> memref<10x10xf32> {
// CHECK-NEXT:     %{{.*}} = constant 0 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     %{{.*}} = constant 1 : index
// CHECK-NEXT:     "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {kernel = @{{.*}}::@{{.*}}} : (index, index, index, index, index, index, memref<10x10xf32>, index, memref<10x10xf32>) -> ()
// CHECK-NEXT:     return %{{.*}} : memref<10x10xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   gpu.module @compute_acc_parallel {
// CHECK-NEXT:     gpu.func @compute_acc_parallel(%{{.*}}: memref<10x10xf32>, %{{.*}}: index, %{{.*}}: memref<10x10xf32>) kernel {
// CHECK-NEXT:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:       gpu.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
