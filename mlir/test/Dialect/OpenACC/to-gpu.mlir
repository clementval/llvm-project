// RUN: mlir-opt %s --convert-openacc-to-gpu | FileCheck %s

func @compute(%A: memref<10x10xf32>, %B: memref<10x10xf32>, %C: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

  acc.parallel {
    acc.loop {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
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
// CHECK-LABEL: gpu.module @compute_acc_parallel {
// CHECK-NEXT:    gpu.func @compute_acc_parallel(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>, %arg2: memref<10x10xf32>, %arg3: index, %arg4: index, %arg5: index) kernel {
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 0 : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 0 : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      %{{.*}} = constant 0 : index
// CHECK-NEXT:      %{{.*}} = constant 1 : index
// CHECK-NEXT:      %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      [[UPPERBOUND:%.*]] = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to [[UPPERBOUND]] step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = remi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = divi_signed %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:        %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:        %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:        %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:        store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x10xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
