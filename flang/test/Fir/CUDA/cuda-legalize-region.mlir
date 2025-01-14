// RUN: fir-opt --split-input-file --pass-pipeline="builtin.module(func.func(cuf-legalize-region))" %s | FileCheck %s

fir.global @_QMdevptrEmod_dev_arr {data_attr = #cuf.cuda<device>} target : !fir.array<4xf32> {
  %0 = fir.zero_bits !fir.array<4xf32>
  fir.has_value %0 : !fir.array<4xf32>
}
func.func @_QQmain() attributes {fir.bindc_name = "test"} {
  %cst = arith.constant 8.000000e+00 : f32
  %cst_0 = arith.constant 4.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = cuf.alloc !fir.array<4xf32> {bindc_name = "a_dev", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa_dev"} -> !fir.ref<!fir.array<4xf32>>
  %1 = fir.shape %c4 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFEa_dev"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %4 = fir.declare %3 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.address_of(@_QMdevptrEmod_dev_arr) : !fir.ref<!fir.array<4xf32>>
  %6 = fir.declare %5(%1) {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMdevptrEmod_dev_arr"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  %7 = fir.address_of(@_QQro.4xr4.0) : !fir.ref<!fir.array<4xf32>>
  %8 = fir.declare %7(%1) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.4xr4.0"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  cuf.data_transfer %8 to %2 {transfer_kind = #cuf.cuda_transfer<host_device>} : !fir.ref<!fir.array<4xf32>>, !fir.ref<!fir.array<4xf32>>
  cuf.kernel<<<*, *>>> (%arg0 : index) = (%c1 : index) to (%c4 : index)  step (%c1 : index) {
    %9 = fir.convert %arg0 : (index) -> i32
    fir.store %9 to %4 : !fir.ref<i32>
    %10 = fir.load %4 : !fir.ref<i32>
    %11 = fir.convert %10 : (i32) -> i64
    %12 = fir.array_coor %2(%1) %11 : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %13 = fir.load %12 : !fir.ref<f32>
    %14 = arith.addf %13, %cst_0 fastmath<contract> : f32
    %15 = fir.array_coor %6(%1) %11 : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    fir.store %14 to %15 : !fir.ref<f32>
    %16 = fir.load %4 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = fir.array_coor %2(%1) %17 : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
    %19 = fir.load %18 : !fir.ref<f32>
    %20 = arith.addf %19, %cst fastmath<contract> : f32
    fir.store %20 to %18 : !fir.ref<f32>
    "fir.end"() : () -> ()
  }
  cuf.free %2 : !fir.ref<!fir.array<4xf32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QQmain() attributes {fir.bindc_name = "test"} 
// CHECK: %[[DEV_ADDR:.*]] = cuf.device_address @_QMdevptrEmod_dev_arr -> <!fir.array<4xf32>>
// CHECK: cuf.kernel<<<*, *>>> (%arg0 : index) = (%c1 : index) to (%c4 : index)  step (%c1 : index) {
// CHECK: fir.array_coor %[[DEV_ADDR]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>

