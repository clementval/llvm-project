// RUN: fir-opt --split-input-file --cuf-convert %s | FileCheck %s

// Check that for a simple scalar constant the device address is retrieved
// for the data transfer.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  fir.global @_QMmod1Econst_int {data_attr = #cuf.cuda<constant>} : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QPsub30() {
    %4 = fir.address_of(@_QMmod1Econst_int) : !fir.ref<i32>
    %5 = fir.declare %4 {data_attr = #cuf.cuda<constant>, uniq_name = "_QMmod1Econst_int"} : (!fir.ref<i32>) -> (!fir.ref<i32>)
    %c4_i32 = arith.constant 4 : i32
    cuf.data_transfer %c4_i32 to %5 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<i32>
    return
  }
}

// CHECK-LABEL: func.func @_QPsub30()
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmod1Econst_int) : !fir.ref<i32>
// CHECK: %[[DECL:.*]] = fir.declare %[[ADDR]] {data_attr = #cuf.cuda<constant>, uniq_name = "_QMmod1Econst_int"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmod1Econst_int) : !fir.ref<i32>
// CHECK: %[[CONV_ADDR:.*]] = fir.convert %[[ADDR]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEV_ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[CONV_ADDR]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEV_ADDR_CONV:.*]] = fir.convert %[[DEV_ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<i32>
// CHECK: %[[DST:.*]] = fir.convert %[[DEV_ADDR_CONV]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFDataTransferPtrPtr(%[[DST]], %{{.*}}, %c4{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, i64, i32, !fir.ref<i8>, i32) -> ()

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  fir.global @_QMmod1Econst_int_array {data_attr = #cuf.cuda<constant>} : !fir.array<10xi32> {
    %0 = fir.zero_bits !fir.array<10xi32>
    fir.has_value %0 : !fir.array<10xi32>
  }
  func.func @_QPsub31() {
    %6 = fir.address_of(@_QMmod1Econst_int_array) : !fir.ref<!fir.array<10xi32>>
    %c10_0 = arith.constant 10 : index
    %7 = fir.shape %c10_0 : (index) -> !fir.shape<1>
    %8 = fir.declare %6(%7) {data_attr = #cuf.cuda<constant>, uniq_name = "_QMmod1Econst_int_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>)
    %c4_i32 = arith.constant 4 : i32
    cuf.data_transfer %c4_i32 to %8 {transfer_kind = #cuf.cuda_transfer<host_device>} : i32, !fir.ref<!fir.array<10xi32>>
    return
  }
}


// CHECK-LABEL: func.func @_QPsub31()
// CHECK: %[[TMP:.*]] = fir.alloca !fir.box<!fir.array<10xi32>>
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmod1Econst_int_array) : !fir.ref<!fir.array<10xi32>>
// CHECK: %{{.*}} = fir.declare %[[ADDR]](%{{.*}}) {data_attr = #cuf.cuda<constant>, uniq_name = "_QMmod1Econst_int_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmod1Econst_int_array) : !fir.ref<!fir.array<10xi32>>
// CHECK: %[[ADDR_CONV:.*]] = fir.convert %[[ADDR]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEV_ADDR:.*]] = fir.call @_FortranACUFGetDeviceAddress(%[[ADDR_CONV]], %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[DEV_ADDR_CONV:.*]] = fir.convert %[[DEV_ADDR]] : (!fir.llvm_ptr<i8>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: %[[EMBOX:.*]] = fir.embox %[[DEV_ADDR_CONV]](%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK: fir.store %[[EMBOX]] to %[[TMP]] : !fir.ref<!fir.box<!fir.array<10xi32>>>   
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[TMP]] : (!fir.ref<!fir.box<!fir.array<10xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFDataTransferCstDesc(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<!fir.box<none>>, i32, !fir.ref<i8>, i32) -> ()

// -----

// Check that access of constant on the host is not done through the device
// address.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  fir.global @_QMconstantsEc1(dense<[8.000000e-01, 2.000000e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<20xf32>) {data_attr = #cuf.cuda<constant>} : !fir.array<20xf32>

  fir.global internal @_QFEh_ratioemitt : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }

  func.func @_QQmain() attributes {fir.bindc_name = "MAIN"} {
    %true = arith.constant true
    %c22 = arith.constant 22 : index
    %c14_i32 = arith.constant 14 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c20 = arith.constant 20 : index
    %1 = fir.address_of(@_QFEh_ratioemitt) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEh_ratioemitt"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %3 = fir.address_of(@_QMconstantsEc1) : !fir.ref<!fir.array<20xf32>>
    %4 = fir.shape %c20 : (index) -> !fir.shape<1>
    %5 = fir.declare %3(%4) {data_attr = #cuf.cuda<constant>, uniq_name = "_QMconstantsEc1"} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<20xf32>>
    %6 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %7 = fir.do_loop %arg0 = %c1 to %c20 step %c1 unordered iter_args(%arg1 = %true) -> (i1) {
      %8 = fir.array_coor %5(%4) %arg0 : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
      %9:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
      %10 = arith.subi %9#0, %c1 : index
      %11 = arith.addi %arg0, %10 : index
      %12 = fir.box_addr %6 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
      %c0_0 = arith.constant 0 : index
      %13:3 = fir.box_dims %6, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
      %14 = fir.shape_shift %13#0, %13#1 : (index, index) -> !fir.shapeshift<1>
      %15 = fir.array_coor %12(%14) %11 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
      %16 = fir.load %8 : !fir.ref<f32>
      %17 = fir.load %15 : !fir.ref<f32>
      %18 = arith.cmpf oeq, %16, %17 fastmath<contract> : f32
      %19 = arith.andi %18, %arg1 : i1
      fir.result %19 : i1
    }
    return
  }
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: %[[CST:.*]] = fir.address_of(@_QMconstantsEc1) : !fir.ref<!fir.array<20xf32>>
// CHECK: %[[CST_DECL:.*]] = fir.declare %[[CST]](%{{.*}}) {data_attr = #cuf.cuda<constant>, uniq_name = "_QMconstantsEc1"} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<20xf32>>
// CHECK: fir.do_loop
// CHECK: fir.array_coor %[[CST_DECL:.*]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>

