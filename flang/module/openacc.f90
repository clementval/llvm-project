!===-- module/openacc.f90 --------------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module openacc

  ! 3.1 Runtime integer parameter openacc_version
  integer(kind=4), parameter :: openacc_version = 201911

  integer(kind=4), parameter :: acc_handle_kind = 4

  ! Type of accelerator device (Defined in standard OpenACC 3.0 (3.1))
  integer(kind=4), parameter :: acc_device_kind = 4
  integer(acc_device_kind), parameter :: acc_device_none = 0
  integer(acc_device_kind), parameter :: acc_device_default = 1
  integer(acc_device_kind), parameter :: acc_device_host = 2
  integer(acc_device_kind), parameter :: acc_device_not_host = 3
  integer(acc_device_kind), parameter :: acc_device_nvidia = 4
  integer(acc_device_kind), parameter :: acc_device_radeon = 5


  integer(kind=4), parameter :: acc_device_property = 4
  integer(acc_device_property), parameter :: acc_property_memory = 0
  integer(acc_device_property), parameter :: acc_property_free_memory = 1
  integer(acc_device_property), parameter :: &
          acc_property_shared_memory_support = 2
  integer(acc_device_property), parameter :: acc_property_name = 4
  integer(acc_device_property), parameter :: acc_property_vendor = 5
  integer(acc_device_property), parameter :: acc_property_driver = 6

  ! Signatures for acc_copyin
  interface acc_copyin
      module procedure &
              acc_copyin_i_1d_p1, acc_copyin_i_2d_p1, acc_copyin_i_3d_p1, &
              acc_copyin_i_4d_p1, &
              acc_copyin_r_1d_p1, acc_copyin_r_2d_p1, acc_copyin_r_3d_p1, &
              acc_copyin_r_4d_p1, &
              acc_copyin_l_1d_p1, acc_copyin_l_2d_p1, acc_copyin_l_3d_p1, &
              acc_copyin_l_4d_p1, &
              acc_copyin_c_1d_p1, acc_copyin_c_2d_p1, acc_copyin_c_3d_p1, &
              acc_copyin_c_4d_p1, &
              acc_copyin_i_l_p2, acc_copyin_r_l_p2, acc_copyin_l_l_p2, &
              acc_copyin_c_l_p2
  end interface

  ! Signatures for acc_copyin_async
  interface acc_copyin_async
      module procedure &
              acc_copyin_async_i_1d_p2, acc_copyin_async_i_2d_p2, &
              acc_copyin_async_i_3d_p2, acc_copyin_async_i_4d_p2, &
              acc_copyin_async_r_1d_p2, acc_copyin_async_r_2d_p2, &
              acc_copyin_async_r_3d_p2, acc_copyin_async_r_4d_p2, &
              acc_copyin_async_l_1d_p2, acc_copyin_async_l_2d_p2, &
              acc_copyin_async_l_3d_p2, acc_copyin_async_l_4d_p2, &
              acc_copyin_async_c_1d_p2, acc_copyin_async_c_2d_p2, &
              acc_copyin_async_c_3d_p2, acc_copyin_async_c_4d_p2, &
              acc_copyin_async_i_l_p3, acc_copyin_async_r_l_p3, &
              acc_copyin_async_l_l_p3, acc_copyin_async_c_l_p3
  end interface

  ! Signatures for acc_create
  interface acc_create
      module procedure &
              acc_create_i_1d_p1, acc_create_i_2d_p1, acc_create_i_3d_p1, &
              acc_create_i_4d_p1, &
              acc_create_r_1d_p1, acc_create_r_2d_p1, acc_create_r_3d_p1, &
              acc_create_r_4d_p1, &
              acc_create_l_1d_p1, acc_create_l_2d_p1, acc_create_l_3d_p1, &
              acc_create_l_4d_p1, &
              acc_create_c_1d_p1, acc_create_c_2d_p1, acc_create_c_3d_p1, &
              acc_create_c_4d_p1, &
              acc_create_i_l_p2, acc_create_r_l_p2, acc_create_l_l_p2, &
              acc_create_c_l_p2
  end interface

  ! Signatures for acc_create_async
  interface acc_create_async
      module procedure &
              acc_create_async_i_1d_p2, acc_create_async_i_2d_p2, &
              acc_create_async_i_3d_p2, acc_create_async_i_4d_p2, &
              acc_create_async_r_1d_p2, acc_create_async_r_2d_p2, &
              acc_create_async_r_3d_p2, acc_create_async_r_4d_p2, &
              acc_create_async_l_1d_p2, acc_create_async_l_2d_p2, &
              acc_create_async_l_3d_p2, acc_create_async_l_4d_p2, &
              acc_create_async_c_1d_p2, acc_create_async_c_2d_p2, &
              acc_create_async_c_3d_p2, acc_create_async_c_4d_p2, &
              acc_create_async_i_l_p3, acc_create_async_r_l_p3, &
              acc_create_async_l_l_p3, acc_create_async_c_l_p3
  end interface

  ! Signatures for acc_copyout
  interface acc_copyout
      module procedure &
              acc_copyout_i_1d_p1, acc_copyout_i_2d_p1, acc_copyout_i_3d_p1, &
              acc_copyout_i_4d_p1, &
              acc_copyout_r_1d_p1, acc_copyout_r_2d_p1, acc_copyout_r_3d_p1, &
              acc_copyout_r_4d_p1, &
              acc_copyout_l_1d_p1, acc_copyout_l_2d_p1, acc_copyout_l_3d_p1, &
              acc_copyout_l_4d_p1, &
              acc_copyout_c_1d_p1, acc_copyout_c_2d_p1, acc_copyout_c_3d_p1, &
              acc_copyout_c_4d_p1, &
              acc_copyout_i_l_p2, acc_copyout_r_l_p2, acc_copyout_l_l_p2, &
              acc_copyout_c_l_p2
  end interface

  ! Signatures for acc_copyout_async
  interface acc_copyout_async
      module procedure &
              acc_copyout_async_i_1d_p2, acc_copyout_async_i_2d_p2, &
              acc_copyout_async_i_3d_p2, acc_copyout_async_i_4d_p2, &
              acc_copyout_async_r_1d_p2, acc_copyout_async_r_2d_p2, &
              acc_copyout_async_r_3d_p2, acc_copyout_async_r_4d_p2, &
              acc_copyout_async_l_1d_p2, acc_copyout_async_l_2d_p2, &
              acc_copyout_async_l_3d_p2, acc_copyout_async_l_4d_p2, &
              acc_copyout_async_c_1d_p2, acc_copyout_async_c_2d_p2, &
              acc_copyout_async_c_3d_p2, acc_copyout_async_c_4d_p2, &
              acc_copyout_async_i_l_p3, acc_copyout_async_r_l_p3, &
              acc_copyout_async_l_l_p3, acc_copyout_async_c_l_p3
  end interface

  ! Signatures for acc_copyout_finalize
  interface acc_copyout_finalize
      module procedure &
              acc_copyout_finalize_i_1d_p1, acc_copyout_finalize_i_2d_p1, &
              acc_copyout_finalize_i_3d_p1, acc_copyout_finalize_i_4d_p1, &
              acc_copyout_finalize_r_1d_p1, acc_copyout_finalize_r_2d_p1, &
              acc_copyout_finalize_r_3d_p1, acc_copyout_finalize_r_4d_p1, &
              acc_copyout_finalize_l_1d_p1, acc_copyout_finalize_l_2d_p1, &
              acc_copyout_finalize_l_3d_p1, acc_copyout_finalize_l_4d_p1, &
              acc_copyout_finalize_c_1d_p1, acc_copyout_finalize_c_2d_p1, &
              acc_copyout_finalize_c_3d_p1, acc_copyout_finalize_c_4d_p1, &
              acc_copyout_finalize_i_l_p2, acc_copyout_finalize_r_l_p2, &
              acc_copyout_finalize_l_l_p2, acc_copyout_finalize_c_l_p2
  end interface

  ! Signatures for acc_copyout_finalize_async
  interface acc_copyout_finalize_async
      module procedure &
              acc_copyout_finalize_async_i_1d_p2, acc_copyout_finalize_async_i_2d_p2, &
              acc_copyout_finalize_async_i_3d_p2, acc_copyout_finalize_async_i_4d_p2, &
              acc_copyout_finalize_async_r_1d_p2, acc_copyout_finalize_async_r_2d_p2, &
              acc_copyout_finalize_async_r_3d_p2, acc_copyout_finalize_async_r_4d_p2, &
              acc_copyout_finalize_async_l_1d_p2, acc_copyout_finalize_async_l_2d_p2, &
              acc_copyout_finalize_async_l_3d_p2, acc_copyout_finalize_async_l_4d_p2, &
              acc_copyout_finalize_async_c_1d_p2, acc_copyout_finalize_async_c_2d_p2, &
              acc_copyout_finalize_async_c_3d_p2, acc_copyout_finalize_async_c_4d_p2, &
              acc_copyout_finalize_async_i_l_p3, acc_copyout_finalize_async_r_l_p3, &
              acc_copyout_finalize_async_l_l_p3, acc_copyout_finalize_async_c_l_p3
  end interface

  ! Signatures for acc_delete
  interface acc_delete
      module procedure &
              acc_delete_i_1d_p1, acc_delete_i_2d_p1, acc_delete_i_3d_p1, &
              acc_delete_i_4d_p1, &
              acc_delete_r_1d_p1, acc_delete_r_2d_p1, acc_delete_r_3d_p1, &
              acc_delete_r_4d_p1, &
              acc_delete_l_1d_p1, acc_delete_l_2d_p1, acc_delete_l_3d_p1, &
              acc_delete_l_4d_p1, &
              acc_delete_c_1d_p1, acc_delete_c_2d_p1, acc_delete_c_3d_p1, &
              acc_delete_c_4d_p1, &
              acc_delete_i_l_p2, acc_delete_r_l_p2, acc_delete_l_l_p2, &
              acc_delete_c_l_p2
  end interface

  ! Signatures for acc_delete_async
  interface acc_delete_async
      module procedure &
              acc_delete_async_i_1d_p2, acc_delete_async_i_2d_p2, &
              acc_delete_async_i_3d_p2, acc_delete_async_i_4d_p2, &
              acc_delete_async_r_1d_p2, acc_delete_async_r_2d_p2, &
              acc_delete_async_r_3d_p2, acc_delete_async_r_4d_p2, &
              acc_delete_async_l_1d_p2, acc_delete_async_l_2d_p2, &
              acc_delete_async_l_3d_p2, acc_delete_async_l_4d_p2, &
              acc_delete_async_c_1d_p2, acc_delete_async_c_2d_p2, &
              acc_delete_async_c_3d_p2, acc_delete_async_c_4d_p2, &
              acc_delete_async_i_l_p3, acc_delete_async_r_l_p3, &
              acc_delete_async_l_l_p3, acc_delete_async_c_l_p3
  end interface

  ! Signatures for acc_delete_finalize
  interface acc_delete_finalize
      module procedure &
              acc_delete_finalize_i_1d_p1, acc_delete_finalize_i_2d_p1, &
              acc_delete_finalize_i_3d_p1, acc_delete_finalize_i_4d_p1, &
              acc_delete_finalize_r_1d_p1, acc_delete_finalize_r_2d_p1, &
              acc_delete_finalize_r_3d_p1, acc_delete_finalize_r_4d_p1, &
              acc_delete_finalize_l_1d_p1, acc_delete_finalize_l_2d_p1, &
              acc_delete_finalize_l_3d_p1, acc_delete_finalize_l_4d_p1, &
              acc_delete_finalize_c_1d_p1, acc_delete_finalize_c_2d_p1, &
              acc_delete_finalize_c_3d_p1, acc_delete_finalize_c_4d_p1, &
              acc_delete_finalize_i_l_p2, acc_delete_finalize_r_l_p2, &
              acc_delete_finalize_l_l_p2, acc_delete_finalize_c_l_p2
  end interface

  ! Signatures for acc_delete_finalize_async
  interface acc_delete_finalize_async
      module procedure &
              acc_delete_finalize_async_i_1d_p2, acc_delete_finalize_async_i_2d_p2, &
              acc_delete_finalize_async_i_3d_p2, acc_delete_finalize_async_i_4d_p2, &
              acc_delete_finalize_async_r_1d_p2, acc_delete_finalize_async_r_2d_p2, &
              acc_delete_finalize_async_r_3d_p2, acc_delete_finalize_async_r_4d_p2, &
              acc_delete_finalize_async_l_1d_p2, acc_delete_finalize_async_l_2d_p2, &
              acc_delete_finalize_async_l_3d_p2, acc_delete_finalize_async_l_4d_p2, &
              acc_delete_finalize_async_c_1d_p2, acc_delete_finalize_async_c_2d_p2, &
              acc_delete_finalize_async_c_3d_p2, acc_delete_finalize_async_c_4d_p2, &
              acc_delete_finalize_async_i_l_p3, acc_delete_finalize_async_r_l_p3, &
              acc_delete_finalize_async_l_l_p3, acc_delete_finalize_async_c_l_p3
  end interface

  ! Signatures for acc_update_device
  interface acc_update_device
      module procedure &
              acc_update_device_i_1d_p1, acc_update_device_i_2d_p1, &
              acc_update_device_i_3d_p1, acc_update_device_i_4d_p1, &
              acc_update_device_r_1d_p1, acc_update_device_r_2d_p1, &
              acc_update_device_r_3d_p1, acc_update_device_r_4d_p1, &
              acc_update_device_l_1d_p1, acc_update_device_l_2d_p1, &
              acc_update_device_l_3d_p1, acc_update_device_l_4d_p1, &
              acc_update_device_c_1d_p1, acc_update_device_c_2d_p1, &
              acc_update_device_c_3d_p1, acc_update_device_c_4d_p1, &
              acc_update_device_i_l_p2, acc_update_device_r_l_p2, &
              acc_update_device_l_l_p2, acc_update_device_c_l_p2
  end interface

  ! Signatures for acc_update_device_async
  interface acc_update_device_async
      module procedure &
              acc_update_device_async_i_1d_p2, acc_update_device_async_i_2d_p2, &
              acc_update_device_async_i_3d_p2, acc_update_device_async_i_4d_p2, &
              acc_update_device_async_r_1d_p2, acc_update_device_async_r_2d_p2, &
              acc_update_device_async_r_3d_p2, acc_update_device_async_r_4d_p2, &
              acc_update_device_async_l_1d_p2, acc_update_device_async_l_2d_p2, &
              acc_update_device_async_l_3d_p2, acc_update_device_async_l_4d_p2, &
              acc_update_device_async_c_1d_p2, acc_update_device_async_c_2d_p2, &
              acc_update_device_async_c_3d_p2, acc_update_device_async_c_4d_p2, &
              acc_update_device_async_i_l_p3, acc_update_device_async_r_l_p3, &
              acc_update_device_async_l_l_p3, acc_update_device_async_c_l_p3
  end interface

  ! Signatures for acc_update_self
  interface acc_update_self
      module procedure &
              acc_update_self_i_1d_p1, acc_update_self_i_2d_p1, &
              acc_update_self_i_3d_p1, acc_update_self_i_4d_p1, &
              acc_update_self_r_1d_p1, acc_update_self_r_2d_p1, &
              acc_update_self_r_3d_p1, acc_update_self_r_4d_p1, &
              acc_update_self_l_1d_p1, acc_update_self_l_2d_p1, &
              acc_update_self_l_3d_p1, acc_update_self_l_4d_p1, &
              acc_update_self_c_1d_p1, acc_update_self_c_2d_p1, &
              acc_update_self_c_3d_p1, acc_update_self_c_4d_p1, &
              acc_update_self_i_l_p2, acc_update_self_r_l_p2, &
              acc_update_self_l_l_p2, acc_update_self_c_l_p2
  end interface

  ! Signatures for acc_update_self_async
  interface acc_update_self_async
      module procedure &
              acc_update_self_async_i_1d_p2, acc_update_self_async_i_2d_p2, &
              acc_update_self_async_i_3d_p2, acc_update_self_async_i_4d_p2, &
              acc_update_self_async_r_1d_p2, acc_update_self_async_r_2d_p2, &
              acc_update_self_async_r_3d_p2, acc_update_self_async_r_4d_p2, &
              acc_update_self_async_l_1d_p2, acc_update_self_async_l_2d_p2, &
              acc_update_self_async_l_3d_p2, acc_update_self_async_l_4d_p2, &
              acc_update_self_async_c_1d_p2, acc_update_self_async_c_2d_p2, &
              acc_update_self_async_c_3d_p2, acc_update_self_async_c_4d_p2, &
              acc_update_self_async_i_l_p3, acc_update_self_async_r_l_p3, &
              acc_update_self_async_l_l_p3, acc_update_self_async_c_l_p3
  end interface

  ! Signatures for acc_is_present
  interface acc_is_present
      module procedure &
              acc_is_present_i_1d_p1, acc_is_present_i_2d_p1, acc_is_present_i_3d_p1, &
              acc_is_present_i_4d_p1, &
              acc_is_present_r_1d_p1, acc_is_present_r_2d_p1, acc_is_present_r_3d_p1, &
              acc_is_present_r_4d_p1, &
              acc_is_present_l_1d_p1, acc_is_present_l_2d_p1, acc_is_present_l_3d_p1, &
              acc_is_present_l_4d_p1, &
              acc_is_present_c_1d_p1, acc_is_present_c_2d_p1, acc_is_present_c_3d_p1, &
              acc_is_present_c_4d_p1, &
              acc_is_present_i_l_p2, acc_is_present_r_l_p2, acc_is_present_l_l_p2, &
              acc_is_present_c_l_p2
  end interface

contains
  ! 3.2.1
  integer function acc_get_num_devices( devicetype )
      integer(acc_device_kind) ::  devicetype
  end function acc_get_num_devices

  ! 3.2.2
  subroutine acc_set_device_type( devicetype )
      integer(acc_device_kind) ::  devicetype
  end subroutine acc_set_device_type

  ! 3.2.3
  function acc_get_device_type()
      integer(acc_device_kind) ::  acc_get_device_type
  end function acc_get_device_type

  ! 3.2.4
  subroutine acc_set_device_num( devicenum, devicetype )
      integer ::  devicenum
      integer(acc_device_kind) ::  devicetype
  end subroutine acc_set_device_num

  ! 3.2.5
  integer function acc_get_device_num( devicetype )
      integer(acc_device_kind) ::  devicetype
  end function acc_get_device_num

  ! 3.2.6
  integer function acc_get_property( devicenum, devicetype, property )
      integer, value ::  devicenum
      integer(acc_device_kind), value ::  devicetype
      integer(acc_device_property), value ::  property
      integer(acc_device_property) ::  acc_get_propert
  end function acc_get_property

  subroutine acc_get_property_string( devicenum, devicetype, property, string )
      integer, value ::  devicenum
      integer(acc_device_kind), value ::  devicetype
      integer(acc_device_property), value ::  property
      character*(*) ::  string
  end subroutine acc_get_property_string

  ! 3.2.7
  subroutine acc_init( devicetype )
      integer(acc_device_kind) ::  devicetype
  end subroutine acc_init

  ! 3.2.8
  subroutine acc_shutdown( devicetype )
      integer(acc_device_kind) ::  devicetype
  end subroutine acc_shutdown

  ! 3.2.9
  logical function acc_async_test( arg )
      integer(acc_handle_kind) ::  arg
  end function acc_async_test

  ! 3.2.10
  logical function acc_async_test_device( arg, device )
      integer(acc_handle_kind) :: arg
      integer :: device
  end function acc_async_test_device

  ! 3.2.11
  logical function acc_async_test_all()
  end function acc_async_test_all

  ! 3.2.11
  logical function acc_async_test_all_device( device )
      integer :: device
  end function acc_async_test_all_device

  !  3.2.13
  subroutine acc_wait( arg )
      integer(acc_handle_kind) :: arg
  end subroutine acc_wait

  !  3.2.14
  subroutine acc_wait_device( arg, device )
      integer(acc_handle_kind) :: arg
      integer :: device
  end subroutine acc_wait_device

  ! 3.2.15
  subroutine acc_wait_async( arg, async )
      integer(acc_handle_kind) :: arg, async
  end subroutine acc_wait_async

  ! 3.2.16
  subroutine acc_wait_device_async( arg, async, device )
      integer(acc_handle_kind) :: arg, async
      integer :: device
  end subroutine acc_wait_device_async

  ! 3.2.17
  subroutine acc_wait_all()
  end subroutine acc_wait_all

  ! 3.2.18
  subroutine acc_wait_all_device( device )
      integer :: device
  end subroutine acc_wait_all_device

  ! 3.2.19
  subroutine acc_wait_all_async( async )
      integer(acc_handle_kind) :: async
  end subroutine acc_wait_all_async

  ! 3.2.20
  subroutine acc_wait_all_device_async( async, device )
      integer(acc_handle_kind) :: async
      integer :: device
  end subroutine acc_wait_all_device_async

  ! 3.2.21
  function acc_get_default_async( )
      integer(acc_handle_kind) :: acc_get_default_async
  end function acc_get_default_async

  ! 3.2.22
  subroutine acc_set_default_async( async )
      integer(acc_handle_kind) :: async
  end subroutine acc_set_default_async

  ! 3.2.23
  logical function acc_on_device( devicetype )
      integer(acc_device_kind) ::  devicetype
  end function acc_on_device

  ! Signatures for acc_copyin
  subroutine acc_copyin_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_copyin_i_1d_p1

  subroutine acc_copyin_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_copyin_i_2d_p1

  subroutine acc_copyin_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_copyin_i_3d_p1

  subroutine acc_copyin_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_copyin_i_4d_p1

  subroutine acc_copyin_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_copyin_r_1d_p1

  subroutine acc_copyin_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_copyin_r_2d_p1

  subroutine acc_copyin_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_copyin_r_3d_p1

  subroutine acc_copyin_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_copyin_r_4d_p1

  subroutine acc_copyin_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_copyin_l_1d_p1

  subroutine acc_copyin_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_copyin_l_2d_p1

  subroutine acc_copyin_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_copyin_l_3d_p1

  subroutine acc_copyin_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_copyin_l_4d_p1

  subroutine acc_copyin_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_copyin_c_1d_p1

  subroutine acc_copyin_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_copyin_c_2d_p1

  subroutine acc_copyin_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_copyin_c_3d_p1

  subroutine acc_copyin_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_copyin_c_4d_p1

  subroutine acc_copyin_i_l_p2( a , len )
      integer :: a
      integer :: len
  end subroutine acc_copyin_i_l_p2

  subroutine acc_copyin_r_l_p2( a , len )
      real(8) :: a
      integer :: len
  end subroutine acc_copyin_r_l_p2

  subroutine acc_copyin_l_l_p2( a , len )
      logical :: a
      integer :: len
  end subroutine acc_copyin_l_l_p2

  subroutine acc_copyin_c_l_p2( a , len )
      character :: a
      integer :: len
  end subroutine acc_copyin_c_l_p2

  ! Signatures for acc_copyin_async
  subroutine acc_copyin_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_i_1d_p2

  subroutine acc_copyin_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_i_2d_p2

  subroutine acc_copyin_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_i_3d_p2

  subroutine acc_copyin_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_i_4d_p2

  subroutine acc_copyin_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_r_1d_p2

  subroutine acc_copyin_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_r_2d_p2

  subroutine acc_copyin_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_r_3d_p2

  subroutine acc_copyin_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_r_4d_p2

  subroutine acc_copyin_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_l_1d_p2

  subroutine acc_copyin_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_l_2d_p2

  subroutine acc_copyin_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_l_3d_p2

  subroutine acc_copyin_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_l_4d_p2

  subroutine acc_copyin_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_c_1d_p2

  subroutine acc_copyin_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_c_2d_p2

  subroutine acc_copyin_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_c_3d_p2

  subroutine acc_copyin_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_c_4d_p2

  subroutine acc_copyin_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_i_l_p3

  subroutine acc_copyin_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_r_l_p3

  subroutine acc_copyin_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_l_l_p3

  subroutine acc_copyin_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyin_async_c_l_p3

  ! Signatures for acc_create
  subroutine acc_create_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_create_i_1d_p1

  subroutine acc_create_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_create_i_2d_p1

  subroutine acc_create_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_create_i_3d_p1

  subroutine acc_create_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_create_i_4d_p1

  subroutine acc_create_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_create_r_1d_p1

  subroutine acc_create_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_create_r_2d_p1

  subroutine acc_create_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_create_r_3d_p1

  subroutine acc_create_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_create_r_4d_p1

  subroutine acc_create_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_create_l_1d_p1

  subroutine acc_create_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_create_l_2d_p1

  subroutine acc_create_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_create_l_3d_p1

  subroutine acc_create_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_create_l_4d_p1

  subroutine acc_create_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_create_c_1d_p1

  subroutine acc_create_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_create_c_2d_p1

  subroutine acc_create_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_create_c_3d_p1

  subroutine acc_create_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_create_c_4d_p1

  subroutine acc_create_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_create_i_l_p2

  subroutine acc_create_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_create_r_l_p2

  subroutine acc_create_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_create_l_l_p2

  subroutine acc_create_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_create_c_l_p2

  ! Signatures for acc_create_async
  subroutine acc_create_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_i_1d_p2

  subroutine acc_create_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_i_2d_p2

  subroutine acc_create_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_i_3d_p2

  subroutine acc_create_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_i_4d_p2

  subroutine acc_create_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_r_1d_p2

  subroutine acc_create_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_r_2d_p2

  subroutine acc_create_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_r_3d_p2

  subroutine acc_create_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_r_4d_p2

  subroutine acc_create_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_l_1d_p2

  subroutine acc_create_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_l_2d_p2

  subroutine acc_create_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_l_3d_p2

  subroutine acc_create_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_l_4d_p2

  subroutine acc_create_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_c_1d_p2

  subroutine acc_create_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_c_2d_p2

  subroutine acc_create_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_c_3d_p2

  subroutine acc_create_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_c_4d_p2

  subroutine acc_create_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_i_l_p3

  subroutine acc_create_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_r_l_p3

  subroutine acc_create_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_l_l_p3

  subroutine acc_create_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_create_async_c_l_p3

  ! Signatures for acc_copyout
  subroutine acc_copyout_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_copyout_i_1d_p1

  subroutine acc_copyout_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_copyout_i_2d_p1

  subroutine acc_copyout_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_copyout_i_3d_p1

  subroutine acc_copyout_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_i_4d_p1

  subroutine acc_copyout_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_copyout_r_1d_p1

  subroutine acc_copyout_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_copyout_r_2d_p1

  subroutine acc_copyout_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_copyout_r_3d_p1

  subroutine acc_copyout_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_copyout_r_4d_p1

  subroutine acc_copyout_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_copyout_l_1d_p1

  subroutine acc_copyout_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_copyout_l_2d_p1

  subroutine acc_copyout_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_copyout_l_3d_p1

  subroutine acc_copyout_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_l_4d_p1

  subroutine acc_copyout_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_copyout_c_1d_p1

  subroutine acc_copyout_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_copyout_c_2d_p1

  subroutine acc_copyout_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_copyout_c_3d_p1

  subroutine acc_copyout_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_c_4d_p1

  subroutine acc_copyout_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_copyout_i_l_p2

  subroutine acc_copyout_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_copyout_r_l_p2

  subroutine acc_copyout_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_copyout_l_l_p2

  subroutine acc_copyout_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_copyout_c_l_p2

  ! Signatures for acc_copyout_async
  subroutine acc_copyout_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_i_1d_p2

  subroutine acc_copyout_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_i_2d_p2

  subroutine acc_copyout_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_i_3d_p2

  subroutine acc_copyout_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_i_4d_p2

  subroutine acc_copyout_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_r_1d_p2

  subroutine acc_copyout_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_r_2d_p2

  subroutine acc_copyout_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_r_3d_p2

  subroutine acc_copyout_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_r_4d_p2

  subroutine acc_copyout_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_l_1d_p2

  subroutine acc_copyout_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_l_2d_p2

  subroutine acc_copyout_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_l_3d_p2

  subroutine acc_copyout_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_l_4d_p2

  subroutine acc_copyout_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_c_1d_p2

  subroutine acc_copyout_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_c_2d_p2

  subroutine acc_copyout_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_c_3d_p2

  subroutine acc_copyout_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_c_4d_p2

  subroutine acc_copyout_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_i_l_p3

  subroutine acc_copyout_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_r_l_p3

  subroutine acc_copyout_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_l_l_p3

  subroutine acc_copyout_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_async_c_l_p3

  ! Signatures for acc_copyout_finalize
  subroutine acc_copyout_finalize_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_copyout_finalize_i_1d_p1

  subroutine acc_copyout_finalize_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_copyout_finalize_i_2d_p1

  subroutine acc_copyout_finalize_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_copyout_finalize_i_3d_p1

  subroutine acc_copyout_finalize_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_finalize_i_4d_p1

  subroutine acc_copyout_finalize_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_copyout_finalize_r_1d_p1

  subroutine acc_copyout_finalize_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_copyout_finalize_r_2d_p1

  subroutine acc_copyout_finalize_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_copyout_finalize_r_3d_p1

  subroutine acc_copyout_finalize_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_copyout_finalize_r_4d_p1

  subroutine acc_copyout_finalize_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_copyout_finalize_l_1d_p1

  subroutine acc_copyout_finalize_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_copyout_finalize_l_2d_p1

  subroutine acc_copyout_finalize_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_copyout_finalize_l_3d_p1

  subroutine acc_copyout_finalize_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_finalize_l_4d_p1

  subroutine acc_copyout_finalize_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_copyout_finalize_c_1d_p1

  subroutine acc_copyout_finalize_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_copyout_finalize_c_2d_p1

  subroutine acc_copyout_finalize_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_copyout_finalize_c_3d_p1

  subroutine acc_copyout_finalize_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_copyout_finalize_c_4d_p1

  subroutine acc_copyout_finalize_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_copyout_finalize_i_l_p2

  subroutine acc_copyout_finalize_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_copyout_finalize_r_l_p2

  subroutine acc_copyout_finalize_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_copyout_finalize_l_l_p2

  subroutine acc_copyout_finalize_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_copyout_finalize_c_l_p2

  ! Signatures for acc_copyout_finalize_async
  subroutine acc_copyout_finalize_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_i_1d_p2

  subroutine acc_copyout_finalize_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_i_2d_p2

  subroutine acc_copyout_finalize_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_i_3d_p2

  subroutine acc_copyout_finalize_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_i_4d_p2

  subroutine acc_copyout_finalize_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_r_1d_p2

  subroutine acc_copyout_finalize_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_r_2d_p2

  subroutine acc_copyout_finalize_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_r_3d_p2

  subroutine acc_copyout_finalize_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_r_4d_p2

  subroutine acc_copyout_finalize_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_l_1d_p2

  subroutine acc_copyout_finalize_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_l_2d_p2

  subroutine acc_copyout_finalize_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_l_3d_p2

  subroutine acc_copyout_finalize_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_l_4d_p2

  subroutine acc_copyout_finalize_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_c_1d_p2

  subroutine acc_copyout_finalize_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_c_2d_p2

  subroutine acc_copyout_finalize_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_c_3d_p2

  subroutine acc_copyout_finalize_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_c_4d_p2

  subroutine acc_copyout_finalize_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_i_l_p3

  subroutine acc_copyout_finalize_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_r_l_p3

  subroutine acc_copyout_finalize_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_l_l_p3

  subroutine acc_copyout_finalize_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_copyout_finalize_async_c_l_p3

  ! Signatures for acc_delete
  subroutine acc_delete_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_delete_i_1d_p1

  subroutine acc_delete_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_delete_i_2d_p1

  subroutine acc_delete_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_delete_i_3d_p1

  subroutine acc_delete_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_delete_i_4d_p1

  subroutine acc_delete_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_delete_r_1d_p1

  subroutine acc_delete_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_delete_r_2d_p1

  subroutine acc_delete_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_delete_r_3d_p1

  subroutine acc_delete_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_delete_r_4d_p1

  subroutine acc_delete_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_delete_l_1d_p1

  subroutine acc_delete_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_delete_l_2d_p1

  subroutine acc_delete_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_delete_l_3d_p1

  subroutine acc_delete_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_delete_l_4d_p1

  subroutine acc_delete_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_delete_c_1d_p1

  subroutine acc_delete_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_delete_c_2d_p1

  subroutine acc_delete_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_delete_c_3d_p1

  subroutine acc_delete_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_delete_c_4d_p1

  subroutine acc_delete_i_l_p2( a, len )
      integer :: a
      integer ::  len
  end subroutine acc_delete_i_l_p2

  subroutine acc_delete_r_l_p2( a, len )
      real(8) :: a
      integer ::  len
  end subroutine acc_delete_r_l_p2

  subroutine acc_delete_l_l_p2( a, len )
      logical :: a
      integer ::  len
  end subroutine acc_delete_l_l_p2

  subroutine acc_delete_c_l_p2( a, len )
      character :: a
      integer ::  len
  end subroutine acc_delete_c_l_p2

  ! Signatures for acc_delete_async
  subroutine acc_delete_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_i_1d_p2

  subroutine acc_delete_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_i_2d_p2

  subroutine acc_delete_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_i_3d_p2

  subroutine acc_delete_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_i_4d_p2

  subroutine acc_delete_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_r_1d_p2

  subroutine acc_delete_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_r_2d_p2

  subroutine acc_delete_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_r_3d_p2

  subroutine acc_delete_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_r_4d_p2

  subroutine acc_delete_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_l_1d_p2

  subroutine acc_delete_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_l_2d_p2

  subroutine acc_delete_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_l_3d_p2

  subroutine acc_delete_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_l_4d_p2

  subroutine acc_delete_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_c_1d_p2

  subroutine acc_delete_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_c_2d_p2

  subroutine acc_delete_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_c_3d_p2

  subroutine acc_delete_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_c_4d_p2

  subroutine acc_delete_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_i_l_p3

  subroutine acc_delete_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_r_l_p3

  subroutine acc_delete_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_l_l_p3

  subroutine acc_delete_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_async_c_l_p3

  ! Signatures for acc_delete_finalize
  subroutine acc_delete_finalize_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_delete_finalize_i_1d_p1

  subroutine acc_delete_finalize_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_delete_finalize_i_2d_p1

  subroutine acc_delete_finalize_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_delete_finalize_i_3d_p1

  subroutine acc_delete_finalize_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_delete_finalize_i_4d_p1

  subroutine acc_delete_finalize_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_delete_finalize_r_1d_p1

  subroutine acc_delete_finalize_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_delete_finalize_r_2d_p1

  subroutine acc_delete_finalize_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_delete_finalize_r_3d_p1

  subroutine acc_delete_finalize_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_delete_finalize_r_4d_p1

  subroutine acc_delete_finalize_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_delete_finalize_l_1d_p1

  subroutine acc_delete_finalize_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_delete_finalize_l_2d_p1

  subroutine acc_delete_finalize_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_delete_finalize_l_3d_p1

  subroutine acc_delete_finalize_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_delete_finalize_l_4d_p1

  subroutine acc_delete_finalize_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_delete_finalize_c_1d_p1

  subroutine acc_delete_finalize_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_delete_finalize_c_2d_p1

  subroutine acc_delete_finalize_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_delete_finalize_c_3d_p1

  subroutine acc_delete_finalize_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_delete_finalize_c_4d_p1

  subroutine acc_delete_finalize_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_delete_finalize_i_l_p2

  subroutine acc_delete_finalize_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_delete_finalize_r_l_p2

  subroutine acc_delete_finalize_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_delete_finalize_l_l_p2

  subroutine acc_delete_finalize_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_delete_finalize_c_l_p2

  ! Signatures for acc_delete_finalize_async
  subroutine acc_delete_finalize_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_i_1d_p2

  subroutine acc_delete_finalize_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_i_2d_p2

  subroutine acc_delete_finalize_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_i_3d_p2

  subroutine acc_delete_finalize_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_i_4d_p2

  subroutine acc_delete_finalize_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_r_1d_p2

  subroutine acc_delete_finalize_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_r_2d_p2

  subroutine acc_delete_finalize_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_r_3d_p2

  subroutine acc_delete_finalize_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_r_4d_p2

  subroutine acc_delete_finalize_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_l_1d_p2

  subroutine acc_delete_finalize_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_l_2d_p2

  subroutine acc_delete_finalize_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_l_3d_p2

  subroutine acc_delete_finalize_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_l_4d_p2

  subroutine acc_delete_finalize_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_c_1d_p2

  subroutine acc_delete_finalize_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_c_2d_p2

  subroutine acc_delete_finalize_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_c_3d_p2

  subroutine acc_delete_finalize_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_c_4d_p2

  subroutine acc_delete_finalize_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_i_l_p3

  subroutine acc_delete_finalize_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_r_l_p3

  subroutine acc_delete_finalize_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_l_l_p3

  subroutine acc_delete_finalize_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_delete_finalize_async_c_l_p3

  ! Signatures for acc_update_device
  subroutine acc_update_device_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_update_device_i_1d_p1

  subroutine acc_update_device_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_update_device_i_2d_p1

  subroutine acc_update_device_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_update_device_i_3d_p1

  subroutine acc_update_device_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_update_device_i_4d_p1

  subroutine acc_update_device_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_update_device_r_1d_p1

  subroutine acc_update_device_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_update_device_r_2d_p1

  subroutine acc_update_device_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_update_device_r_3d_p1

  subroutine acc_update_device_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_update_device_r_4d_p1

  subroutine acc_update_device_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_update_device_l_1d_p1

  subroutine acc_update_device_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_update_device_l_2d_p1

  subroutine acc_update_device_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_update_device_l_3d_p1

  subroutine acc_update_device_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_update_device_l_4d_p1

  subroutine acc_update_device_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_update_device_c_1d_p1

  subroutine acc_update_device_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_update_device_c_2d_p1

  subroutine acc_update_device_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_update_device_c_3d_p1

  subroutine acc_update_device_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_update_device_c_4d_p1

  subroutine acc_update_device_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_update_device_i_l_p2

  subroutine acc_update_device_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_update_device_r_l_p2

  subroutine acc_update_device_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_update_device_l_l_p2

  subroutine acc_update_device_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_update_device_c_l_p2

  ! Signatures for acc_update_device_async
  subroutine acc_update_device_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_i_1d_p2

  subroutine acc_update_device_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_i_2d_p2

  subroutine acc_update_device_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_i_3d_p2

  subroutine acc_update_device_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_i_4d_p2

  subroutine acc_update_device_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_r_1d_p2

  subroutine acc_update_device_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_r_2d_p2

  subroutine acc_update_device_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_r_3d_p2

  subroutine acc_update_device_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_r_4d_p2

  subroutine acc_update_device_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_l_1d_p2

  subroutine acc_update_device_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_l_2d_p2

  subroutine acc_update_device_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_l_3d_p2

  subroutine acc_update_device_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_l_4d_p2

  subroutine acc_update_device_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_c_1d_p2

  subroutine acc_update_device_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_c_2d_p2

  subroutine acc_update_device_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_c_3d_p2

  subroutine acc_update_device_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_c_4d_p2

  subroutine acc_update_device_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_i_l_p3

  subroutine acc_update_device_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_r_l_p3

  subroutine acc_update_device_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_l_l_p3

  subroutine acc_update_device_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_device_async_c_l_p3

  ! Signatures for acc_update_self
  subroutine acc_update_self_i_1d_p1( a )
      integer, dimension(:) :: a
  end subroutine acc_update_self_i_1d_p1

  subroutine acc_update_self_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end subroutine acc_update_self_i_2d_p1

  subroutine acc_update_self_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end subroutine acc_update_self_i_3d_p1

  subroutine acc_update_self_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end subroutine acc_update_self_i_4d_p1

  subroutine acc_update_self_r_1d_p1( a )
      real(8), dimension(:) :: a
  end subroutine acc_update_self_r_1d_p1

  subroutine acc_update_self_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end subroutine acc_update_self_r_2d_p1

  subroutine acc_update_self_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end subroutine acc_update_self_r_3d_p1

  subroutine acc_update_self_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end subroutine acc_update_self_r_4d_p1

  subroutine acc_update_self_l_1d_p1( a )
      logical, dimension(:) :: a
  end subroutine acc_update_self_l_1d_p1

  subroutine acc_update_self_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end subroutine acc_update_self_l_2d_p1

  subroutine acc_update_self_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end subroutine acc_update_self_l_3d_p1

  subroutine acc_update_self_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end subroutine acc_update_self_l_4d_p1

  subroutine acc_update_self_c_1d_p1( a )
      character, dimension(:) :: a
  end subroutine acc_update_self_c_1d_p1

  subroutine acc_update_self_c_2d_p1( a )
      character, dimension(:,:) :: a
  end subroutine acc_update_self_c_2d_p1

  subroutine acc_update_self_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end subroutine acc_update_self_c_3d_p1

  subroutine acc_update_self_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end subroutine acc_update_self_c_4d_p1

  subroutine acc_update_self_i_l_p2( a, len )
      integer :: a
      integer :: len
  end subroutine acc_update_self_i_l_p2

  subroutine acc_update_self_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end subroutine acc_update_self_r_l_p2

  subroutine acc_update_self_l_l_p2( a, len )
      logical :: a
      integer :: len
  end subroutine acc_update_self_l_l_p2

  subroutine acc_update_self_c_l_p2( a, len )
      character :: a
      integer :: len
  end subroutine acc_update_self_c_l_p2

  ! Signatures for acc_update_self_async
  subroutine acc_update_self_async_i_1d_p2( a, async )
      integer, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_i_1d_p2

  subroutine acc_update_self_async_i_2d_p2( a, async )
      integer, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_i_2d_p2

  subroutine acc_update_self_async_i_3d_p2( a, async )
      integer, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_i_3d_p2

  subroutine acc_update_self_async_i_4d_p2( a, async )
      integer, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_i_4d_p2

  subroutine acc_update_self_async_r_1d_p2( a, async )
      real(8), dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_r_1d_p2

  subroutine acc_update_self_async_r_2d_p2( a, async )
      real(8), dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_r_2d_p2

  subroutine acc_update_self_async_r_3d_p2( a, async )
      real(8), dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_r_3d_p2

  subroutine acc_update_self_async_r_4d_p2( a, async )
      real(8), dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_r_4d_p2

  subroutine acc_update_self_async_l_1d_p2( a, async )
      logical, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_l_1d_p2

  subroutine acc_update_self_async_l_2d_p2( a, async )
      logical, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_l_2d_p2

  subroutine acc_update_self_async_l_3d_p2( a, async )
      logical, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_l_3d_p2

  subroutine acc_update_self_async_l_4d_p2( a, async )
      logical, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_l_4d_p2

  subroutine acc_update_self_async_c_1d_p2( a, async )
      character, dimension(:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_c_1d_p2

  subroutine acc_update_self_async_c_2d_p2( a, async )
      character, dimension(:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_c_2d_p2

  subroutine acc_update_self_async_c_3d_p2( a, async )
      character, dimension(:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_c_3d_p2

  subroutine acc_update_self_async_c_4d_p2( a, async )
      character, dimension(:,:,:,:) :: a
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_c_4d_p2

  subroutine acc_update_self_async_i_l_p3( a, len, async )
      integer :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_i_l_p3

  subroutine acc_update_self_async_r_l_p3( a, len, async )
      real(8) :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_r_l_p3

  subroutine acc_update_self_async_l_l_p3( a, len, async )
      logical :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_l_l_p3

  subroutine acc_update_self_async_c_l_p3( a, len, async )
      character :: a
      integer :: len
      integer(acc_handle_kind) :: async
  end subroutine acc_update_self_async_c_l_p3

  ! Signatures for acc_is_present
  logical function acc_is_present_i_1d_p1( a )
      integer, dimension(:) :: a
  end function acc_is_present_i_1d_p1

  logical function acc_is_present_i_2d_p1( a )
      integer, dimension(:,:) :: a
  end function acc_is_present_i_2d_p1

  logical function acc_is_present_i_3d_p1( a )
      integer, dimension(:,:,:) :: a
  end function acc_is_present_i_3d_p1

  logical function acc_is_present_i_4d_p1( a )
      integer, dimension(:,:,:,:) :: a
  end function acc_is_present_i_4d_p1

  logical function acc_is_present_r_1d_p1( a )
      real(8), dimension(:) :: a
  end function acc_is_present_r_1d_p1

  logical function acc_is_present_r_2d_p1( a )
      real(8), dimension(:,:) :: a
  end function acc_is_present_r_2d_p1

  logical function acc_is_present_r_3d_p1( a )
      real(8), dimension(:,:,:) :: a
  end function acc_is_present_r_3d_p1

  logical function acc_is_present_r_4d_p1( a )
      real(8), dimension(:,:,:,:) :: a
  end function acc_is_present_r_4d_p1

  logical function acc_is_present_l_1d_p1( a )
      logical, dimension(:) :: a
  end function acc_is_present_l_1d_p1

  logical function acc_is_present_l_2d_p1( a )
      logical, dimension(:,:) :: a
  end function acc_is_present_l_2d_p1

  logical function acc_is_present_l_3d_p1( a )
      logical, dimension(:,:,:) :: a
  end function acc_is_present_l_3d_p1

  logical function acc_is_present_l_4d_p1( a )
      logical, dimension(:,:,:,:) :: a
  end function acc_is_present_l_4d_p1

  logical function acc_is_present_c_1d_p1( a )
      character, dimension(:) :: a
  end function acc_is_present_c_1d_p1

  logical function acc_is_present_c_2d_p1( a )
      character, dimension(:,:) :: a
  end function acc_is_present_c_2d_p1

  logical function acc_is_present_c_3d_p1( a )
      character, dimension(:,:,:) :: a
  end function acc_is_present_c_3d_p1

  logical function acc_is_present_c_4d_p1( a )
      character, dimension(:,:,:,:) :: a
  end function acc_is_present_c_4d_p1

  logical function acc_is_present_i_l_p2( a, len )
      integer :: a
      integer :: len
  end function acc_is_present_i_l_p2

  logical function acc_is_present_r_l_p2( a, len )
      real(8) :: a
      integer :: len
  end function acc_is_present_r_l_p2

  logical function acc_is_present_l_l_p2( a, len )
      logical :: a
      integer :: len
  end function acc_is_present_l_l_p2

  logical function acc_is_present_c_l_p2( a, len )
      character :: a
      integer :: len
  end function acc_is_present_c_l_p2

end module openacc
