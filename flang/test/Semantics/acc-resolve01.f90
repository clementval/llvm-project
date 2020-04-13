! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
! OPTIONS: -fopenacc

! Data-Sharing Attribute Clauses
! 2.15.14 default Clause

subroutine default_none()
  integer a(3)

  A = 1
  B = 2
  !$acc parallel default(none) private(c)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing clause
  A(1:2) = 3
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing clause
  B = 4
  C = 5
  !$acc end parallel
end subroutine default_none

program mm
  call default_none()
  !TODO: private, firstprivate, shared
end
