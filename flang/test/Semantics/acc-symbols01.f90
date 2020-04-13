! RUN: %S/test_symbols.sh %s %flang %t
! OPTIONS: -fopenacc

! Test clauses that accept list.
! 2.1 Directive Format
!   A list consists of a comma-separated collection of one or more list items.
!   A list item is a variable, array section or common block name (enclosed in
!   slashes).

!DEF: /mm MainProgram
program mm
  !DEF: /mm/x ObjectEntity REAL(4)
  !DEF: /mm/y ObjectEntity REAL(4)
  real x, y
  !DEF: /mm/a (AccPrivate) ObjectEntity INTEGER(4)
  !DEF: /mm/b (AccFirstPrivate) ObjectEntity INTEGER(4)
  !DEF: /mm/c ObjectEntity INTEGER(4)
  !DEF: /mm/i (AccPrivate, AccPreDetermined) ObjectEntity INTEGER(4)
  integer a(10), b(10), c(10), i
  !REF: /mm/b
  b = 2
  !$acc parallel present(c) firstprivate(b) private(a)
  !$acc loop
  !REF: /mm/i
  do i=1,10
   !REF: /mm/a
   !REF: /mm/i
   !REF: /mm/b
   a(i) = b(i)
  end do
  !$acc end parallel
 end program
 