! Module mwd_common defined in file smash/solver/module/mwd_common.f90

subroutine f90wrap_mwd_common__get__sp(f90wrap_sp)
    use mwd_common, only: mwd_common_sp => sp
    implicit none
    integer, intent(out) :: f90wrap_sp
    
    f90wrap_sp = mwd_common_sp
end subroutine f90wrap_mwd_common__get__sp

subroutine f90wrap_mwd_common__get__dp(f90wrap_dp)
    use mwd_common, only: mwd_common_dp => dp
    implicit none
    integer, intent(out) :: f90wrap_dp
    
    f90wrap_dp = mwd_common_dp
end subroutine f90wrap_mwd_common__get__dp

subroutine f90wrap_mwd_common__get__lchar(f90wrap_lchar)
    use mwd_common, only: mwd_common_lchar => lchar
    implicit none
    integer, intent(out) :: f90wrap_lchar
    
    f90wrap_lchar = mwd_common_lchar
end subroutine f90wrap_mwd_common__get__lchar

subroutine f90wrap_mwd_common__get__np(f90wrap_np)
    use mwd_common, only: mwd_common_np => np
    implicit none
    integer, intent(out) :: f90wrap_np
    
    f90wrap_np = mwd_common_np
end subroutine f90wrap_mwd_common__get__np

subroutine f90wrap_mwd_common__get__ns(f90wrap_ns)
    use mwd_common, only: mwd_common_ns => ns
    implicit none
    integer, intent(out) :: f90wrap_ns
    
    f90wrap_ns = mwd_common_ns
end subroutine f90wrap_mwd_common__get__ns

subroutine f90wrap_mwd_common__array__name_parameters(dummy_this, nd, dtype, dshape, dloc)
    use mwd_common, only: mwd_common_name_parameters => name_parameters
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    integer, intent(in) :: dummy_this(2)
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 2
    dshape(1:2) = (/len(mwd_common_name_parameters(1)), shape(mwd_common_name_parameters)/)
    dloc = loc(mwd_common_name_parameters)
end subroutine f90wrap_mwd_common__array__name_parameters

subroutine f90wrap_mwd_common__array__name_states(dummy_this, nd, dtype, dshape, dloc)
    use mwd_common, only: mwd_common_name_states => name_states
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    integer, intent(in) :: dummy_this(2)
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 2
    dshape(1:2) = (/len(mwd_common_name_states(1)), shape(mwd_common_name_states)/)
    dloc = loc(mwd_common_name_states)
end subroutine f90wrap_mwd_common__array__name_states

! End of module mwd_common defined in file smash/solver/module/mwd_common.f90

