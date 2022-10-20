! Module mwd_output defined in file smash/solver/module/mwd_output.f90

subroutine f90wrap_outputdt__array__qsim(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%qsim)) then
        dshape(1:2) = shape(this_ptr%p%qsim)
        dloc = loc(this_ptr%p%qsim)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__qsim

subroutine f90wrap_outputdt__array__qsim_domain(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%qsim_domain)) then
        dshape(1:3) = shape(this_ptr%p%qsim_domain)
        dloc = loc(this_ptr%p%qsim_domain)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__qsim_domain

subroutine f90wrap_outputdt__array__sparse_qsim_domain(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%sparse_qsim_domain)) then
        dshape(1:2) = shape(this_ptr%p%sparse_qsim_domain)
        dloc = loc(this_ptr%p%sparse_qsim_domain)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__sparse_qsim_domain

subroutine f90wrap_outputdt__array__net_prcp_domain(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%net_prcp_domain)) then
        dshape(1:3) = shape(this_ptr%p%net_prcp_domain)
        dloc = loc(this_ptr%p%net_prcp_domain)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__net_prcp_domain

subroutine f90wrap_outputdt__array__sparse_net_prcp_domain(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%sparse_net_prcp_domain)) then
        dshape(1:2) = shape(this_ptr%p%sparse_net_prcp_domain)
        dloc = loc(this_ptr%p%sparse_net_prcp_domain)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__sparse_net_prcp_domain

subroutine f90wrap_outputdt__array__parameters_gradient(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%parameters_gradient)) then
        dshape(1:3) = shape(this_ptr%p%parameters_gradient)
        dloc = loc(this_ptr%p%parameters_gradient)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__parameters_gradient

subroutine f90wrap_outputdt__get__cost(this, f90wrap_cost)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_cost
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_cost = this_ptr%p%cost
end subroutine f90wrap_outputdt__get__cost

subroutine f90wrap_outputdt__set__cost(this, f90wrap_cost)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_cost
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%cost = f90wrap_cost
end subroutine f90wrap_outputdt__set__cost

subroutine f90wrap_outputdt__get__sp1(this, f90wrap_sp1)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_sp1
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_sp1 = this_ptr%p%sp1
end subroutine f90wrap_outputdt__get__sp1

subroutine f90wrap_outputdt__set__sp1(this, f90wrap_sp1)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_sp1
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%sp1 = f90wrap_sp1
end subroutine f90wrap_outputdt__set__sp1

subroutine f90wrap_outputdt__get__sp2(this, f90wrap_sp2)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_sp2
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_sp2 = this_ptr%p%sp2
end subroutine f90wrap_outputdt__get__sp2

subroutine f90wrap_outputdt__set__sp2(this, f90wrap_sp2)
    use mwd_output, only: outputdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_sp2
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%sp2 = f90wrap_sp2
end subroutine f90wrap_outputdt__set__sp2

subroutine f90wrap_outputdt__array__an(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%an)) then
        dshape(1:1) = shape(this_ptr%p%an)
        dloc = loc(this_ptr%p%an)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__an

subroutine f90wrap_outputdt__array__ian(this, nd, dtype, dshape, dloc)
    use mwd_output, only: outputdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%ian)) then
        dshape(1:1) = shape(this_ptr%p%ian)
        dloc = loc(this_ptr%p%ian)
    else
        dloc = 0
    end if
end subroutine f90wrap_outputdt__array__ian

subroutine f90wrap_outputdt__get__fstates(this, f90wrap_fstates)
    use mwd_output, only: outputdt
    use mwd_states, only: statesdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_fstates(2)
    type(statesdt_ptr_type) :: fstates_ptr
    
    this_ptr = transfer(this, this_ptr)
    fstates_ptr%p => this_ptr%p%fstates
    f90wrap_fstates = transfer(fstates_ptr,f90wrap_fstates)
end subroutine f90wrap_outputdt__get__fstates

subroutine f90wrap_outputdt__set__fstates(this, f90wrap_fstates)
    use mwd_output, only: outputdt
    use mwd_states, only: statesdt
    implicit none
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer, intent(in)   :: this(2)
    type(outputdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_fstates(2)
    type(statesdt_ptr_type) :: fstates_ptr
    
    this_ptr = transfer(this, this_ptr)
    fstates_ptr = transfer(f90wrap_fstates,fstates_ptr)
    this_ptr%p%fstates = fstates_ptr%p
end subroutine f90wrap_outputdt__set__fstates

subroutine f90wrap_outputdt_initialise(output, setup, mesh)
    use mwd_output, only: outputdt, outputdt_initialise
    use mwd_setup, only: setupdt
    use mwd_mesh, only: meshdt
    implicit none
    
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(outputdt_ptr_type) :: output_ptr
    integer, intent(out), dimension(2) :: output
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    allocate(output_ptr%p)
    call outputdt_initialise(output=output_ptr%p, setup=setup_ptr%p, mesh=mesh_ptr%p)
    output = transfer(output_ptr, output)
end subroutine f90wrap_outputdt_initialise

subroutine f90wrap_outputdt_finalise(this)
    use mwd_output, only: outputdt
    implicit none
    
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type(outputdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_outputdt_finalise

! End of module mwd_output defined in file smash/solver/module/mwd_output.f90

