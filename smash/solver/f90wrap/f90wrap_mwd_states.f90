! Module mwd_states defined in file smash/solver/module/mwd_states.f90

subroutine f90wrap_statesdt__array__hi(this, nd, dtype, dshape, dloc)
    use mwd_states, only: statesdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(statesdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%hi)) then
        dshape(1:2) = shape(this_ptr%p%hi)
        dloc = loc(this_ptr%p%hi)
    else
        dloc = 0
    end if
end subroutine f90wrap_statesdt__array__hi

subroutine f90wrap_statesdt__array__hp(this, nd, dtype, dshape, dloc)
    use mwd_states, only: statesdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(statesdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%hp)) then
        dshape(1:2) = shape(this_ptr%p%hp)
        dloc = loc(this_ptr%p%hp)
    else
        dloc = 0
    end if
end subroutine f90wrap_statesdt__array__hp

subroutine f90wrap_statesdt__array__hft(this, nd, dtype, dshape, dloc)
    use mwd_states, only: statesdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(statesdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%hft)) then
        dshape(1:2) = shape(this_ptr%p%hft)
        dloc = loc(this_ptr%p%hft)
    else
        dloc = 0
    end if
end subroutine f90wrap_statesdt__array__hft

subroutine f90wrap_statesdt__array__hst(this, nd, dtype, dshape, dloc)
    use mwd_states, only: statesdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(statesdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%hst)) then
        dshape(1:2) = shape(this_ptr%p%hst)
        dloc = loc(this_ptr%p%hst)
    else
        dloc = 0
    end if
end subroutine f90wrap_statesdt__array__hst

subroutine f90wrap_statesdt__array__hlr(this, nd, dtype, dshape, dloc)
    use mwd_states, only: statesdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(statesdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%hlr)) then
        dshape(1:2) = shape(this_ptr%p%hlr)
        dloc = loc(this_ptr%p%hlr)
    else
        dloc = 0
    end if
end subroutine f90wrap_statesdt__array__hlr

subroutine f90wrap_statesdt_initialise(states, mesh)
    use mwd_states, only: statesdt, statesdt_initialise
    use mwd_mesh, only: meshdt
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(out), dimension(2) :: states
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    mesh_ptr = transfer(mesh, mesh_ptr)
    allocate(states_ptr%p)
    call statesdt_initialise(states=states_ptr%p, mesh=mesh_ptr%p)
    states = transfer(states_ptr, states)
end subroutine f90wrap_statesdt_initialise

subroutine f90wrap_statesdt_finalise(this)
    use mwd_states, only: statesdt
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type(statesdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_statesdt_finalise

subroutine f90wrap_states_to_matrix(states, matrix, n0, n1, n2)
    use mwd_states, only: statesdt, states_to_matrix
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    real(4), intent(inout), dimension(n0,n1,n2) :: matrix
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,2)
    states_ptr = transfer(states, states_ptr)
    call states_to_matrix(states=states_ptr%p, matrix=matrix)
end subroutine f90wrap_states_to_matrix

subroutine f90wrap_matrix_to_states(matrix, states, n0, n1, n2)
    use mwd_states, only: statesdt, matrix_to_states
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    real(4), intent(in), dimension(n0,n1,n2) :: matrix
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,2)
    states_ptr = transfer(states, states_ptr)
    call matrix_to_states(matrix=matrix, states=states_ptr%p)
end subroutine f90wrap_matrix_to_states

subroutine f90wrap_vector_to_states(vector, states, n0)
    use mwd_states, only: statesdt, vector_to_states
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    real(4), intent(in), dimension(n0) :: vector
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    integer :: n0
    !f2py intent(hide), depend(vector) :: n0 = shape(vector,0)
    states_ptr = transfer(states, states_ptr)
    call vector_to_states(vector=vector, states=states_ptr%p)
end subroutine f90wrap_vector_to_states

subroutine f90wrap_set0_states(states)
    use mwd_states, only: statesdt, set0_states
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    states_ptr = transfer(states, states_ptr)
    call set0_states(states=states_ptr%p)
end subroutine f90wrap_set0_states

subroutine f90wrap_set1_states(states)
    use mwd_states, only: statesdt, set1_states
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    states_ptr = transfer(states, states_ptr)
    call set1_states(states=states_ptr%p)
end subroutine f90wrap_set1_states

! End of module mwd_states defined in file smash/solver/module/mwd_states.f90

