! Module mwd_parameters defined in file smash/solver/module/mwd_parameters.f90

subroutine f90wrap_parametersdt__array__ci(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%ci)) then
        dshape(1:2) = shape(this_ptr%p%ci)
        dloc = loc(this_ptr%p%ci)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__ci

subroutine f90wrap_parametersdt__array__cp(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%cp)) then
        dshape(1:2) = shape(this_ptr%p%cp)
        dloc = loc(this_ptr%p%cp)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__cp

subroutine f90wrap_parametersdt__array__beta(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%beta)) then
        dshape(1:2) = shape(this_ptr%p%beta)
        dloc = loc(this_ptr%p%beta)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__beta

subroutine f90wrap_parametersdt__array__cft(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%cft)) then
        dshape(1:2) = shape(this_ptr%p%cft)
        dloc = loc(this_ptr%p%cft)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__cft

subroutine f90wrap_parametersdt__array__cst(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%cst)) then
        dshape(1:2) = shape(this_ptr%p%cst)
        dloc = loc(this_ptr%p%cst)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__cst

subroutine f90wrap_parametersdt__array__alpha(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%alpha)) then
        dshape(1:2) = shape(this_ptr%p%alpha)
        dloc = loc(this_ptr%p%alpha)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__alpha

subroutine f90wrap_parametersdt__array__exc(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%exc)) then
        dshape(1:2) = shape(this_ptr%p%exc)
        dloc = loc(this_ptr%p%exc)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__exc

subroutine f90wrap_parametersdt__array__lr(this, nd, dtype, dshape, dloc)
    use mwd_parameters, only: parametersdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(parametersdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%lr)) then
        dshape(1:2) = shape(this_ptr%p%lr)
        dloc = loc(this_ptr%p%lr)
    else
        dloc = 0
    end if
end subroutine f90wrap_parametersdt__array__lr

subroutine f90wrap_parametersdt_initialise(parameters, mesh)
    use mwd_mesh, only: meshdt
    use mwd_parameters, only: parametersdt_initialise, parametersdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(out), dimension(2) :: parameters
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    mesh_ptr = transfer(mesh, mesh_ptr)
    allocate(parameters_ptr%p)
    call parametersdt_initialise(parameters=parameters_ptr%p, mesh=mesh_ptr%p)
    parameters = transfer(parameters_ptr, parameters)
end subroutine f90wrap_parametersdt_initialise

subroutine f90wrap_parametersdt_finalise(this)
    use mwd_parameters, only: parametersdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type(parametersdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_parametersdt_finalise

subroutine f90wrap_parameters_to_matrix(parameters, matrix, n0, n1, n2)
    use mwd_parameters, only: parametersdt, parameters_to_matrix
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    real(4), intent(inout), dimension(n0,n1,n2) :: matrix
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,2)
    parameters_ptr = transfer(parameters, parameters_ptr)
    call parameters_to_matrix(parameters=parameters_ptr%p, matrix=matrix)
end subroutine f90wrap_parameters_to_matrix

subroutine f90wrap_matrix_to_parameters(matrix, parameters, n0, n1, n2)
    use mwd_parameters, only: parametersdt, matrix_to_parameters
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    real(4), intent(in), dimension(n0,n1,n2) :: matrix
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,2)
    parameters_ptr = transfer(parameters, parameters_ptr)
    call matrix_to_parameters(matrix=matrix, parameters=parameters_ptr%p)
end subroutine f90wrap_matrix_to_parameters

subroutine f90wrap_vector_to_parameters(vector, parameters, n0)
    use mwd_parameters, only: vector_to_parameters, parametersdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    real(4), intent(in), dimension(n0) :: vector
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    integer :: n0
    !f2py intent(hide), depend(vector) :: n0 = shape(vector,0)
    parameters_ptr = transfer(parameters, parameters_ptr)
    call vector_to_parameters(vector=vector, parameters=parameters_ptr%p)
end subroutine f90wrap_vector_to_parameters

subroutine f90wrap_set0_parameters(parameters)
    use mwd_parameters, only: set0_parameters, parametersdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    parameters_ptr = transfer(parameters, parameters_ptr)
    call set0_parameters(parameters=parameters_ptr%p)
end subroutine f90wrap_set0_parameters

subroutine f90wrap_set1_parameters(parameters)
    use mwd_parameters, only: parametersdt, set1_parameters
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    parameters_ptr = transfer(parameters, parameters_ptr)
    call set1_parameters(parameters=parameters_ptr%p)
end subroutine f90wrap_set1_parameters

! End of module mwd_parameters defined in file smash/solver/module/mwd_parameters.f90

