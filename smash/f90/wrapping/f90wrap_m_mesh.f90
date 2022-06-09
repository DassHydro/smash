! Module m_mesh defined in file smash/f90/wrapped_module/m_mesh.f90

subroutine f90wrap_meshdt__get__nbx(this, f90wrap_nbx)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nbx
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nbx = this_ptr%p%nbx
end subroutine f90wrap_meshdt__get__nbx

subroutine f90wrap_meshdt__set__nbx(this, f90wrap_nbx)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nbx
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nbx = f90wrap_nbx
end subroutine f90wrap_meshdt__set__nbx

subroutine f90wrap_meshdt__get__nby(this, f90wrap_nby)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nby
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nby = this_ptr%p%nby
end subroutine f90wrap_meshdt__get__nby

subroutine f90wrap_meshdt__set__nby(this, f90wrap_nby)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nby
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nby = f90wrap_nby
end subroutine f90wrap_meshdt__set__nby

subroutine f90wrap_meshdt__get__nbc(this, f90wrap_nbc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nbc
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nbc = this_ptr%p%nbc
end subroutine f90wrap_meshdt__get__nbc

subroutine f90wrap_meshdt__set__nbc(this, f90wrap_nbc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nbc
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nbc = f90wrap_nbc
end subroutine f90wrap_meshdt__set__nbc

subroutine f90wrap_meshdt__get__xll(this, f90wrap_xll)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_xll
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_xll = this_ptr%p%xll
end subroutine f90wrap_meshdt__get__xll

subroutine f90wrap_meshdt__set__xll(this, f90wrap_xll)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_xll
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%xll = f90wrap_xll
end subroutine f90wrap_meshdt__set__xll

subroutine f90wrap_meshdt__get__yll(this, f90wrap_yll)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_yll
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_yll = this_ptr%p%yll
end subroutine f90wrap_meshdt__get__yll

subroutine f90wrap_meshdt__set__yll(this, f90wrap_yll)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_yll
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%yll = f90wrap_yll
end subroutine f90wrap_meshdt__set__yll

subroutine f90wrap_meshdt__array__flow(this, nd, dtype, dshape, dloc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: nd
    integer, intent(out) :: dtype
    integer, dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%flow)) then
        dshape(1:2) = shape(this_ptr%p%flow)
        dloc = loc(this_ptr%p%flow)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__flow

subroutine f90wrap_meshdt__array__drained_area(this, nd, dtype, dshape, dloc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: nd
    integer, intent(out) :: dtype
    integer, dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%drained_area)) then
        dshape(1:2) = shape(this_ptr%p%drained_area)
        dloc = loc(this_ptr%p%drained_area)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__drained_area

subroutine f90wrap_meshdt__array__code(this, nd, dtype, dshape, dloc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: nd
    integer, intent(out) :: dtype
    integer, dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 2
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%code)) then
        dshape(1:2) = (/len(this_ptr%p%code(1)), shape(this_ptr%p%code)/)
        dloc = loc(this_ptr%p%code)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__code

subroutine f90wrap_meshdt__array__area(this, nd, dtype, dshape, dloc)
    use m_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: nd
    integer, intent(out) :: dtype
    integer, dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 12
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%area)) then
        dshape(1:1) = shape(this_ptr%p%area)
        dloc = loc(this_ptr%p%area)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__area

subroutine f90wrap_meshdt_initialise(mesh, setup, nbx, nby, nbc)
    use m_mesh, only: meshdt, meshdt_initialise
    use m_setup, only: setupdt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(out), dimension(2) :: mesh
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    integer, intent(in) :: nbx
    integer, intent(in) :: nby
    integer, intent(in) :: nbc
    setup_ptr = transfer(setup, setup_ptr)
    allocate(mesh_ptr%p)
    call meshdt_initialise(mesh=mesh_ptr%p, setup=setup_ptr%p, nbx=nbx, nby=nby, nbc=nbc)
    mesh = transfer(mesh_ptr, mesh)
end subroutine f90wrap_meshdt_initialise

subroutine f90wrap_meshdt_finalise(this)
    use m_mesh, only: meshdt
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_meshdt_finalise

! End of module m_mesh defined in file smash/f90/wrapped_module/m_mesh.f90

