! Module mwd_mesh defined in file smash/solver/module/mwd_mesh.f90

subroutine f90wrap_meshdt__get__dx(this, f90wrap_dx)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_dx
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_dx = this_ptr%p%dx
end subroutine f90wrap_meshdt__get__dx

subroutine f90wrap_meshdt__set__dx(this, f90wrap_dx)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_dx
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%dx = f90wrap_dx
end subroutine f90wrap_meshdt__set__dx

subroutine f90wrap_meshdt__get__nrow(this, f90wrap_nrow)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nrow
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nrow = this_ptr%p%nrow
end subroutine f90wrap_meshdt__get__nrow

subroutine f90wrap_meshdt__set__nrow(this, f90wrap_nrow)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nrow
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nrow = f90wrap_nrow
end subroutine f90wrap_meshdt__set__nrow

subroutine f90wrap_meshdt__get__ncol(this, f90wrap_ncol)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_ncol
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_ncol = this_ptr%p%ncol
end subroutine f90wrap_meshdt__get__ncol

subroutine f90wrap_meshdt__set__ncol(this, f90wrap_ncol)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_ncol
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%ncol = f90wrap_ncol
end subroutine f90wrap_meshdt__set__ncol

subroutine f90wrap_meshdt__get__ng(this, f90wrap_ng)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_ng
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_ng = this_ptr%p%ng
end subroutine f90wrap_meshdt__get__ng

subroutine f90wrap_meshdt__set__ng(this, f90wrap_ng)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_ng
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%ng = f90wrap_ng
end subroutine f90wrap_meshdt__set__ng

subroutine f90wrap_meshdt__get__nac(this, f90wrap_nac)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nac
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nac = this_ptr%p%nac
end subroutine f90wrap_meshdt__get__nac

subroutine f90wrap_meshdt__set__nac(this, f90wrap_nac)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nac
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nac = f90wrap_nac
end subroutine f90wrap_meshdt__set__nac

subroutine f90wrap_meshdt__get__xmin(this, f90wrap_xmin)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_xmin
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_xmin = this_ptr%p%xmin
end subroutine f90wrap_meshdt__get__xmin

subroutine f90wrap_meshdt__set__xmin(this, f90wrap_xmin)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_xmin
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%xmin = f90wrap_xmin
end subroutine f90wrap_meshdt__set__xmin

subroutine f90wrap_meshdt__get__ymax(this, f90wrap_ymax)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_ymax
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_ymax = this_ptr%p%ymax
end subroutine f90wrap_meshdt__get__ymax

subroutine f90wrap_meshdt__set__ymax(this, f90wrap_ymax)
    use mwd_mesh, only: meshdt
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in)   :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_ymax
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%ymax = f90wrap_ymax
end subroutine f90wrap_meshdt__set__ymax

subroutine f90wrap_meshdt__array__flwdir(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%flwdir)) then
        dshape(1:2) = shape(this_ptr%p%flwdir)
        dloc = loc(this_ptr%p%flwdir)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__flwdir

subroutine f90wrap_meshdt__array__drained_area(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
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

subroutine f90wrap_meshdt__array__path(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%path)) then
        dshape(1:2) = shape(this_ptr%p%path)
        dloc = loc(this_ptr%p%path)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__path

subroutine f90wrap_meshdt__array__active_cell(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%active_cell)) then
        dshape(1:2) = shape(this_ptr%p%active_cell)
        dloc = loc(this_ptr%p%active_cell)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__active_cell

subroutine f90wrap_meshdt__array__flwdst(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%flwdst)) then
        dshape(1:2) = shape(this_ptr%p%flwdst)
        dloc = loc(this_ptr%p%flwdst)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__flwdst

subroutine f90wrap_meshdt__array__gauge_pos(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%gauge_pos)) then
        dshape(1:2) = shape(this_ptr%p%gauge_pos)
        dloc = loc(this_ptr%p%gauge_pos)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__gauge_pos

subroutine f90wrap_meshdt__array__code(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
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
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%area)) then
        dshape(1:1) = shape(this_ptr%p%area)
        dloc = loc(this_ptr%p%area)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__area

subroutine f90wrap_meshdt__array__wgauge(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%wgauge)) then
        dshape(1:1) = shape(this_ptr%p%wgauge)
        dloc = loc(this_ptr%p%wgauge)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__wgauge

subroutine f90wrap_meshdt__array__rowcol_to_ind_sparse(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%rowcol_to_ind_sparse)) then
        dshape(1:2) = shape(this_ptr%p%rowcol_to_ind_sparse)
        dloc = loc(this_ptr%p%rowcol_to_ind_sparse)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__rowcol_to_ind_sparse

subroutine f90wrap_meshdt__array__local_active_cell(this, nd, dtype, dshape, dloc)
    use mwd_mesh, only: meshdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(meshdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%local_active_cell)) then
        dshape(1:2) = shape(this_ptr%p%local_active_cell)
        dloc = loc(this_ptr%p%local_active_cell)
    else
        dloc = 0
    end if
end subroutine f90wrap_meshdt__array__local_active_cell

subroutine f90wrap_meshdt_initialise(mesh, setup, nrow, ncol, ng)
    use mwd_setup, only: setupdt
    use mwd_mesh, only: meshdt_initialise, meshdt
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
    integer, intent(in) :: nrow
    integer, intent(in) :: ncol
    integer, intent(in) :: ng
    setup_ptr = transfer(setup, setup_ptr)
    allocate(mesh_ptr%p)
    call meshdt_initialise(mesh=mesh_ptr%p, setup=setup_ptr%p, nrow=nrow, ncol=ncol, ng=ng)
    mesh = transfer(mesh_ptr, mesh)
end subroutine f90wrap_meshdt_initialise

subroutine f90wrap_meshdt_finalise(this)
    use mwd_mesh, only: meshdt
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_meshdt_finalise

! End of module mwd_mesh defined in file smash/solver/module/mwd_mesh.f90

