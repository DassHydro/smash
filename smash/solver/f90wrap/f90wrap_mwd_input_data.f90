! Module mwd_input_data defined in file smash/solver/module/mwd_input_data.f90

subroutine f90wrap_prcp_indicedt__array__p0(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%p0)) then
        dshape(1:2) = shape(this_ptr%p%p0)
        dloc = loc(this_ptr%p%p0)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__p0

subroutine f90wrap_prcp_indicedt__array__p1(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%p1)) then
        dshape(1:2) = shape(this_ptr%p%p1)
        dloc = loc(this_ptr%p%p1)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__p1

subroutine f90wrap_prcp_indicedt__array__p2(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%p2)) then
        dshape(1:2) = shape(this_ptr%p%p2)
        dloc = loc(this_ptr%p%p2)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__p2

subroutine f90wrap_prcp_indicedt__array__g1(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%g1)) then
        dshape(1:1) = shape(this_ptr%p%g1)
        dloc = loc(this_ptr%p%g1)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__g1

subroutine f90wrap_prcp_indicedt__array__g2(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%g2)) then
        dshape(1:1) = shape(this_ptr%p%g2)
        dloc = loc(this_ptr%p%g2)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__g2

subroutine f90wrap_prcp_indicedt__array__md1(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%md1)) then
        dshape(1:2) = shape(this_ptr%p%md1)
        dloc = loc(this_ptr%p%md1)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__md1

subroutine f90wrap_prcp_indicedt__array__md2(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%md2)) then
        dshape(1:2) = shape(this_ptr%p%md2)
        dloc = loc(this_ptr%p%md2)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__md2

subroutine f90wrap_prcp_indicedt__array__std(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%std)) then
        dshape(1:2) = shape(this_ptr%p%std)
        dloc = loc(this_ptr%p%std)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__std

subroutine f90wrap_prcp_indicedt__array__flwdst_qtl(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%flwdst_qtl)) then
        dshape(1:2) = shape(this_ptr%p%flwdst_qtl)
        dloc = loc(this_ptr%p%flwdst_qtl)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__flwdst_qtl

subroutine f90wrap_prcp_indicedt__array__wf(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%wf)) then
        dshape(1:2) = shape(this_ptr%p%wf)
        dloc = loc(this_ptr%p%wf)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__wf

subroutine f90wrap_prcp_indicedt__array__pwf(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%pwf)) then
        dshape(1:3) = shape(this_ptr%p%pwf)
        dloc = loc(this_ptr%p%pwf)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__pwf

subroutine f90wrap_prcp_indicedt__array__vg(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: prcp_indicedt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%vg)) then
        dshape(1:2) = shape(this_ptr%p%vg)
        dloc = loc(this_ptr%p%vg)
    else
        dloc = 0
    end if
end subroutine f90wrap_prcp_indicedt__array__vg

subroutine f90wrap_prcp_indicedt_initialise(prcp_indice, setup, mesh)
    use mwd_input_data, only: prcp_indicedt_initialise, prcp_indicedt
    use mwd_setup, only: setupdt
    use mwd_mesh, only: meshdt
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    type(prcp_indicedt_ptr_type) :: prcp_indice_ptr
    integer, intent(out), dimension(2) :: prcp_indice
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    allocate(prcp_indice_ptr%p)
    call prcp_indicedt_initialise(prcp_indice=prcp_indice_ptr%p, setup=setup_ptr%p, mesh=mesh_ptr%p)
    prcp_indice = transfer(prcp_indice_ptr, prcp_indice)
end subroutine f90wrap_prcp_indicedt_initialise

subroutine f90wrap_prcp_indicedt_finalise(this)
    use mwd_input_data, only: prcp_indicedt
    implicit none
    
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    type(prcp_indicedt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_prcp_indicedt_finalise

subroutine f90wrap_input_datadt__array__qobs(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%qobs)) then
        dshape(1:2) = shape(this_ptr%p%qobs)
        dloc = loc(this_ptr%p%qobs)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__qobs

subroutine f90wrap_input_datadt__array__prcp(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%prcp)) then
        dshape(1:3) = shape(this_ptr%p%prcp)
        dloc = loc(this_ptr%p%prcp)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__prcp

subroutine f90wrap_input_datadt__array__pet(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%pet)) then
        dshape(1:3) = shape(this_ptr%p%pet)
        dloc = loc(this_ptr%p%pet)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__pet

subroutine f90wrap_input_datadt__array__descriptor(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 3
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%descriptor)) then
        dshape(1:3) = shape(this_ptr%p%descriptor)
        dloc = loc(this_ptr%p%descriptor)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__descriptor

subroutine f90wrap_input_datadt__array__sparse_prcp(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%sparse_prcp)) then
        dshape(1:2) = shape(this_ptr%p%sparse_prcp)
        dloc = loc(this_ptr%p%sparse_prcp)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__sparse_prcp

subroutine f90wrap_input_datadt__array__sparse_pet(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%sparse_pet)) then
        dshape(1:2) = shape(this_ptr%p%sparse_pet)
        dloc = loc(this_ptr%p%sparse_pet)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__sparse_pet

subroutine f90wrap_input_datadt__array__mean_prcp(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%mean_prcp)) then
        dshape(1:2) = shape(this_ptr%p%mean_prcp)
        dloc = loc(this_ptr%p%mean_prcp)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__mean_prcp

subroutine f90wrap_input_datadt__array__mean_pet(this, nd, dtype, dshape, dloc)
    use mwd_input_data, only: input_datadt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%mean_pet)) then
        dshape(1:2) = shape(this_ptr%p%mean_pet)
        dloc = loc(this_ptr%p%mean_pet)
    else
        dloc = 0
    end if
end subroutine f90wrap_input_datadt__array__mean_pet

subroutine f90wrap_input_datadt__get__prcp_indice(this, f90wrap_prcp_indice)
    use mwd_input_data, only: prcp_indicedt, input_datadt
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer, intent(in)   :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_prcp_indice(2)
    type(prcp_indicedt_ptr_type) :: prcp_indice_ptr
    
    this_ptr = transfer(this, this_ptr)
    prcp_indice_ptr%p => this_ptr%p%prcp_indice
    f90wrap_prcp_indice = transfer(prcp_indice_ptr,f90wrap_prcp_indice)
end subroutine f90wrap_input_datadt__get__prcp_indice

subroutine f90wrap_input_datadt__set__prcp_indice(this, f90wrap_prcp_indice)
    use mwd_input_data, only: prcp_indicedt, input_datadt
    implicit none
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type prcp_indicedt_ptr_type
        type(prcp_indicedt), pointer :: p => NULL()
    end type prcp_indicedt_ptr_type
    integer, intent(in)   :: this(2)
    type(input_datadt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_prcp_indice(2)
    type(prcp_indicedt_ptr_type) :: prcp_indice_ptr
    
    this_ptr = transfer(this, this_ptr)
    prcp_indice_ptr = transfer(f90wrap_prcp_indice,prcp_indice_ptr)
    this_ptr%p%prcp_indice = prcp_indice_ptr%p
end subroutine f90wrap_input_datadt__set__prcp_indice

subroutine f90wrap_input_datadt_initialise(input_data, setup, mesh)
    use mwd_input_data, only: input_datadt, input_datadt_initialise
    use mwd_setup, only: setupdt
    use mwd_mesh, only: meshdt
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(out), dimension(2) :: input_data
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    allocate(input_data_ptr%p)
    call input_datadt_initialise(input_data=input_data_ptr%p, setup=setup_ptr%p, mesh=mesh_ptr%p)
    input_data = transfer(input_data_ptr, input_data)
end subroutine f90wrap_input_datadt_initialise

subroutine f90wrap_input_datadt_finalise(this)
    use mwd_input_data, only: input_datadt
    implicit none
    
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type(input_datadt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_input_datadt_finalise

! End of module mwd_input_data defined in file smash/solver/module/mwd_input_data.f90

