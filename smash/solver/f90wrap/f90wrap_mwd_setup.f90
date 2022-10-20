! Module mwd_setup defined in file smash/solver/module/mwd_setup.f90

subroutine f90wrap_setupdt__get__dt(this, f90wrap_dt)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_dt
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_dt = this_ptr%p%dt
end subroutine f90wrap_setupdt__get__dt

subroutine f90wrap_setupdt__set__dt(this, f90wrap_dt)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_dt
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%dt = f90wrap_dt
end subroutine f90wrap_setupdt__set__dt

subroutine f90wrap_setupdt__get__start_time(this, f90wrap_start_time)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_start_time
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_start_time = this_ptr%p%start_time
end subroutine f90wrap_setupdt__get__start_time

subroutine f90wrap_setupdt__set__start_time(this, f90wrap_start_time)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_start_time
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%start_time = f90wrap_start_time
end subroutine f90wrap_setupdt__set__start_time

subroutine f90wrap_setupdt__get__end_time(this, f90wrap_end_time)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_end_time
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_end_time = this_ptr%p%end_time
end subroutine f90wrap_setupdt__get__end_time

subroutine f90wrap_setupdt__set__end_time(this, f90wrap_end_time)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_end_time
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%end_time = f90wrap_end_time
end subroutine f90wrap_setupdt__set__end_time

subroutine f90wrap_setupdt__get__sparse_storage(this, f90wrap_sparse_storage)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_sparse_storage
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_sparse_storage = this_ptr%p%sparse_storage
end subroutine f90wrap_setupdt__get__sparse_storage

subroutine f90wrap_setupdt__set__sparse_storage(this, f90wrap_sparse_storage)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_sparse_storage
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%sparse_storage = f90wrap_sparse_storage
end subroutine f90wrap_setupdt__set__sparse_storage

subroutine f90wrap_setupdt__get__read_qobs(this, f90wrap_read_qobs)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_read_qobs
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_read_qobs = this_ptr%p%read_qobs
end subroutine f90wrap_setupdt__get__read_qobs

subroutine f90wrap_setupdt__set__read_qobs(this, f90wrap_read_qobs)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_read_qobs
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%read_qobs = f90wrap_read_qobs
end subroutine f90wrap_setupdt__set__read_qobs

subroutine f90wrap_setupdt__get__qobs_directory(this, f90wrap_qobs_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_qobs_directory
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_qobs_directory = this_ptr%p%qobs_directory
end subroutine f90wrap_setupdt__get__qobs_directory

subroutine f90wrap_setupdt__set__qobs_directory(this, f90wrap_qobs_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_qobs_directory
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%qobs_directory = f90wrap_qobs_directory
end subroutine f90wrap_setupdt__set__qobs_directory

subroutine f90wrap_setupdt__get__read_prcp(this, f90wrap_read_prcp)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_read_prcp
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_read_prcp = this_ptr%p%read_prcp
end subroutine f90wrap_setupdt__get__read_prcp

subroutine f90wrap_setupdt__set__read_prcp(this, f90wrap_read_prcp)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_read_prcp
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%read_prcp = f90wrap_read_prcp
end subroutine f90wrap_setupdt__set__read_prcp

subroutine f90wrap_setupdt__get__prcp_format(this, f90wrap_prcp_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_prcp_format
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_prcp_format = this_ptr%p%prcp_format
end subroutine f90wrap_setupdt__get__prcp_format

subroutine f90wrap_setupdt__set__prcp_format(this, f90wrap_prcp_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_prcp_format
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%prcp_format = f90wrap_prcp_format
end subroutine f90wrap_setupdt__set__prcp_format

subroutine f90wrap_setupdt__get__prcp_conversion_factor(this, f90wrap_prcp_conversion_factor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_prcp_conversion_factor
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_prcp_conversion_factor = this_ptr%p%prcp_conversion_factor
end subroutine f90wrap_setupdt__get__prcp_conversion_factor

subroutine f90wrap_setupdt__set__prcp_conversion_factor(this, f90wrap_prcp_conversion_factor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_prcp_conversion_factor
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%prcp_conversion_factor = f90wrap_prcp_conversion_factor
end subroutine f90wrap_setupdt__set__prcp_conversion_factor

subroutine f90wrap_setupdt__get__prcp_directory(this, f90wrap_prcp_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_prcp_directory
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_prcp_directory = this_ptr%p%prcp_directory
end subroutine f90wrap_setupdt__get__prcp_directory

subroutine f90wrap_setupdt__set__prcp_directory(this, f90wrap_prcp_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_prcp_directory
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%prcp_directory = f90wrap_prcp_directory
end subroutine f90wrap_setupdt__set__prcp_directory

subroutine f90wrap_setupdt__get__read_pet(this, f90wrap_read_pet)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_read_pet
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_read_pet = this_ptr%p%read_pet
end subroutine f90wrap_setupdt__get__read_pet

subroutine f90wrap_setupdt__set__read_pet(this, f90wrap_read_pet)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_read_pet
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%read_pet = f90wrap_read_pet
end subroutine f90wrap_setupdt__set__read_pet

subroutine f90wrap_setupdt__get__pet_format(this, f90wrap_pet_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_pet_format
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_pet_format = this_ptr%p%pet_format
end subroutine f90wrap_setupdt__get__pet_format

subroutine f90wrap_setupdt__set__pet_format(this, f90wrap_pet_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_pet_format
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%pet_format = f90wrap_pet_format
end subroutine f90wrap_setupdt__set__pet_format

subroutine f90wrap_setupdt__get__pet_conversion_factor(this, f90wrap_pet_conversion_factor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_pet_conversion_factor
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_pet_conversion_factor = this_ptr%p%pet_conversion_factor
end subroutine f90wrap_setupdt__get__pet_conversion_factor

subroutine f90wrap_setupdt__set__pet_conversion_factor(this, f90wrap_pet_conversion_factor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_pet_conversion_factor
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%pet_conversion_factor = f90wrap_pet_conversion_factor
end subroutine f90wrap_setupdt__set__pet_conversion_factor

subroutine f90wrap_setupdt__get__pet_directory(this, f90wrap_pet_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_pet_directory
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_pet_directory = this_ptr%p%pet_directory
end subroutine f90wrap_setupdt__get__pet_directory

subroutine f90wrap_setupdt__set__pet_directory(this, f90wrap_pet_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_pet_directory
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%pet_directory = f90wrap_pet_directory
end subroutine f90wrap_setupdt__set__pet_directory

subroutine f90wrap_setupdt__get__daily_interannual_pet(this, f90wrap_daily_interannual_pet)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_daily_interannual_pet
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_daily_interannual_pet = this_ptr%p%daily_interannual_pet
end subroutine f90wrap_setupdt__get__daily_interannual_pet

subroutine f90wrap_setupdt__set__daily_interannual_pet(this, f90wrap_daily_interannual_pet)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_daily_interannual_pet
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%daily_interannual_pet = f90wrap_daily_interannual_pet
end subroutine f90wrap_setupdt__set__daily_interannual_pet

subroutine f90wrap_setupdt__get__mean_forcing(this, f90wrap_mean_forcing)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_mean_forcing
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_mean_forcing = this_ptr%p%mean_forcing
end subroutine f90wrap_setupdt__get__mean_forcing

subroutine f90wrap_setupdt__set__mean_forcing(this, f90wrap_mean_forcing)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_mean_forcing
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%mean_forcing = f90wrap_mean_forcing
end subroutine f90wrap_setupdt__set__mean_forcing

subroutine f90wrap_setupdt__get__prcp_indice(this, f90wrap_prcp_indice)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_prcp_indice
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_prcp_indice = this_ptr%p%prcp_indice
end subroutine f90wrap_setupdt__get__prcp_indice

subroutine f90wrap_setupdt__set__prcp_indice(this, f90wrap_prcp_indice)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_prcp_indice
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%prcp_indice = f90wrap_prcp_indice
end subroutine f90wrap_setupdt__set__prcp_indice

subroutine f90wrap_setupdt__get__read_descriptor(this, f90wrap_read_descriptor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_read_descriptor
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_read_descriptor = this_ptr%p%read_descriptor
end subroutine f90wrap_setupdt__get__read_descriptor

subroutine f90wrap_setupdt__set__read_descriptor(this, f90wrap_read_descriptor)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_read_descriptor
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%read_descriptor = f90wrap_read_descriptor
end subroutine f90wrap_setupdt__set__read_descriptor

subroutine f90wrap_setupdt__get__descriptor_format(this, f90wrap_descriptor_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_descriptor_format
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_descriptor_format = this_ptr%p%descriptor_format
end subroutine f90wrap_setupdt__get__descriptor_format

subroutine f90wrap_setupdt__set__descriptor_format(this, f90wrap_descriptor_format)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_descriptor_format
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%descriptor_format = f90wrap_descriptor_format
end subroutine f90wrap_setupdt__set__descriptor_format

subroutine f90wrap_setupdt__get__descriptor_directory(this, f90wrap_descriptor_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_descriptor_directory
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_descriptor_directory = this_ptr%p%descriptor_directory
end subroutine f90wrap_setupdt__get__descriptor_directory

subroutine f90wrap_setupdt__set__descriptor_directory(this, f90wrap_descriptor_directory)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_descriptor_directory
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%descriptor_directory = f90wrap_descriptor_directory
end subroutine f90wrap_setupdt__set__descriptor_directory

subroutine f90wrap_setupdt__array__descriptor_name(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 2
    dtype = 2
    this_ptr = transfer(this, this_ptr)
    if (allocated(this_ptr%p%descriptor_name)) then
        dshape(1:2) = (/len(this_ptr%p%descriptor_name(1)), shape(this_ptr%p%descriptor_name)/)
        dloc = loc(this_ptr%p%descriptor_name)
    else
        dloc = 0
    end if
end subroutine f90wrap_setupdt__array__descriptor_name

subroutine f90wrap_setupdt__get__interception_module(this, f90wrap_interception_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_interception_module
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_interception_module = this_ptr%p%interception_module
end subroutine f90wrap_setupdt__get__interception_module

subroutine f90wrap_setupdt__set__interception_module(this, f90wrap_interception_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_interception_module
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%interception_module = f90wrap_interception_module
end subroutine f90wrap_setupdt__set__interception_module

subroutine f90wrap_setupdt__get__production_module(this, f90wrap_production_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_production_module
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_production_module = this_ptr%p%production_module
end subroutine f90wrap_setupdt__get__production_module

subroutine f90wrap_setupdt__set__production_module(this, f90wrap_production_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_production_module
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%production_module = f90wrap_production_module
end subroutine f90wrap_setupdt__set__production_module

subroutine f90wrap_setupdt__get__transfer_module(this, f90wrap_transfer_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_transfer_module
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_transfer_module = this_ptr%p%transfer_module
end subroutine f90wrap_setupdt__get__transfer_module

subroutine f90wrap_setupdt__set__transfer_module(this, f90wrap_transfer_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_transfer_module
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%transfer_module = f90wrap_transfer_module
end subroutine f90wrap_setupdt__set__transfer_module

subroutine f90wrap_setupdt__get__exchange_module(this, f90wrap_exchange_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_exchange_module
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_exchange_module = this_ptr%p%exchange_module
end subroutine f90wrap_setupdt__get__exchange_module

subroutine f90wrap_setupdt__set__exchange_module(this, f90wrap_exchange_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_exchange_module
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%exchange_module = f90wrap_exchange_module
end subroutine f90wrap_setupdt__set__exchange_module

subroutine f90wrap_setupdt__get__routing_module(this, f90wrap_routing_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_routing_module
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_routing_module = this_ptr%p%routing_module
end subroutine f90wrap_setupdt__get__routing_module

subroutine f90wrap_setupdt__set__routing_module(this, f90wrap_routing_module)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_routing_module
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%routing_module = f90wrap_routing_module
end subroutine f90wrap_setupdt__set__routing_module

subroutine f90wrap_setupdt__get__save_qsim_domain(this, f90wrap_save_qsim_domain)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_save_qsim_domain
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_save_qsim_domain = this_ptr%p%save_qsim_domain
end subroutine f90wrap_setupdt__get__save_qsim_domain

subroutine f90wrap_setupdt__set__save_qsim_domain(this, f90wrap_save_qsim_domain)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_save_qsim_domain
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%save_qsim_domain = f90wrap_save_qsim_domain
end subroutine f90wrap_setupdt__set__save_qsim_domain

subroutine f90wrap_setupdt__get__save_net_prcp_domain(this, f90wrap_save_net_prcp_domain)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_save_net_prcp_domain
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_save_net_prcp_domain = this_ptr%p%save_net_prcp_domain
end subroutine f90wrap_setupdt__get__save_net_prcp_domain

subroutine f90wrap_setupdt__set__save_net_prcp_domain(this, f90wrap_save_net_prcp_domain)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_save_net_prcp_domain
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%save_net_prcp_domain = f90wrap_save_net_prcp_domain
end subroutine f90wrap_setupdt__set__save_net_prcp_domain

subroutine f90wrap_setupdt__get__ntime_step(this, f90wrap_ntime_step)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_ntime_step
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_ntime_step = this_ptr%p%ntime_step
end subroutine f90wrap_setupdt__get__ntime_step

subroutine f90wrap_setupdt__set__ntime_step(this, f90wrap_ntime_step)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_ntime_step
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%ntime_step = f90wrap_ntime_step
end subroutine f90wrap_setupdt__set__ntime_step

subroutine f90wrap_setupdt__get__nd(this, f90wrap_nd)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nd
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nd = this_ptr%p%nd
end subroutine f90wrap_setupdt__get__nd

subroutine f90wrap_setupdt__set__nd(this, f90wrap_nd)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nd
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nd = f90wrap_nd
end subroutine f90wrap_setupdt__set__nd

subroutine f90wrap_setupdt__get__algorithm(this, f90wrap_algorithm)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_algorithm
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_algorithm = this_ptr%p%algorithm
end subroutine f90wrap_setupdt__get__algorithm

subroutine f90wrap_setupdt__set__algorithm(this, f90wrap_algorithm)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_algorithm
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%algorithm = f90wrap_algorithm
end subroutine f90wrap_setupdt__set__algorithm

subroutine f90wrap_setupdt__get__jobs_fun(this, f90wrap_jobs_fun)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_jobs_fun
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_jobs_fun = this_ptr%p%jobs_fun
end subroutine f90wrap_setupdt__get__jobs_fun

subroutine f90wrap_setupdt__set__jobs_fun(this, f90wrap_jobs_fun)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_jobs_fun
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%jobs_fun = f90wrap_jobs_fun
end subroutine f90wrap_setupdt__set__jobs_fun

subroutine f90wrap_setupdt__get__jreg_fun(this, f90wrap_jreg_fun)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_jreg_fun
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_jreg_fun = this_ptr%p%jreg_fun
end subroutine f90wrap_setupdt__get__jreg_fun

subroutine f90wrap_setupdt__set__jreg_fun(this, f90wrap_jreg_fun)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_jreg_fun
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%jreg_fun = f90wrap_jreg_fun
end subroutine f90wrap_setupdt__set__jreg_fun

subroutine f90wrap_setupdt__get__wjreg(this, f90wrap_wjreg)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(out) :: f90wrap_wjreg
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_wjreg = this_ptr%p%wjreg
end subroutine f90wrap_setupdt__get__wjreg

subroutine f90wrap_setupdt__set__wjreg(this, f90wrap_wjreg)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(4), intent(in) :: f90wrap_wjreg
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%wjreg = f90wrap_wjreg
end subroutine f90wrap_setupdt__set__wjreg

subroutine f90wrap_setupdt__get__optim_start_step(this, f90wrap_optim_start_step)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_optim_start_step
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_optim_start_step = this_ptr%p%optim_start_step
end subroutine f90wrap_setupdt__get__optim_start_step

subroutine f90wrap_setupdt__set__optim_start_step(this, f90wrap_optim_start_step)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_optim_start_step
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%optim_start_step = f90wrap_optim_start_step
end subroutine f90wrap_setupdt__set__optim_start_step

subroutine f90wrap_setupdt__array__optim_parameters(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%optim_parameters)
    dloc = loc(this_ptr%p%optim_parameters)
end subroutine f90wrap_setupdt__array__optim_parameters

subroutine f90wrap_setupdt__array__optim_states(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 5
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%optim_states)
    dloc = loc(this_ptr%p%optim_states)
end subroutine f90wrap_setupdt__array__optim_states

subroutine f90wrap_setupdt__array__lb_parameters(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%lb_parameters)
    dloc = loc(this_ptr%p%lb_parameters)
end subroutine f90wrap_setupdt__array__lb_parameters

subroutine f90wrap_setupdt__array__ub_parameters(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%ub_parameters)
    dloc = loc(this_ptr%p%ub_parameters)
end subroutine f90wrap_setupdt__array__ub_parameters

subroutine f90wrap_setupdt__array__lb_states(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%lb_states)
    dloc = loc(this_ptr%p%lb_states)
end subroutine f90wrap_setupdt__array__lb_states

subroutine f90wrap_setupdt__array__ub_states(this, nd, dtype, dshape, dloc)
    use mwd_setup, only: setupdt
    use, intrinsic :: iso_c_binding, only : c_int
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer(c_int), intent(in) :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer(c_int), intent(out) :: nd
    integer(c_int), intent(out) :: dtype
    integer(c_int), dimension(10), intent(out) :: dshape
    integer*8, intent(out) :: dloc
    
    nd = 1
    dtype = 11
    this_ptr = transfer(this, this_ptr)
    dshape(1:1) = shape(this_ptr%p%ub_states)
    dloc = loc(this_ptr%p%ub_states)
end subroutine f90wrap_setupdt__array__ub_states

subroutine f90wrap_setupdt__get__maxiter(this, f90wrap_maxiter)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_maxiter
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_maxiter = this_ptr%p%maxiter
end subroutine f90wrap_setupdt__get__maxiter

subroutine f90wrap_setupdt__set__maxiter(this, f90wrap_maxiter)
    use mwd_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_maxiter
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%maxiter = f90wrap_maxiter
end subroutine f90wrap_setupdt__set__maxiter

subroutine f90wrap_setupdt_initialise(setup, nd)
    use mwd_setup, only: setupdt, setupdt_initialise
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(out), dimension(2) :: setup
    integer, intent(in) :: nd
    allocate(setup_ptr%p)
    call setupdt_initialise(setup=setup_ptr%p, nd=nd)
    setup = transfer(setup_ptr, setup)
end subroutine f90wrap_setupdt_initialise

subroutine f90wrap_setupdt_finalise(this)
    use mwd_setup, only: setupdt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_setupdt_finalise

! End of module mwd_setup defined in file smash/solver/module/mwd_setup.f90

