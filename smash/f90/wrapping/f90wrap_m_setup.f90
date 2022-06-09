! Module m_setup defined in file smash/f90/wrapped_module/m_setup.f90

subroutine f90wrap_setupdt__get__dt(this, f90wrap_dt)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(8), intent(out) :: f90wrap_dt
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_dt = this_ptr%p%dt
end subroutine f90wrap_setupdt__get__dt

subroutine f90wrap_setupdt__set__dt(this, f90wrap_dt)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(8), intent(in) :: f90wrap_dt
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%dt = f90wrap_dt
end subroutine f90wrap_setupdt__set__dt

subroutine f90wrap_setupdt__get__dx(this, f90wrap_dx)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(8), intent(out) :: f90wrap_dx
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_dx = this_ptr%p%dx
end subroutine f90wrap_setupdt__get__dx

subroutine f90wrap_setupdt__set__dx(this, f90wrap_dx)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    real(8), intent(in) :: f90wrap_dx
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%dx = f90wrap_dx
end subroutine f90wrap_setupdt__set__dx

subroutine f90wrap_setupdt__get__start_time(this, f90wrap_start_time)
    use m_setup, only: setupdt
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
    use m_setup, only: setupdt
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
    use m_setup, only: setupdt
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
    use m_setup, only: setupdt
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

subroutine f90wrap_setupdt__get__optim_start_time(this, f90wrap_optim_start_time)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(out) :: f90wrap_optim_start_time
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_optim_start_time = this_ptr%p%optim_start_time
end subroutine f90wrap_setupdt__get__optim_start_time

subroutine f90wrap_setupdt__set__optim_start_time(this, f90wrap_optim_start_time)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    character(1024), intent(in) :: f90wrap_optim_start_time
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%optim_start_time = f90wrap_optim_start_time
end subroutine f90wrap_setupdt__set__optim_start_time

subroutine f90wrap_setupdt__get__nb_time_step(this, f90wrap_nb_time_step)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_nb_time_step
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_nb_time_step = this_ptr%p%nb_time_step
end subroutine f90wrap_setupdt__get__nb_time_step

subroutine f90wrap_setupdt__set__nb_time_step(this, f90wrap_nb_time_step)
    use m_setup, only: setupdt
    implicit none
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    integer, intent(in)   :: this(2)
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_nb_time_step
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%nb_time_step = f90wrap_nb_time_step
end subroutine f90wrap_setupdt__set__nb_time_step

subroutine f90wrap_setupdt__get__optim_start_step(this, f90wrap_optim_start_step)
    use m_setup, only: setupdt
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
    use m_setup, only: setupdt
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

subroutine f90wrap_setupdt_initialise(this)
    use m_setup, only: setupdt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(out), dimension(2) :: this
    allocate(this_ptr%p)
    this = transfer(this_ptr, this)
end subroutine f90wrap_setupdt_initialise

subroutine f90wrap_setupdt_finalise(this)
    use m_setup, only: setupdt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(setupdt_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_setupdt_finalise

! End of module m_setup defined in file smash/f90/wrapped_module/m_setup.f90

