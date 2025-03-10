!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ControlDT
!%          Control vector used in optimize and quantities required by the optimizer
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``x``                      Control vector
!%          ``l``                      Control vector lower bound
!%          ``u``                      Control vector upper bound
!%          ``x_raw``                  Control vector raw
!%          ``l_raw``                  Control vector lower bound raw
!%          ``u_raw``                  Control vector upper bound raw
!%          ``nbd``                    Control vector kind of bound
!%
!§      Subroutine
!%      ----------
!%
!%      - ControlDT_initialise
!%      - ControlDT_copy

module mwd_control

    use md_constant !% only: sp

    implicit none

    type ControlDT

        integer :: n
        ! Kinds: rr_parameters, rr_initial_states, serr_mu_parameters, serr_sigma_parameters, nn_parameters
        integer, dimension(5) :: nbk
        real(sp), dimension(:), allocatable :: x
        real(sp), dimension(:), allocatable :: l
        real(sp), dimension(:), allocatable :: u

        real(sp), dimension(:), allocatable :: x_raw
        real(sp), dimension(:), allocatable :: l_raw
        real(sp), dimension(:), allocatable :: u_raw

        integer, dimension(:), allocatable :: nbd
        character(lchar), dimension(:), allocatable :: name !$F90W char-array

    end type ControlDT

contains

    subroutine ControlDT_initialise(this, nbk)

        implicit none

        type(ControlDT), intent(inout) :: this
        integer, dimension(size(this%nbk)), intent(in) :: nbk

        call ControlDT_finalise(this)

        this%nbk = nbk
        this%n = sum(this%nbk)

        allocate (this%x(this%n))
        this%x = -99._sp

        allocate (this%l(this%n))
        this%l = -99._sp

        allocate (this%u(this%n))
        this%u = -99._sp

        allocate (this%x_raw(this%n))
        this%x_raw = 0._sp

        allocate (this%l_raw(this%n))
        this%l_raw = -99._sp

        allocate (this%u_raw(this%n))
        this%u_raw = -99._sp

        allocate (this%nbd(this%n))
        this%nbd = -99

        allocate (this%name(this%n))
        this%name = "..."

    end subroutine ControlDT_initialise

    subroutine ControlDT_finalise(this)

        implicit none

        type(ControlDT), intent(inout) :: this

        if (allocated(this%x)) then

            deallocate (this%x)

            deallocate (this%l)

            deallocate (this%u)

            deallocate (this%x_raw)

            deallocate (this%l_raw)

            deallocate (this%u_raw)

            deallocate (this%nbd)

            deallocate (this%name)

        end if

    end subroutine ControlDT_finalise

    subroutine ControlDT_copy(this, this_copy)

        implicit none

        type(ControlDT), intent(in) :: this
        type(ControlDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ControlDT_copy

    ! To manually deallocate from Python. ControlDT_finalise is used as
    ! __del__ method for garbage collecting (implemented by f90wrap automatically)
    subroutine ControlDT_dealloc(this)

        implicit none

        type(ControlDT), intent(inout) :: this

        call ControlDT_finalise(this)

    end subroutine ControlDT_dealloc

end module mwd_control
