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
!%          ``x_bkg``                  Control vector background
!%          ``l_bkg``                  Control vector lower bound background
!%          ``u_bkg``                  Control vector upper bound background
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
        ! Four kinds of parameters (rr_parameters, rr_initial_states, serr_mu_parameters, serr_sigma_parameters)
        integer, dimension(4) :: nbk
        real(sp), dimension(:), allocatable :: x
        real(sp), dimension(:), allocatable :: l
        real(sp), dimension(:), allocatable :: u

        real(sp), dimension(:), allocatable :: x_bkg
        real(sp), dimension(:), allocatable :: l_bkg
        real(sp), dimension(:), allocatable :: u_bkg

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

        allocate (this%x_bkg(this%n))
        this%x_bkg = 0._sp

        allocate (this%l_bkg(this%n))
        this%l_bkg = -99._sp

        allocate (this%u_bkg(this%n))
        this%u_bkg = -99._sp

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

            deallocate (this%x_bkg)

            deallocate (this%l_bkg)

            deallocate (this%u_bkg)

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
