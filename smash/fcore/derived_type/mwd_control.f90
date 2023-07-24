!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ControlDT
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

        real(sp), dimension(:), allocatable :: x
        real(sp), dimension(:), allocatable :: l
        real(sp), dimension(:), allocatable :: u

!~         real(sp), dimension(:), allocatable :: x_bkg
        real(sp), dimension(:), allocatable :: l_bkg
        real(sp), dimension(:), allocatable :: u_bkg

        integer, dimension(:), allocatable :: nbd

    end type ControlDT

contains

    subroutine ControlDT_initialise(this, n)

        implicit none

        type(ControlDT), intent(inout) :: this
        integer, intent(in) :: n

        allocate (this%x(n))
        this%x = -99._sp

!~         allocate (this%x_bkg(n))
!~         this%x_bkg = 0._sp

        allocate (this%l(n))
        this%l = -99._sp

        allocate (this%l_bkg(n))
        this%l_bkg = -99._sp

        allocate (this%u(n))
        this%u = -99._sp

        allocate (this%u_bkg(n))
        this%u_bkg = -99._sp

        allocate (this%nbd(n))
        this%nbd = -99

    end subroutine ControlDT_initialise

    subroutine ControlDT_finalise(this)

        implicit none

        type(ControlDT), intent(inout) :: this

        if (allocated(this%x)) then

            deallocate (this%x)

!~         deallocate (this%x_bkg(n))

            deallocate (this%l)

            deallocate (this%l_bkg)

            deallocate (this%u)

            deallocate (this%u_bkg)

            deallocate (this%nbd)

        end if

    end subroutine ControlDT_finalise

    subroutine ControlDT_copy(this, this_copy)

        implicit none

        type(ControlDT), intent(in) :: this
        type(ControlDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ControlDT_copy

end module mwd_control
