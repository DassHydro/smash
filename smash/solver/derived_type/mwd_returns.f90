!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      gr_a_forward

module mwd_returns

    use md_constant !% only: sp

    implicit none

    type ReturnsDT

        real(sp) :: cost

    end type ReturnsDT

contains

    subroutine ReturnsDT_initialise(this)

        implicit none

        type(ReturnsDT), intent(inout) :: this

    end subroutine ReturnsDT_initialise

    subroutine ReturnsDT_copy(this, this_copy)

        implicit none

        type(ReturnsDT), intent(in) :: this
        type(ReturnsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ReturnsDT_copy

end module mwd_returns
