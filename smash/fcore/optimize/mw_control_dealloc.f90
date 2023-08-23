!%      (MW) Module Wrapped.
!%
!§      Subroutine
!%      ----------
!%
!%      - dealloc_control

module mw_control_dealloc

    use mwd_control, only: ControlDT

    implicit none

contains

    subroutine dealloc_control(this)

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


    end subroutine dealloc_control

end module mw_control_dealloc
