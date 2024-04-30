!%      (M) Module.
!%
!%      Subroutine
!%      ----------
!%
!%      - display_iteration_progress

module m_screen_display

    use md_constant, only: lchar

    implicit none

contains

    subroutine display_iteration_progress(iter, niter, task)

        implicit none

        integer, intent(in) :: iter, niter
        character(lchar), intent(in) :: task

        integer :: per

        per = 100*iter/niter

        if (per .ne. 100*(iter - 1)/niter) then

            write (*, "(a,4x,a,1x,i0,a,i0,1x,a,i0,a)", advance="no") achar(13), trim(task), iter, "/", niter, "(", per, "%)"

        end if

        if (iter == niter) write (*, "(a)") ""

    end subroutine display_iteration_progress

end module m_screen_display
