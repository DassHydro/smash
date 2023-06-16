!%      (M) Module.
!%
!%      Interface
!%      ---------
!%
!%      - arange
!%          - arange_i
!%          - arange_r
!%
!%      - linspace
!%          - linspace_i
!%          - linspace_r
!%
!%      Subroutine
!%      ----------
!%
!%      - arange_i
!%      - arange_r
!%      - linspace_i
!%      - linspace_r

module m_array_creation

    use md_constant, only: sp

    implicit none

    interface arange

        module procedure arange_i
        module procedure arange_r

    end interface arange

    interface linspace

        module procedure linspace_i
        module procedure linspace_r

    end interface linspace

contains

    subroutine arange_i(stt, stp, step, res)

        integer, intent(in) :: stt, stp
        real(sp), intent(in) :: step
        real(sp), dimension(ceiling(real(stp - stt, kind=sp)/step)), intent(inout) :: res

        integer :: i

        do i = 1, ceiling(real(stp - stt, kind=sp)/step)

            res(i) = stt + (i - 1)*step

        end do

    end subroutine arange_i

    subroutine arange_r(stt, stp, step, res)

        real(sp), intent(in) :: stt, stp, step
        real(sp), dimension(ceiling((stp - stt)/step)), intent(inout) :: res

        integer :: i

        do i = 1, ceiling((stp - stt)/step)

            res(i) = stt + (i - 1)*step

        end do

    end subroutine arange_r

    subroutine linspace_i(stt, stp, n, res)

        integer, intent(in) :: stt, stp, n
        real(sp), dimension(n), intent(inout) :: res

        real(sp) :: step
        integer :: i

        if (n .eq. 1) then

            res(1) = stt

        else

            step = real(stp - stt, kind=sp)/real(n - 1, kind=sp)

            do i = 1, n

                res(i) = stt + (i - 1)*step

            end do

        end if

    end subroutine linspace_i

    subroutine linspace_r(stt, stp, n, res)

        real(sp), intent(in) :: stt, stp
        integer, intent(in) :: n
        real(sp), dimension(n), intent(inout) :: res

        real(sp) :: step
        integer :: i

        if (n .eq. 1) then

            res(1) = stt

        else

            step = (stp - stt)/real(n - 1, kind=sp)

            do i = 1, n

                res(i) = stt + (i - 1)*step

            end do

        end if

    end subroutine linspace_r

end module m_array_creation
