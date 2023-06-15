!%      (M) Module.
!%
!%      Interface
!%      ---------
!%
!%      - arange_dev
!%          - arange_i_dev
!%          - arange_r_dev
!%
!%      - linspace_dev
!%          - linspace_i_dev
!%          - linspace_r_dev
!%
!%      Subroutine
!%      ----------
!%
!%      - arange_i_dev
!%      - arange_r_dev
!%      - linspace_i_dev
!%      - linspace_r_dev

module m_array_creation_dev

    use md_constant_dev, only: sp

    implicit none

    interface arange_dev

        module procedure arange_i_dev
        module procedure arange_r_dev

    end interface arange_dev

    interface linspace_dev

        module procedure linspace_i_dev
        module procedure linspace_r_dev

    end interface linspace_dev

contains

    subroutine arange_i_dev(stt, stp, step, res)

        integer, intent(in) :: stt, stp
        real(sp), intent(in) :: step
        real(sp), dimension(ceiling(real(stp - stt, kind=sp)/step)), intent(inout) :: res

        integer :: i

        do i = 1, ceiling(real(stp - stt, kind=sp)/step)

            res(i) = stt + (i - 1)*step

        end do

    end subroutine arange_i_dev

    subroutine arange_r_dev(stt, stp, step, res)

        real(sp), intent(in) :: stt, stp, step
        real(sp), dimension(ceiling((stp - stt)/step)), intent(inout) :: res

        integer :: i

        do i = 1, ceiling((stp - stt)/step)

            res(i) = stt + (i - 1)*step

        end do

    end subroutine arange_r_dev

    subroutine linspace_i_dev(stt, stp, n, res)

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

    end subroutine linspace_i_dev

    subroutine linspace_r_dev(stt, stp, n, res)

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

    end subroutine linspace_r_dev

end module m_array_creation_dev
