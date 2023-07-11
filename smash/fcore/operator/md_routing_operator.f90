!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - upstream_discharge
!%      - linear_routing
!%      - kinematic_wave1d

module md_routing_operator

    use md_constant !% only : sp

    implicit none

contains

    subroutine upstream_discharge(nrow, ncol, row, col, dx, dy, &
    & fa, flwdir, q, qup)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, intent(in) :: row, col
        real(sp), intent(in) :: dx, dy, fa
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(sp), dimension(nrow, ncol), intent(in) :: q
        real(sp), intent(out) :: qup

        integer :: i, row_imd, col_imd
        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)

        qup = 0._sp

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

            if (flwdir(row_imd, col_imd) .eq. i) qup = qup + q(row_imd, col_imd)

        end do

    end subroutine upstream_discharge

    subroutine linear_routing(dt, dx, dy, fa, llr, hlr, qup, qrout)

        implicit none

        real(sp), intent(in) :: dt, dx, dy, fa
        real(sp), intent(in) :: llr
        real(sp), intent(inout) :: hlr, qup
        real(sp), intent(out) :: qrout

        real(sp) :: hlr_imd

        qup = (qup*dt)/(1e-3_sp*(fa - dx*dy))

        hlr_imd = hlr + qup

        hlr = hlr_imd*exp(-dt/(llr*60._sp))

        qrout = hlr_imd - hlr

        qrout = qrout*1e-3_sp*(fa - dx*dy)/dt

    end subroutine linear_routing

    subroutine kinematic_wave1d(dt, dx, akw, bkw, qlijm1, qlij, qim1j, qijm1, qij)

        implicit none

        real(sp), intent(in) :: dt, dx
        real(sp), intent(in) :: akw, bkw
        real(sp), intent(in) :: qlijm1, qlij, qim1j, qijm1
        real(sp), intent(inout) :: qij

        real(sp) :: wqlijm1, wqlij, wqim1j, wqijm1
        real(sp) :: dtddx, n1, n2, n3, d1, d2, rhs, rsd, rsd_d
        integer :: iter, maxiter

        !% Avoid numerical issues
        wqlijm1 = max(1e-6_sp, qlijm1)
        wqlij = max(1e-6_sp, qlij)
        wqim1j = max(1e-6_sp, qim1j)
        wqijm1 = max(1e-6_sp, qijm1)

        dtddx = dt/dx

        d1 = dtddx
        d2 = akw*bkw*((wqijm1 + wqim1j)/2._sp)**(bkw - 1._sp)

        n1 = dtddx*wqim1j
        n2 = wqijm1*d2
        n3 = dtddx*(wqlijm1 + wqlij)/2._sp

        !% Linearized solution
        qij = (n1 + n2 + n3)/(d1 + d2)

        !% Non-Linear solution solved with Newton-Raphson
        !% Commented while testing Linearized solution

!~         rhs = n1 + akw*wqijm1**bkw + n3

!~         iter = 0
!~         maxiter = 2
!~         rsd = 1._sp

!~         do while (abs(rsd) > 1e-6 .and. iter < maxiter)

!~             rsd = dtddx*qij + akw*qij**bkw - rhs
!~             rsd_d = dtddx + akw*bkw*qij**(bkw - 1._sp)

!~             qij = qij - rsd/rsd_d

!~             qij = max(qij, 0._sp)

!~             iter = iter + 1

!~         end do

    end subroutine kinematic_wave1d

end module md_routing_operator
