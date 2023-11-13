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
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

contains

    subroutine upstream_discharge(row, col, flwdir2d, q2d, qup)

        implicit none

        integer, intent(in) :: row, col
        integer, dimension(:, :), intent(in) :: flwdir2d
        real(sp), dimension(:, :), intent(in) :: q2d
        real(sp), intent(out) :: qup

        integer :: nrow, ncol, i, row_imd, col_imd
        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)

        nrow = size(flwdir2d, 1)
        ncol = size(flwdir2d, 2)

        qup = 0._sp

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

            if (flwdir2d(row_imd, col_imd) .eq. i) qup = qup + q2d(row_imd, col_imd)

        end do

    end subroutine upstream_discharge

    subroutine linear_routing(dx, dy, dt, flwacc, llr, hlr, qup, q)

        implicit none

        real(sp), intent(in) :: dx, dy, dt, flwacc
        real(sp), intent(in) :: llr
        real(sp), intent(inout) :: hlr, qup, q

        real(sp) :: hlr_imd

        qup = (qup*dt)/(1e-3_sp*(flwacc - dx*dy))

        hlr_imd = hlr + qup

        hlr = hlr_imd*exp(-dt/(llr*60._sp))

        q = q + (hlr_imd - hlr)*1e-3_sp*(flwacc - dx*dy)/dt

    end subroutine linear_routing

    subroutine kinematic_wave1d(dx, dy, dt, akw, bkw, qlijm1, qlij, qim1j, qijm1, qij)

        implicit none

        real(sp), intent(in) :: dx, dy, dt
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

    subroutine lag0_timestep(setup, mesh, qt, q)

        implicit none

        integer, parameter :: zq = 1

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(in) :: qt
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(inout) :: q

        integer :: i, row, col
        real(sp) :: qup

        q(:, :, zq) = qt(:, :, zq)

        ! TODO: replace this loop with loop on routing partitions
        do i = 1, mesh%nrow*mesh%ncol

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

            if (mesh%flwacc(row, col) .le. mesh%dx(row, col)*mesh%dy(row, col)) cycle

            call upstream_discharge(row, col, mesh%flwdir, q(:, :, zq), qup)

            q(row, col, zq) = q(row, col, zq) + qup

        end do

    end subroutine lag0_timestep

    subroutine lr_timestep(setup, mesh, qt, llr, hlr, q)

        implicit none

        integer, parameter :: zq = 1

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(in) :: qt
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: llr
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: hlr
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(inout) :: q

        integer :: i, row, col
        real(sp) :: qup

        q(:, :, zq) = qt(:, :, zq)

        ! TODO: replace this loop with loop on routing partitions
        do i = 1, mesh%nrow*mesh%ncol

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

            if (mesh%flwacc(row, col) .le. mesh%dx(row, col)*mesh%dy(row, col)) cycle

            call upstream_discharge(row, col, mesh%flwdir, q(:, :, zq), qup)

            call linear_routing(mesh%dx(row, col), mesh%dy(row, col), setup%dt, mesh%flwacc(row, col), &
            & llr(row, col), hlr(row, col), qup, q(row, col, zq))

        end do

    end subroutine lr_timestep

    subroutine kw_timestep(setup, mesh, qt, akw, bkw, q)

        implicit none

        integer, parameter :: zq = 2

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(in) :: qt
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: akw, bkw
        real(sp), dimension(mesh%nrow, mesh%ncol, zq), intent(inout) :: q

        integer :: i, row, col
        real(sp) :: qlijm1, qlij, qim1j, qijm1

        q(:, :, zq) = qt(:, :, zq)

        ! TODO: replace this loop with loop on routing partitions
        do i = 1, mesh%nrow*mesh%ncol

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

            if (mesh%flwacc(row, col) .le. mesh%dx(row, col)*mesh%dy(row, col)) cycle

            qlijm1 = qt(row, col, zq - 1)
            qlij = qt(row, col, zq)
            qijm1 = q(row, col, zq - 1)

            call upstream_discharge(row, col, mesh%flwdir, q(:, :, zq), qim1j)

            call kinematic_wave1d(mesh%dx(row, col), mesh%dy(row, col), setup%dt, &
            & akw(row, col), bkw(row, col), qlijm1, qlij, qim1j, qijm1, q(row, col, zq))

        end do

    end subroutine kw_timestep

end module md_routing_operator
