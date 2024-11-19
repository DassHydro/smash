module mw_mesh

    implicit none

    real(4), parameter :: TWOPI = 6.2831853_4
    real(4), parameter :: DEGREE_TO_RADIAN = TWOPI/360._4
    real(4), parameter :: RADIUS_EARTH = 6371228._4
    real(4), parameter :: DEGREE_TO_METER = RADIUS_EARTH*DEGREE_TO_RADIAN

contains

    subroutine latlon_dxdy(nrow, ncol, xres, yres, ymax, dx, dy)

        implicit none

        integer, intent(in) :: nrow, ncol
        real(4), intent(in) :: xres, yres, ymax
        real(4), dimension(nrow, ncol), intent(out) :: dx, dy

        real(4) :: lat
        integer :: row

        dy = yres*DEGREE_TO_METER

        lat = ymax + (yres*0.5_4)

        do row = 1, nrow

            lat = lat - yres

            dx(row, :) = (xres*cos(lat*DEGREE_TO_RADIAN)*DEGREE_TO_METER)

        end do

    end subroutine latlon_dxdy

    recursive subroutine mask_upstream_cells(nrow, ncol, flwdir, row, col, mask, sink)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, intent(in) :: row, col
        integer, dimension(nrow, ncol), intent(inout) :: mask
        logical, intent(inout) :: sink

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)
        integer :: i, row_imd, col_imd

        mask(row, col) = 1

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            ! % Bounds
            if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

            ! % Non upstream cell
            if (flwdir(row_imd, col_imd) .ne. i) cycle

            ! % Upstream sink cell, must return
            if (mask(row_imd, col_imd) .eq. 1) then
                sink = .true.
                return
            end if

            call mask_upstream_cells(nrow, ncol, flwdir, row_imd, col_imd, mask, sink)

        end do

    end subroutine mask_upstream_cells

    subroutine catchment_dln_area_based(nrow, ncol, flwdir, dx, dy, row, col, area, &
    & max_depth, mask_dln, row_dln, col_dln, sink_dln)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(4), dimension(nrow, ncol), intent(in) :: dx, dy
        integer, intent(inout) :: row, col
        real(4), intent(in) :: area
        integer, intent(in) :: max_depth
        integer, dimension(nrow, ncol), intent(out) :: mask_dln
        integer, intent(out) :: col_dln, row_dln
        logical, intent(out) :: sink_dln

        integer, dimension(nrow, ncol) :: mask_dln_imd
        integer :: i, j, row_imd, col_imd
        real(4) :: min_tol, tol
        logical :: sink_dln_imd

        !% Transform from Python to FORTRAN index
        row = row + 1
        col = col + 1

        min_tol = huge(0._4)

        do i = -max_depth, max_depth

            do j = -max_depth, max_depth

                row_imd = row + j
                col_imd = col + i
                mask_dln_imd = 0
                sink_dln_imd = .false.

                if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

                call mask_upstream_cells(nrow, ncol, flwdir, row_imd, &
                & col_imd, mask_dln_imd, sink_dln_imd)

                if (sink_dln_imd) then
                    sink_dln = .true.
                    cycle
                end if

                !% Compare "observed" area (area) with the "computed" area (sum(mask_dln_imd*dx*dy))
                tol = abs(area - sum(mask_dln_imd*dx*dy))/area

                if (tol .ge. min_tol) cycle

                min_tol = tol

                !% Transform from FORTRAN to Python index
                row_dln = row_imd - 1
                col_dln = col_imd - 1

                mask_dln = mask_dln_imd

            end do

        end do

        !% Transform from FORTRAN to Python index
        row = row - 1
        col = col - 1

    end subroutine catchment_dln_area_based

    subroutine catchment_dln_contour_based(nrow, ncol, flwdir, mask, row, col, &
    & max_depth, mask_dln, row_dln, col_dln, sink_dln)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir, mask
        integer, intent(inout) :: row, col
        integer, intent(in) :: max_depth
        integer, dimension(nrow, ncol), intent(out) :: mask_dln
        integer, intent(out) :: col_dln, row_dln
        logical, intent(out) :: sink_dln

        integer, dimension(nrow, ncol) :: mask_dln_imd
        integer :: i, j, row_imd, col_imd
        real(4) :: min_tol, tol
        logical :: sink_dln_imd

        !% Transform from Python to FORTRAN index
        row = row + 1
        col = col + 1

        min_tol = huge(0._4)

        do i = -max_depth, max_depth

            do j = -max_depth, max_depth

                row_imd = row + j
                col_imd = col + i
                mask_dln_imd = 0
                sink_dln_imd = .false.

                if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

                call mask_upstream_cells(nrow, ncol, flwdir, row_imd, &
                & col_imd, mask_dln_imd, sink_dln_imd)

                if (sink_dln_imd) then
                    sink_dln = .true.
                    cycle
                end if

                !% Compare "observed" contour (mask) with the "computed" contour (mask_dln_imd)
                tol = sum(abs(mask - mask_dln_imd))

                if (tol .ge. min_tol) cycle

                min_tol = tol

                !% Transform from FORTRAN to Python index
                row_dln = row_imd - 1
                col_dln = col_imd - 1

                mask_dln = mask_dln_imd

            end do

        end do

        !% Transform from FORTRAN to Python index
        row = row - 1
        col = col - 1

    end subroutine catchment_dln_contour_based

    subroutine fill_nidp(nrow, ncol, flwdir, nidp)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(inout) :: nidp

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)

        integer :: row, col, row_imd, col_imd, i

        nidp = 0

        do col = 1, ncol

            do row = 1, nrow

                do i = 1, 8

                    row_imd = row + drow(i)
                    col_imd = col + dcol(i)

                    if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

                    if (flwdir(row_imd, col_imd) .eq. i) nidp(row, col) = nidp(row, col) + 1

                end do

            end do

        end do

    end subroutine fill_nidp

    recursive subroutine acc_par_downstream_cell(nrow, ncol, flwdir, nidp, row, col, flwacc, flwpar)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(inout) :: nidp
        integer, intent(in) :: row, col
        real(4), dimension(nrow, ncol), intent(inout) :: flwacc
        integer, dimension(nrow, ncol), intent(inout) :: flwpar

        integer :: fd, row_imd, col_imd
        integer, dimension(8) :: drow = (/-1, -1, 0, 1, 1, 1, 0, -1/)
        integer, dimension(8) :: dcol = (/0, 1, 1, 1, 0, -1, -1, -1/)

        fd = flwdir(row, col)

        !% Check no data
        if (fd .le. 0) return

        row_imd = row + drow(fd)
        col_imd = col + dcol(fd)

        !% Check bounds
        if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) return

        !% Check pit
        if (abs(fd - flwdir(row_imd, col_imd)) .eq. 4) return

        flwacc(row_imd, col_imd) = flwacc(row_imd, col_imd) + flwacc(row, col)
        flwpar(row_imd, col_imd) = max(flwpar(row_imd, col_imd), flwpar(row, col) + 1)

        if (nidp(row_imd, col_imd) .gt. 1) then

            nidp(row_imd, col_imd) = nidp(row_imd, col_imd) - 1

        else

            call acc_par_downstream_cell(nrow, ncol, flwdir, nidp, row_imd, col_imd, flwacc, flwpar)

        end if

    end subroutine acc_par_downstream_cell

    subroutine flow_accumulation_partition(nrow, ncol, flwdir, dx, dy, flwacc, flwpar)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in):: flwdir
        real(4), dimension(nrow, ncol), intent(in) :: dx, dy
        real(4), dimension(nrow, ncol), intent(out) :: flwacc
        integer, dimension(nrow, ncol), intent(out) :: flwpar

        integer :: row, col
        integer, dimension(nrow, ncol) :: nidp

        call fill_nidp(nrow, ncol, flwdir, nidp)

        flwacc = dx*dy
        flwpar = 1

        do col = 1, ncol

            do row = 1, nrow

                if (nidp(row, col) .eq. 0) call acc_par_downstream_cell(nrow, ncol, flwdir, nidp, row, col, flwacc, flwpar)

            end do

        end do

    end subroutine flow_accumulation_partition

    !% Works for small array
    function argsort_i(a, asc) result(b)

        implicit none

        integer, dimension(:), intent(in) :: a
        logical, intent(in) :: asc
        integer, dimension(size(a)) :: b

        integer :: n, i, iloc, temp
        integer, dimension(size(a)) :: a2

        a2 = a
        n = size(a)

        do i = 1, n

            b(i) = i

        end do

        do i = 1, n - 1

            if (asc) then

                iloc = minloc(a2(i:), 1) + i - 1

            else

                iloc = maxloc(a2(i:), 1) + i - 1

            end if

            if (iloc .ne. i) then

                temp = a2(i); a2(i) = a2(iloc); a2(iloc) = temp
                temp = b(i); b(i) = b(iloc); b(iloc) = temp

            end if

        end do

    end function argsort_i

    !% Works for small array
    function argsort_r(a, asc) result(b)

        implicit none

        real(4), dimension(:), intent(in) :: a
        logical, intent(in) :: asc
        integer, dimension(size(a)) :: b

        integer :: n, i, iloc, temp_i
        real(4) :: temp_r
        real(4), dimension(size(a)) :: a2

        a2 = a
        n = size(a)

        do i = 1, n

            b(i) = i

        end do

        do i = 1, n - 1

            if (asc) then

                iloc = minloc(a2(i:), 1) + i - 1

            else

                iloc = maxloc(a2(i:), 1) + i - 1

            end if

            if (iloc .ne. i) then

                temp_r = a2(i); a2(i) = a2(iloc); a2(iloc) = temp_r
                temp_i = b(i); b(i) = b(iloc); b(iloc) = temp_i

            end if

        end do

    end function argsort_r

    recursive subroutine distance_upstream_cells(nrow, ncol, ng, &
    & flwdir, dx, dy, row, col, row_dln, col_dln, flag, flwdst)

        implicit none

        integer, intent(in) :: nrow, ncol, ng
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(4), dimension(nrow, ncol), intent(in) :: dx, dy
        integer, intent(in) :: row, col
        integer, dimension(ng), intent(in) :: row_dln, col_dln
        integer, dimension(ng), intent(inout) :: flag
        real(4), dimension(nrow, ncol), intent(inout) :: flwdst

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)
        integer :: i, j, row_imd, col_imd

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle

            if (flwdir(row_imd, col_imd) .ne. i) cycle

            if (abs(flwdir(row, col) - flwdir(row_imd, col_imd)) .eq. 4) cycle

            !% Check for nested catchment and set flag
            do j = 1, ng

                if (row_imd .eq. row_dln(j) .and. &
                & col_imd .eq. col_dln(j)) flag(j) = 1

            end do

            !% Avoid to compute square root if not diagonal
            if (dcol(i) .eq. 0) then

                flwdst(row_imd, col_imd) = flwdst(row, col) + dy(row, col)

            else if (drow(i) .eq. 0) then

                flwdst(row_imd, col_imd) = flwdst(row, col) + dx(row, col)

            else

                flwdst(row_imd, col_imd) = flwdst(row, col) + &
                & sqrt(dx(row, col)*dx(row, col) + dy(row, col)*dy(row, col))

            end if

            call distance_upstream_cells(nrow, ncol, ng, flwdir, &
            & dx, dy, row_imd, col_imd, row_dln, col_dln, flag, flwdst)

        end do

    end subroutine distance_upstream_cells

    subroutine flow_distance(nrow, ncol, ng, flwdir, dx, dy, row_dln, col_dln, area_dln, &
    & flwdst)

        implicit none

        integer, intent(in) :: nrow, ncol, ng
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(4), dimension(nrow, ncol), intent(in) :: dx, dy
        integer, dimension(ng), intent(inout) :: row_dln, col_dln
        real(4), dimension(ng), intent(in) :: area_dln
        real(4), dimension(nrow, ncol), intent(out) :: flwdst

        integer, dimension(ng) :: indices, flag
        integer :: i, ind

        !% Transform from Python to FORTRAN index
        row_dln = row_dln + 1
        col_dln = col_dln + 1

        flwdst = -99._4

        indices = argsort_r(area_dln, .false.)

        flag = 0

        do i = 1, ng

            ind = indices(i)

            if (flag(ind) .eq. 0) then

                flwdst(row_dln(ind), col_dln(ind)) = 0._4

                call distance_upstream_cells(nrow, ncol, ng, flwdir, dx, dy, row_dln(ind), &
                & col_dln(ind), row_dln, col_dln, flag, flwdst)

            end if

        end do

        !% Transform from FORTRAN to Python index
        row_dln = row_dln - 1
        col_dln = col_dln - 1

    end subroutine flow_distance

    subroutine flow_partition_variable(nrow, ncol, npar, flwpar, ncpar, cscpar, cpar_to_rowcol)

        implicit none

        integer, intent(in) :: nrow, ncol, npar
        integer, dimension(nrow, ncol), intent(in) :: flwpar
        integer, dimension(npar), intent(out) :: ncpar, cscpar
        integer, dimension(nrow*ncol, 2), intent(out) :: cpar_to_rowcol

        integer :: row, col, fp, ind

        ncpar = 0
        cscpar = 0

        do col = 1, ncol

            do row = 1, nrow

                fp = flwpar(row, col)
                if (fp .eq. npar) cycle
                cscpar(fp + 1:npar) = cscpar(fp + 1:npar) + 1

            end do

        end do

        do col = 1, ncol

            do row = 1, nrow

                fp = flwpar(row, col)
                ncpar(fp) = ncpar(fp) + 1
                ind = cscpar(fp) + ncpar(fp)

                !% Transform from FORTRAN to Python index
                cpar_to_rowcol(ind, :) = (/row - 1, col - 1/)

            end do

        end do

    end subroutine flow_partition_variable

    !% This function only check "pair" of cells. A sink can be more than 2 cells ...
    subroutine detect_sink(nrow, ncol, flwdir, sink)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(out) :: sink

        integer :: row, col, i, row_imd, col_imd
        integer, dimension(3) :: drow = (/0, 1, 1/)
        integer, dimension(3) :: dcol = (/1, 1, 0/)
        integer, dimension(3) :: dir = (/3, 4, 5/)
        integer, dimension(3) :: opposite_dir = (/7, 8, 1/)

        sink = 0

        do col = 1, ncol - 1

            do row = 1, nrow - 1

                do i = 1, 3

                    row_imd = row + drow(i)
                    col_imd = col + dcol(i)

                    if (flwdir(row_imd, col_imd) .eq. opposite_dir(i) .and. flwdir(row, col) .eq. dir(i)) then

                        sink(row, col) = 1
                        sink(row_imd, col_imd) = 1

                    end if

                end do

            end do

        end do

    end subroutine detect_sink

end module mw_mesh
