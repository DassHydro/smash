module mw_meshing

    implicit none

contains

    recursive subroutine mask_upstream_cells(nrow, ncol, flwdir, row, col, mask)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, intent(in) :: row, col
        integer, dimension(nrow, ncol), intent(inout) :: mask

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)
        integer :: i, row_imd, col_imd

        mask(row, col) = 1

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            if (row_imd .gt. 0 .and. row_imd .le. nrow .and. &
                col_imd .gt. 0 .and. col_imd .le. ncol) then

                if (abs(flwdir(row, col) - flwdir(row_imd, col_imd)) .ne. 4) then

                    if (flwdir(row_imd, col_imd) .eq. i) then

                        call mask_upstream_cells(nrow, ncol, flwdir, row_imd, &
                        & col_imd, mask)

                    end if

                end if

            end if

        end do

    end subroutine mask_upstream_cells

    subroutine catchment_dln(nrow, ncol, flwdir, row, col, xres, yres, area, &
    & max_depth, mask_dln, row_dln, col_dln)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, intent(inout) :: col, row
        real(4), intent(in) :: xres, yres, area
        integer, intent(in) :: max_depth
        integer, dimension(nrow, ncol), intent(out) :: mask_dln
        integer, intent(out) :: col_dln, row_dln

        integer, dimension(nrow, ncol) :: mask_dln_imd
        integer :: i, j, row_imd, col_imd
        real(4) :: min_tol, tol

        !% Transform from Python to FORTRAN index
        row = row + 1
        col = col + 1

        min_tol = 1._4

        do i = -max_depth, max_depth

            do j = -max_depth, max_depth

                row_imd = row + j
                col_imd = col + i
                mask_dln_imd(:, :) = 0

                if (row_imd .gt. 0 .and. row_imd .le. nrow .and. &
                    col_imd .gt. 0 .and. col_imd .le. ncol) then

                    call mask_upstream_cells(nrow, ncol, flwdir, row_imd, &
                    & col_imd, mask_dln_imd)

                    tol = abs(area - (count(mask_dln_imd == 1) &
                                     & *(xres*yres)))/area

                    if (tol .lt. min_tol) then

                        min_tol = tol

                        !% Transform from FORTRAN to Python index
                        row_dln = row_imd - 1
                        col_dln = col_imd - 1

                        mask_dln = mask_dln_imd

                    end if

                end if

            end do

        end do

        !% Transform from FORTRAN to Python index
        row = row - 1
        col = col - 1

    end subroutine catchment_dln

    subroutine fill_nipd(nrow, ncol, flwdir, nipd)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(inout) :: nipd

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)

        integer :: row, col, row_imd, col_imd, i

        nipd(:, :) = 0

        do col = 1, ncol

            do row = 1, nrow

                do i = 1, 8

                    row_imd = row + drow(i)
                    col_imd = col + dcol(i)

                    if (row_imd .gt. 0 .and. row_imd .le. nrow .and. &
                    &   col_imd .gt. 0 .and. col_imd .le. ncol) then

                        if (flwdir(row_imd, col_imd) .eq. i) then

                            nipd(row, col) = nipd(row, col) + 1

                        end if

                    end if

                end do

            end do

        end do

    end subroutine fill_nipd

    recursive subroutine downstream_cell_flwacc(nrow, ncol, flwdir, row, col, nipd, flwacc)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, intent(in) :: row, col
        integer, dimension(nrow, ncol), intent(inout) :: nipd, flwacc

        integer, dimension(8) :: drow = (/-1, -1, 0, 1, 1, 1, 0, -1/)
        integer, dimension(8) :: dcol = (/0, 1, 1, 1, 0, -1, -1, -1/)

        integer :: fd, row_imd, col_imd

        fd = flwdir(row, col)

        !% Check no data
        if (fd .gt. 0) then

            row_imd = row + drow(fd)
            col_imd = col + dcol(fd)

            !% Check bounds
            if (row_imd .gt. 0 .and. row_imd .le. nrow .and. &
            &   col_imd .gt. 0 .and. col_imd .le. ncol) then

                !% Check pit
                if (abs(flwdir(row, col) - flwdir(row_imd, col_imd)) .ne. 4) then

                    flwacc(row_imd, col_imd) = flwacc(row_imd, col_imd) + flwacc(row, col)

                    if (nipd(row_imd, col_imd) .gt. 1) then

                        nipd(row_imd, col_imd) = nipd(row_imd, col_imd) - 1

                    else

                        call downstream_cell_flwacc(nrow, ncol, flwdir, &
                        & row_imd, col_imd, nipd, flwacc)

                    end if

                end if

            end if

        end if

    end subroutine downstream_cell_flwacc

    subroutine flow_accumulation(nrow, ncol, flwdir, flwacc)

        implicit none

        integer, intent(in) :: nrow, ncol
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(out) :: flwacc

        integer, dimension(nrow, ncol) :: nipd
        integer :: row, col

        flwacc(:, :) = 1

        call fill_nipd(nrow, ncol, flwdir, nipd)

        do col = 1, ncol

            do row = 1, nrow

                if (nipd(row, col) .eq. 0) then

                    call downstream_cell_flwacc(nrow, ncol, flwdir, row, col, nipd, flwacc)

                end if

            end do

        end do

    end subroutine flow_accumulation

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
    & flwdir, row, col, row_dln, col_dln, flag, dx, flwdst)

        implicit none

        integer, intent(in) :: nrow, ncol, ng
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, intent(in) :: row, col
        integer, dimension(ng), intent(in) :: row_dln, col_dln
        integer, dimension(ng), intent(inout) :: flag
        real(4), intent(in) :: dx

        real(4), dimension(nrow, ncol), intent(inout) :: flwdst

        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)

        integer :: i, j, row_imd, col_imd

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)

            if (row_imd .gt. 0 .and. row_imd .le. nrow .and. &
            &   col_imd .gt. 0 .and. col_imd .le. ncol) then

                if (flwdir(row_imd, col_imd) .eq. i) then

                    do j = 1, ng

                        if (row_imd .eq. row_dln(j) .and. &
                        & col_imd .eq. col_dln(j)) flag(j) = 1

                    end do

                    !% Avoid to compute square root if not diagonal
                    if (dcol(i) .eq. 0) then

                        flwdst(row_imd, col_imd) = flwdst(row, col) + dx

                    else if (drow(i) .eq. 0) then

                        flwdst(row_imd, col_imd) = flwdst(row, col) + dx

                    else

                        flwdst(row_imd, col_imd) = flwdst(row, col) + sqrt(2._4*dx*dx)

                    end if

                    call distance_upstream_cells(nrow, ncol, ng, flwdir, &
                    & row_imd, col_imd, row_dln, col_dln, flag, dx, flwdst)

                end if

            end if

        end do

    end subroutine distance_upstream_cells

    subroutine flow_distance(nrow, ncol, ng, flwdir, row_dln, col_dln, area_dln, dx, &
    & flwdst)

        implicit none

        integer, intent(in) :: nrow, ncol, ng
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(ng), intent(inout) :: row_dln, col_dln
        real(4), dimension(ng), intent(in) :: area_dln
        real(4), intent(in) :: dx
        real(4), dimension(nrow, ncol), intent(out) :: flwdst

        integer, dimension(ng) :: indices, flag
        integer :: i, ind

        !% Transform from Python to FORTRAN index
        row_dln = row_dln + 1
        col_dln = col_dln + 1

        flwdst(:, :) = -99._4

        indices = argsort_r(area_dln, .false.)

        flag = 0

        do i = 1, ng

            ind = indices(i)

            if (flag(ind) .eq. 0) then

                flwdst(row_dln(ind), col_dln(ind)) = 0._4

                call distance_upstream_cells(nrow, ncol, ng, flwdir, row_dln(ind), &
                & col_dln(ind), row_dln, col_dln, flag, dx, flwdst)

            end if

        end do

        !% Transform from FORTRAN to Python index
        row_dln = row_dln - 1
        col_dln = col_dln - 1

    end subroutine flow_distance

end module mw_meshing
