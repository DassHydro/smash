module mw_meshing

    contains

    recursive subroutine mask_upstream_cells(flwdir, col, row, ncol, &
    & nrow, mask)
        
        integer, intent(in) :: ncol, nrow, col, row
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        logical, dimension(nrow, ncol), intent(inout) :: mask
        
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)
        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dkind = (/1, 2, 3, 4, 5, 6, 7, 8/)
        integer :: i, col_imd, row_imd
        
        mask(row, col) = .true.
        
        do i=1, 8
        
            col_imd = col + dcol(i)
            row_imd = row + drow(i)
            
            if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
            &   row_imd .gt. 0 .and. row_imd .le. nrow) then
            
                if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
                    
                    call mask_upstream_cells(flwdir, col_imd, row_imd, &
                    & ncol, nrow, mask)
                
                end if
                
            end if
        
        end do
        
    end subroutine mask_upstream_cells


    subroutine catchment_dln(flwdir, col, row, xres, yres, area, &
    & max_depth, mask_dln, col_otl, row_otl)
        
        integer, dimension(:,:), intent(in) :: flwdir
        integer, intent(inout) :: col, row
        integer, intent(in) :: max_depth
        real(4), intent(in) :: xres, yres, area
        logical, dimension(size(flwdir, 1), size(flwdir, 2)), &
        & intent(out) :: mask_dln
        integer, intent(out) :: col_otl, row_otl
        
        logical, dimension(size(flwdir, 1), size(flwdir, 2)) :: &
        & mask_dln_imd
        integer :: i, j, col_imd, row_imd, ncol, nrow
        real(4) :: min_tol, tol
        
        !% Transform from Python to FORTRAN index
        col = col + 1
        row = row + 1
        
        ncol = size(flwdir, 2)
        nrow = size(flwdir, 1)

        min_tol = 1.
        
        do i=-max_depth, max_depth
        
            do j=-max_depth, max_depth
            
                col_imd = col + i
                row_imd = row + j
                mask_dln_imd = .false.
                
                if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
                &   row_imd .gt. 0 .and. row_imd .le. nrow) then
            
                    call mask_upstream_cells(flwdir, col_imd, &
                    & row_imd, ncol, nrow, mask_dln_imd)
                    
                    tol = abs(area - (count(mask_dln_imd) &
                    & * (xres * yres))) / area
                    
                    if (tol .lt. min_tol) then
                    
                        min_tol = tol
                        
                        ! Transform from FORTRAN to Python index
                        col_otl = col_imd - 1
                        row_otl = row_imd - 1
                        
                        mask_dln = mask_dln_imd
                        
                    end if
                
                end if
            
            end do
        
        end do
        
        !% Transform from FORTRAN to Python index
        col = col - 1
        row = row - 1

    end subroutine catchment_dln


    recursive subroutine downstream_cell_drained_area(flwdir, col, &
    & row, ncol, nrow, da)
        
        integer, intent(in) :: col, row, ncol, nrow
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        integer, dimension(nrow, ncol), intent(inout) :: da
        
        integer, dimension(8) :: dcol = (/0, 1, 1, 1, 0, -1, -1, -1/)
        integer, dimension(8) :: drow = (/-1, -1, 0, 1, 1, 1, 0, -1/)
        
        integer :: fd
        
        fd = flwdir(row, col)
        
        !% Check no data
        if (fd .gt. 0) then
        
            !% Check bounds
            if (col + dcol(fd) .gt. 0 .and. col + dcol(fd) .le. ncol &
            & .and. row + drow(fd) .gt. 0 .and. &
            & row + drow(fd) .le. nrow) then
                
                !% Check pit
                if (abs(flwdir(row, col) - flwdir(row + drow(fd), &
                & col + dcol(fd))) .ne. 4) then
                
                    da(row + drow(fd), col + dcol(fd)) = &
                    & da(row + drow(fd), col + dcol(fd)) + 1
                    
                    call downstream_cell_drained_area(flwdir, &
                    & col + dcol(fd), row + drow(fd), ncol, nrow, da)
                    
                end if
            
            end if
        
        end if
            
    end subroutine downstream_cell_drained_area


    subroutine drained_area(flwdir, da)

        integer, dimension(:,:), intent(in) :: flwdir
        integer, dimension(size(flwdir, 1), size(flwdir, 2)), &
        & intent(out) :: da
        
        integer :: col, row, ncol, nrow
        
        da = 1
        ncol = size(flwdir, 2)
        nrow = size(flwdir, 1)
        
        do row=1, nrow
        
            do col=1, ncol
                
                call downstream_cell_drained_area(flwdir, col, row, &
                & ncol, nrow, da)
            
            end do

        end do

    end subroutine drained_area

    !% Workq for small array
    function argsort_i(a, asc) result(b)

        implicit none

        integer, dimension(:), intent(in) :: a
        logical, intent(in) :: asc
        integer, dimension(size(a)) :: b
        

        integer :: n, i, iloc, temp
        integer, dimension(size(a)) :: a2
        
        a2 = a
        n = size(a)
        
        do i=1, n
            
            b(i) = i
        
        end do
        
        do i=1, n-1
            
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

    
    recursive subroutine distance_upstream_cells(flwdir, col, row, ncol, &
    & nrow, col_ol, row_ol, flag, dx, flwdst)
    
        implicit none
        
        integer, intent(in) :: ncol, nrow, col, row
        integer, dimension(:), intent(in) :: col_ol, row_ol
        integer, dimension(:), intent(inout) :: flag
        real(4), intent(in) :: dx
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(4), dimension(nrow, ncol), intent(inout) :: flwdst
        
        integer, dimension(8) :: dcol = (/0, -1, -1, -1, 0, 1, 1, 1/)
        integer, dimension(8) :: drow = (/1, 1, 0, -1, -1, -1, 0, 1/)
        integer, dimension(8) :: dkind = (/1, 2, 3, 4, 5, 6, 7, 8/)
        integer :: i, j, col_imd, row_imd, cdst, rdst
        
        do i=1, 8
        
            col_imd = col + dcol(i)
            row_imd = row + drow(i)
            
            do j=1, size(flag)
            
                if (col_imd .eq. col_ol(j) .and. &
                & row_imd .eq. row_ol(j)) flag(j) = 1
                        
            end do
            
            if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
            &   row_imd .gt. 0 .and. row_imd .le. nrow) then
            
                if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
            
                    cdst = abs(col - col_imd)
                    rdst = abs(row - row_imd)
                    
                    if (cdst + rdst .gt. 1) then
                        
                        flwdst(row_imd, col_imd) = flwdst(row, col) + sqrt(2 * dx * dx)
                        
                    else
                    
                        flwdst(row_imd, col_imd) = flwdst(row, col) + dx
                        
                    end if
                    
                    call distance_upstream_cells(flwdir, col_imd, row_imd, &
                    & ncol, nrow, col_ol, row_ol, flag, dx, flwdst)
                
                end if
                
            end if
        
        end do
    
    end subroutine distance_upstream_cells
    
    
    subroutine flow_distance(flwdir, col_ol, row_ol, area_ol, dx, &
    & flwdst)

        implicit none
        
        integer, dimension(:,:), intent(in) :: flwdir
        integer, dimension(:), intent(inout) :: col_ol, row_ol
        integer, dimension(:), intent(in) :: area_ol
        real(4), intent(in) :: dx
        real(4), dimension(size(flwdir, 1), size(flwdir, 2)), &
        & intent(out) :: flwdst
        
        integer, dimension(size(area_ol)) :: indices, flag
        integer :: ncol, nrow, i, ind
        
        !% Transform from Python to FORTRAN index
        col_ol = col_ol + 1
        row_ol = row_ol + 1
        
        flwdst = -99.
        
        ncol = size(flwdir, 2)
        nrow = size(flwdir, 1)
        
        indices = argsort_i(area_ol, .false.)
        
        flag = 0

        do i=1, size(area_ol)
        
            ind = indices(i)
        
            if (flag(ind) .eq. 0) then
            
                flwdst(row_ol(ind), col_ol(ind)) = 0.
            
                call distance_upstream_cells(flwdir, col_ol(ind), &
                & row_ol(ind), ncol, nrow, col_ol, row_ol, flag, dx, flwdst)
                
            end if
        
        end do
        
        !% Transform from FORTRAN to Python index
        col_ol = col_ol - 1
        row_ol = row_ol - 1
        
    end subroutine flow_distance
  

end module mw_meshing
