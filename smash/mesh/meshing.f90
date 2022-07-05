recursive subroutine mask_upstream_cells(flow, col, row, ncol, &
& nrow, mask)
    
    integer, intent(in) :: ncol, nrow, col, row
    integer, dimension(nrow, ncol), intent(in) :: flow
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
        
            if (flow(row_imd, col_imd) .eq. dkind(i)) then
                
                call mask_upstream_cells(flow, col_imd, row_imd, &
                & ncol, nrow, mask)
            
            end if
            
        end if
    
    end do
    
end subroutine mask_upstream_cells

subroutine catchment_dln(flow, col, row, xres, yres, area, &
& max_depth, mask_dln, col_otl, row_otl)
    
    integer, dimension(:,:), intent(in) :: flow
    integer, intent(inout) :: col, row
    integer, intent(in) :: max_depth
    real(4), intent(in) :: xres, yres, area
    logical, dimension(size(flow, 1), size(flow, 2)), intent(out) :: &
    & mask_dln
    integer, intent(out) :: col_otl, row_otl
    
    logical, dimension(size(flow, 1), size(flow, 2)) :: &
    & mask_dln_imd
    integer :: i, j, col_imd, row_imd, ncol, nrow
    real(4) :: min_tol, tol
    
    ! Transform from Python to FORTRAN index
    col = col + 1
    row = row + 1
    
    ncol = size(flow, 2)
    nrow = size(flow, 1)

    min_tol = 1.
    
    do i=-max_depth, max_depth
    
        do j=-max_depth, max_depth
        
            col_imd = col + i
            row_imd = row + j
            mask_dln_imd = .false.
            
            if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
            &   row_imd .gt. 0 .and. row_imd .le. nrow) then
        
                call mask_upstream_cells(flow, col_imd, row_imd, ncol, &
                & nrow, mask_dln_imd)
                
                tol = abs(area - count(mask_dln_imd) &
                & * xres * yres) / area
                
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
    
    ! Transform from FORTRAN to Python index
    col = col - 1
    row = row - 1

end subroutine catchment_dln

recursive subroutine downstream_cell_drained_area(flow, col, row, ncol, &
& nrow, da)
    
    integer, intent(in) :: col, row, ncol, nrow
    integer, dimension(nrow, ncol), intent(in) :: flow
    integer, dimension(nrow, ncol), intent(inout) :: da
    
    integer, dimension(8) :: dcol = (/0, 1, 1, 1, 0, -1, -1, -1/)
    integer, dimension(8) :: drow = (/-1, -1, 0, 1, 1, 1, 0, -1/)
    
    integer :: fd
    
    fd = flow(row, col)
    
    ! Check no data
    if (fd .gt. 0) then
    
        ! Check bounds
        if (col + dcol(fd) .gt. 0 .and. col + dcol(fd) .le. ncol .and. &
        &   row + drow(fd) .gt. 0 .and. row + drow(fd) .le. nrow) then
            
            ! Check pit
            if (abs(flow(row, col) - flow(row + drow(fd), col + dcol(fd))) &
            & .ne. 4) then
            
                da(row + drow(fd), col + dcol(fd)) = &
                & da(row + drow(fd), col + dcol(fd)) + 1
                
                call downstream_cell_drained_area(flow, col + dcol(fd), &
                & row + drow(fd), ncol, nrow, da)
                
            end if
        
        end if
    
    end if
        
end subroutine downstream_cell_drained_area

subroutine drained_area(flow, da)

    integer, dimension(:,:), intent(in) :: flow
    integer, dimension(size(flow, 1), size(flow, 2)), intent(out) :: da
    
    integer :: col, row, ncol, nrow
    
    da = 1
    ncol = size(flow, 2)
    nrow = size(flow, 1)
    
    do row=1, nrow
    
        do col=1, ncol
            
            call downstream_cell_drained_area(flow, col, row, ncol, &
            & nrow, da)
        
        end do

    end do

end subroutine drained_area
