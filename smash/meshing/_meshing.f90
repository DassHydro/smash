recursive subroutine mask_upstream_cells(data, col, row, colsize, &
& rowsize, dkind, mask)
    
    integer, intent(in) :: colsize, rowsize, col, row
    integer, dimension(rowsize, colsize), intent(in) :: data
    integer, dimension(8), intent(in) :: dkind
    logical, dimension(rowsize, colsize), intent(inout) :: mask
    
    integer, dimension(8) :: dx = [0, -1, -1, -1, 0, 1, 1, 1]
    integer, dimension(8) :: dy = [1, 1, 0, -1, -1, -1, 0, 1]
    integer :: i, col_s, row_s
    
    mask(row, col) = .true.
    
    do i=1, 8
    
        col_s = col + dx(i)
        row_s = row + dy(i)
        
        if (col_s .gt. 0 .and. col_s .le. colsize .and. &
        &   row_s .gt. 0 .and. row_s .le. rowsize) then
        
            if (data(row_s, col_s) .eq. dkind(i)) then
                
                call mask_upstream_cells(data, col_s, row_s, &
                & colsize, rowsize, dkind, mask)
            
            end if
            
        end if
    
    end do
    
end subroutine mask_upstream_cells

subroutine catchment_dln(data, col, row, xres, yres, area, dkind, &
& max_depth, logical_dln, col_otl, row_otl)
    
    integer, dimension(:,:), intent(in) :: data
    integer, intent(inout) :: col, row
    integer, intent(in) :: max_depth
    real, intent(in) :: xres, yres, area
    integer, dimension(8), intent(in) :: dkind
    logical, dimension(size(data, 1), size(data, 2)), intent(out) :: &
    & logical_dln
    integer, intent(out) :: col_otl, row_otl
    
    logical, dimension(size(data, 1), size(data, 2)) :: &
    & logical_dln_loop
    integer :: i, j, col_loop, row_loop, colsize, rowsize
    real :: min_tol, tol
    
    ! Transform from Python to FORTRAN index
    col = col + 1
    row = row + 1
    
    colsize = size(data, 2)
    rowsize = size(data, 1)
    
    min_tol = 1.
    
    do i=-max_depth, max_depth
    
        do j=-max_depth, max_depth
        
            col_loop = col + i
            row_loop = row + j
            logical_dln_loop = .false.
            
            if (col_loop .gt. 0 .and. col_loop .le. colsize .and. &
            &   row_loop .gt. 0 .and. row_loop .le. rowsize) then
        
                call mask_upstream_cells(data, col_loop, row_loop, colsize, &
                & rowsize, dkind, logical_dln_loop)
                
                tol = abs(area - count(logical_dln_loop) &
                & * xres * yres) / area
                
                if (tol .lt. min_tol) then
                
                    min_tol = tol
                    ! Transform from FORTRAN to Python index
                    col_otl = col_loop - 1
                    row_otl = row_loop - 1
                    logical_dln = logical_dln_loop
                    
                end if
            
            end if
        
        end do
    
    end do

end subroutine catchment_dln

recursive subroutine downstream_cell_drained_area(data, col, row, colsize, &
& rowsize, da)
    
    integer, intent(in) :: col, row, colsize, rowsize
    integer, dimension(rowsize, colsize), intent(in) :: data
    integer, dimension(rowsize, colsize), intent(inout) :: da
    
    integer, dimension(8) :: dx = [0, 1, 1, 1, 0, -1, -1, -1]
    integer, dimension(8) :: dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    
    integer :: fd
    
    fd = data(row, col)
    
    ! Check no data
    if (fd .gt. 0) then
    
        ! Check bounds
        if (col + dx(fd) .gt. 0 .and. col + dx(fd) .le. colsize .and. &
        &   row + dy(fd) .gt. 0 .and. row + dy(fd) .le. rowsize) then
            
            ! Check pit
            if (abs(data(row, col) - data(row + dy(fd), col + dx(fd))) &
            & .ne. 4) then
            
                da(row + dy(fd), col + dx(fd)) = &
                & da(row + dy(fd), col + dx(fd)) + 1
                
                call downstream_cell_drained_area(data, col + dx(fd), &
                & row + dy(fd), colsize, rowsize, da)
                
            end if
        
        end if
    
    end if
        
end subroutine downstream_cell_drained_area

subroutine drained_area(data, da)

    integer, dimension(:,:), intent(in) :: data
    integer, dimension(size(data, 1), size(data, 2)), intent(out) :: da
    
    integer :: col, row, colsize, rowsize
    
    da = 1
    rowsize = size(data, 1)
    colsize = size(data, 2)
    
    do row=1, rowsize
    
        do col=1, colsize
            
            call downstream_cell_drained_area(data, col, row, colsize, &
            & rowsize, da)
        
        end do

    end do

end subroutine drained_area
