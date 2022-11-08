!%      This module `mw_mask` encapsulates all SMASH mask routine.
!%      This module is wrapped

module mw_mask

    use md_constant, only: sp, np, ns
    use mwd_mesh, only: MeshDT
    
    contains

        recursive subroutine mask_upstream_cells(row, col, mesh, mask)
        
            !% Notes
            !% -----
            !%
            !% Masked upstream cells subroutine
            !% Given (row, col) indices, MeshDT, 
            !% it returns a mask where True values are upstream cells of
            !% the given row, col indices.
        
            implicit none
            
            integer, intent(in) :: row, col
            type(MeshDT), intent(in) :: mesh
            logical, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: mask
            
            integer :: i, col_imd, row_imd
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            mask(row, col) = .true.
    
            do i=1, 8
                
                col_imd = col + dcol(i)
                row_imd = row + drow(i)
                
                if (col_imd .gt. 0 .and. col_imd .le. mesh%ncol .and. &
                &   row_imd .gt. 0 .and. row_imd .le. mesh%nrow) then
                
                    if (mesh%flwdir(row_imd, col_imd) .eq. dkind(i)) then
                        
                        call mask_upstream_cells(row_imd, col_imd, &
                        & mesh, mask)
                    
                    end if
                    
                end if
            
            end do
                    
        end subroutine mask_upstream_cells

end module mw_mask
