!%      (MW) Module Wrapped.
!% 
!%      Subroutine
!%      ----------
!%
!%      - compute_mean_forcing

module mw_mask_dev

    use md_constant_dev, only: sp
    use mwd_mesh_dev, only: MeshDT_dev

contains

    recursive subroutine mask_upstream_cells_dev(mesh, row, col, mask)

        implicit none

        type(MeshDT_dev), intent(in) :: mesh
        integer, intent(in) :: row, col
        logical, dimension(mesh%nrow, mesh%ncol), intent(inout) &
        & :: mask

        integer :: i, row_imd, col_imd
        integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
        integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]

        mask(row, col) = .true.

        do i = 1, 8

            row_imd = row + drow(i)
            col_imd = col + dcol(i)
            
            if (row_imd .lt. 1 .or. row_imd .gt. mesh%nrow .or. col_imd .lt. 1 .or. col_imd .gt. mesh%ncol) cycle
            
            if (mesh%flwdir(row_imd, col_imd) .eq. i) call mask_upstream_cells_dev(mesh, row_imd, col_imd, mask)

        end do

    end subroutine mask_upstream_cells_dev

end module mw_mask_dev
