!%      (MW) Module Wrapped.
!% 
!%      Subroutine
!%      ----------
!%
!%      - compute_mean_forcing

module mw_atmos_statistic_dev

    use md_constant_dev, only: sp
    use mwd_setup_dev, only: SetupDT_dev
    use mwd_mesh_dev, only: MeshDT_dev
    use mwd_input_data_dev, only: Input_DataDT_dev
    use mw_sparse_storage_dev, only: sparse_vector_to_matrix_r_dev
    use mw_mask_dev, only: mask_upstream_cells_dev

    implicit none

contains

    subroutine compute_mean_atmos_dev(setup, mesh, input_data)

        implicit none

        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        type(Input_DataDT_dev), intent(inout) :: input_data

        logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
        logical, dimension(mesh%nrow, mesh%ncol) :: mask_prcp, mask_pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet
        integer :: i, j

        mask_gauge = .false.

        do i = 1, mesh%ng

            call mask_upstream_cells_dev(mesh, mesh%gauge_pos(i, 1), &
            & mesh%gauge_pos(i, 2), mask_gauge(:, :, i))

        end do

        do i = 1, setup%ntime_step

            if (setup%sparse_storage) then

                call sparse_vector_to_matrix_r_dev(mesh, input_data%atmos_data%sparse_prcp(:, i), matrix_prcp)
                call sparse_vector_to_matrix_r_dev(mesh, input_data%atmos_data%sparse_pet(:, i), matrix_pet)

            else

                matrix_prcp = input_data%atmos_data%prcp(:, :, i)
                matrix_pet = input_data%atmos_data%pet(:, :, i)

            end if

            do j = 1, mesh%ng

                mask_prcp = (matrix_prcp .ge. 0._sp .and. mask_gauge(:, :, j))
                mask_pet = (matrix_pet .ge. 0._sp .and. mask_gauge(:, :, j))

                input_data%atmos_data%mean_prcp(j, i) = sum(matrix_prcp, mask=mask_prcp)/count(mask_prcp)
                input_data%atmos_data%mean_pet(j, i) = sum(matrix_pet, mask=mask_pet)/count(mask_pet)

            end do

        end do

    end subroutine compute_mean_atmos_dev

end module mw_atmos_statistic_dev
