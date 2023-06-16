!%      (MW) Module Wrapped.
!% 
!%      Subroutine
!%      ----------
!%
!%      - compute_mean_forcing

module mw_atmos_statistic

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mw_sparse_storage, only: sparse_vector_to_matrix_r
    use mw_mask, only: mask_upstream_cells

    implicit none

contains

    subroutine compute_mean_atmos(setup, mesh, input_data)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data

        logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
        logical, dimension(mesh%nrow, mesh%ncol) :: mask_prcp, mask_pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet
        integer :: i, j

        mask_gauge = .false.

        do i = 1, mesh%ng

            call mask_upstream_cells(mesh, mesh%gauge_pos(i, 1), &
            & mesh%gauge_pos(i, 2), mask_gauge(:, :, i))

        end do

        do i = 1, setup%ntime_step

            if (setup%sparse_storage) then

                call sparse_vector_to_matrix_r(mesh, input_data%atmos_data%sparse_prcp(:, i), matrix_prcp)
                call sparse_vector_to_matrix_r(mesh, input_data%atmos_data%sparse_pet(:, i), matrix_pet)

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

    end subroutine compute_mean_atmos

end module mw_atmos_statistic
