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
    use mwd_sparse_matrix_manipulation, only: sparse_matrix_to_matrix, matrix_to_sparse_matrix
    use mw_mask, only: mask_upstream_cells

    implicit none

contains

    subroutine compute_mean_atmos(setup, mesh, input_data)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data

        logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
        logical, dimension(mesh%nrow, mesh%ncol) :: mask_prcp, mask_pet, mask_snow, mask_temp
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet, matrix_snow, matrix_temp
        integer :: i, j, n_prcp, n_pet, n_snow, n_temp

        mask_gauge = .false.

        do i = 1, mesh%ng

            call mask_upstream_cells(mesh, mesh%gauge_pos(i, 1), &
            & mesh%gauge_pos(i, 2), mask_gauge(:, :, i))

        end do

        do i = 1, setup%ntime_step

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(i), matrix_prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(i), matrix_pet)

                if (setup%snow_module_present) then

                    call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_snow(i), matrix_snow)
                    call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_temp(i), matrix_temp)

                end if

            else

                matrix_prcp = input_data%atmos_data%prcp(:, :, i)
                matrix_pet = input_data%atmos_data%pet(:, :, i)

                if (setup%snow_module_present) then

                    matrix_snow = input_data%atmos_data%snow(:, :, i)
                    matrix_temp = input_data%atmos_data%temp(:, :, i)

                end if

            end if

            do j = 1, mesh%ng

                mask_prcp = (matrix_prcp .ge. 0._sp .and. mask_gauge(:, :, j))
                n_prcp = count(mask_prcp)
                mask_pet = (matrix_pet .ge. 0._sp .and. mask_gauge(:, :, j))
                n_pet = count(mask_pet)

                ! Will be -99 if n is equal to 0
                if (n_prcp .gt. 0) input_data%atmos_data%mean_prcp(j, i) = sum(matrix_prcp, mask=mask_prcp)/n_prcp
                if (n_pet .gt. 0) input_data%atmos_data%mean_pet(j, i) = sum(matrix_pet, mask=mask_pet)/n_pet

                if (setup%snow_module_present) then

                    mask_snow = (matrix_snow .ge. 0._sp .and. mask_gauge(:, :, j))
                    n_snow = count(mask_snow)
                    mask_temp = (matrix_temp .gt. -99._sp .and. mask_gauge(:, :, j))
                    n_temp = count(mask_temp)

                    ! Will be -99 if n is equal to 0
                    if (n_snow .gt. 0) input_data%atmos_data%mean_snow(j, i) = sum(matrix_snow, mask=mask_snow)/n_snow
                    if (n_temp .gt. 0) input_data%atmos_data%mean_temp(j, i) = sum(matrix_temp, mask=mask_temp)/n_temp

                end if

            end do

        end do

    end subroutine compute_mean_atmos

    subroutine compute_prcp_partitioning(setup, mesh, input_data)
        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data

        integer :: i
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_snow, matrix_temp, ratio

        do i = 1, setup%ntime_step

            if (setup%sparse_storage) then

                ! Do not check for snow module because this subroutine should be only called when a snow module
                ! has been selected
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(i), matrix_prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_snow(i), matrix_snow)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_temp(i), matrix_temp)

            else

                matrix_prcp = input_data%atmos_data%prcp(:, :, i)
                matrix_snow = input_data%atmos_data%snow(:, :, i)
                matrix_temp = input_data%atmos_data%temp(:, :, i)

            end if

            where (matrix_snow .ge. 0._sp) matrix_prcp = matrix_prcp + matrix_snow

            ! If there is no temperature data, precipitation is assumed to be only liquid
            where (matrix_temp .gt. -99._sp)

                ratio = 1._sp - 1._sp/(1._sp + exp(10._sp/4._sp*matrix_temp - 1._sp))

            else where

                ratio = 1._sp

            end where

            where (matrix_prcp .ge. 0._sp)

                matrix_snow = (1._sp - ratio)*matrix_prcp
                matrix_prcp = matrix_prcp - matrix_snow

            else where

                matrix_snow = -99._sp

            end where

            if (setup%sparse_storage) then

                call matrix_to_sparse_matrix(mesh, matrix_prcp, 0._sp, input_data%atmos_data%sparse_prcp(i))
                call matrix_to_sparse_matrix(mesh, matrix_snow, 0._sp, input_data%atmos_data%sparse_snow(i))

            else

                input_data%atmos_data%prcp(:, :, i) = matrix_prcp
                input_data%atmos_data%snow(:, :, i) = matrix_snow

            end if

        end do

    end subroutine compute_prcp_partitioning

end module mw_atmos_statistic
