!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - get_mean_gauge_atmos_data
!%      - compute_mean_atmos
!%      - compute_prcp_partitioning

module mw_atmos_statistic

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_atmos_manipulation, only: get_atmos_data_time_step, set_atmos_data_time_step
    use mwd_sparse_matrix_manipulation, only: sparse_matrix_to_matrix, matrix_to_sparse_matrix
    use mw_mask, only: mask_upstream_cells

    implicit none

contains

    subroutine get_mean_gauge_atmos_data(mesh, mask_gauge, mask_atmos, mat_atmos, mean_gauge_atmos)

        implicit none

        type(MeshDT), intent(in) :: mesh
        logical, dimension(mesh%nrow, mesh%ncol), intent(in) :: mask_gauge
        logical, dimension(mesh%nrow, mesh%ncol), intent(in) :: mask_atmos
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: mat_atmos
        real(sp), intent(inout) :: mean_gauge_atmos

        integer :: row, col, n_gauge_atmos

        mean_gauge_atmos = 0._sp
        n_gauge_atmos = 0

        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (.not. mask_gauge(row, col) .or. .not. mask_atmos(row, col)) cycle

                mean_gauge_atmos = mean_gauge_atmos + mat_atmos(row, col)
                n_gauge_atmos = n_gauge_atmos + 1

            end do
        end do

        if (n_gauge_atmos .gt. 0) then
            mean_gauge_atmos = mean_gauge_atmos/n_gauge_atmos

        else
            mean_gauge_atmos = -99._sp

        end if

    end subroutine get_mean_gauge_atmos_data

    ! TODO: All the atmospheric data variables are hard-coded. This subroutine might need to be rewritten if
    ! the total number of atmospheric data increase.
    subroutine compute_mean_atmos(setup, mesh, input_data)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data

        integer :: i, t
        logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
        real(sp), dimension(mesh%nrow, mesh%ncol) :: mat_prcp, mat_pet
        logical, dimension(mesh%nrow, mesh%ncol) :: mask_prcp, mask_pet
        real(sp), dimension(:, :), allocatable :: mat_snow, mat_temp
        logical, dimension(:, :), allocatable :: mask_snow, mask_temp

        mask_gauge = .false.

        do i = 1, mesh%ng

            call mask_upstream_cells(mesh, mesh%gauge_pos(i, 1), &
            & mesh%gauge_pos(i, 2), mask_gauge(:, :, i))

        end do

        !% Only allocate snow and temp variables if a snow module has been choosen
        if (setup%snow_module_present) then
            allocate (mat_snow(mesh%nrow, mesh%ncol), mat_temp(mesh%nrow, mesh%ncol))
            allocate (mask_snow(mesh%nrow, mesh%ncol), mask_temp(mesh%nrow, mesh%ncol))
        end if

        do t = 1, setup%ntime_step

            call get_atmos_data_time_step(setup, mesh, input_data, t, "prcp", mat_prcp)
            call get_atmos_data_time_step(setup, mesh, input_data, t, "pet", mat_pet)

            if (setup%snow_module_present) then
                call get_atmos_data_time_step(setup, mesh, input_data, t, "snow", mat_snow)
                call get_atmos_data_time_step(setup, mesh, input_data, t, "temp", mat_temp)
            end if

            mask_prcp = (mat_prcp .ge. 0._sp)
            mask_pet = (mat_pet .ge. 0._sp)

            if (setup%snow_module_present) then
                mask_snow = (mat_snow .ge. 0._sp)
                mask_temp = (mat_temp .gt. -99._sp)
            end if

            do i = 1, mesh%ng

                call get_mean_gauge_atmos_data(mesh, mask_gauge(:, :, i), mask_prcp, mat_prcp, &
                & input_data%atmos_data%mean_prcp(i, t))
                call get_mean_gauge_atmos_data(mesh, mask_gauge(:, :, i), mask_pet, mat_pet, &
                & input_data%atmos_data%mean_pet(i, t))

                if (setup%snow_module_present) then
                    call get_mean_gauge_atmos_data(mesh, mask_gauge(:, :, i), mask_snow, mat_snow, &
                    & input_data%atmos_data%mean_snow(i, t))
                    call get_mean_gauge_atmos_data(mesh, mask_gauge(:, :, i), mask_temp, mat_temp, &
                    & input_data%atmos_data%mean_temp(i, t))
                end if

            end do

        end do

    end subroutine compute_mean_atmos

    subroutine compute_prcp_partitioning(setup, mesh, input_data)
        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data

        integer :: t
        real(sp), dimension(mesh%nrow, mesh%ncol) :: mat_prcp, mat_snow, mat_temp, ratio

        do t = 1, setup%ntime_step

            call get_atmos_data_time_step(setup, mesh, input_data, t, "prcp", mat_prcp)
            call get_atmos_data_time_step(setup, mesh, input_data, t, "snow", mat_snow)
            call get_atmos_data_time_step(setup, mesh, input_data, t, "temp", mat_temp)

            where (mat_snow .ge. 0._sp) mat_prcp = mat_prcp + mat_snow

            ! If there is no temperature data, precipitation is assumed to be only liquid
            where (mat_temp .gt. -99._sp)

                ratio = 1._sp - 1._sp/(1._sp + exp(10._sp/4._sp*mat_temp - 1._sp))

            else where

                ratio = 1._sp

            end where

            where (mat_prcp .ge. 0._sp)

                mat_snow = (1._sp - ratio)*mat_prcp
                mat_prcp = mat_prcp - mat_snow

            else where

                mat_snow = -99._sp

            end where

            call set_atmos_data_time_step(setup, mesh, input_data, t, "prcp", mat_prcp)
            call set_atmos_data_time_step(setup, mesh, input_data, t, "snow", mat_snow)

        end do

    end subroutine compute_prcp_partitioning

end module mw_atmos_statistic
