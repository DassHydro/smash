!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - adjust_interception_capacity

module mw_interception_capacity

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_sparse_matrix_manipulation, only: sparse_matrix_to_matrix
    use md_gr_operator, only: gr_interception
    use m_array_creation, only: arange
    use mwd_atmos_manipulation, only: get_atmos_data_time_step

    implicit none

contains

    subroutine adjust_interception_capacity(setup, mesh, input_data, day_index, nday, ci)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, dimension(setup%ntime_step), intent(in) :: day_index
        integer, intent(in) :: nday
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: ci

        real(sp), dimension(mesh%nrow, mesh%ncol, nday) :: daily_prcp, daily_pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: mat_prcp, mat_pet, h, daily_cumulated, sub_daily_cumulated
        real(sp), dimension(:), allocatable :: cmax
        real(sp), dimension(:, :, :), allocatable :: diff
        real(sp) :: stt, stp, step, pn, en, ei
        integer :: t, i, ind, row, col, n

        !% =========================================================================================================== %!
        !%   Daily aggregation of precipitation and evapotranspiration
        !% =========================================================================================================== %!

        daily_prcp = 0._sp
        daily_pet = 0._sp

        do t = 1, setup%ntime_step

            call get_atmos_data_time_step(setup, mesh, input_data, t, "prcp", mat_prcp)
            call get_atmos_data_time_step(setup, mesh, input_data, t, "pet", mat_pet)

            ind = day_index(t)

            where (mat_prcp .ge. 0._sp) daily_prcp(:, :, ind) = daily_prcp(:, :, ind) + mat_prcp
            where (mat_pet .ge. 0._sp) daily_pet(:, :, ind) = daily_pet(:, :, ind) + mat_pet

        end do

        daily_cumulated = 0._sp

        do t = 1, nday

            do col = 1, mesh%ncol
                do row = 1, mesh%nrow

                    daily_cumulated(row, col) = daily_cumulated(row, col) + min(daily_prcp(row, col, t), daily_pet(row, col, t))

                end do
            end do

        end do

        !% =========================================================================================================== %!
        !%   Calculate interception storage
        !% =========================================================================================================== %!

        stt = 0.1_sp
        stp = 3._sp
        step = 0.1_sp
        n = ceiling((stp - stt)/step)

        allocate (cmax(n), diff(mesh%nrow, mesh%ncol, n))

        call arange(stt, stp, step, cmax)

        do i = 1, n

            h = 0._sp
            sub_daily_cumulated = 0._sp

            do t = 1, setup%ntime_step

                call get_atmos_data_time_step(setup, mesh, input_data, t, "prcp", mat_prcp)
                call get_atmos_data_time_step(setup, mesh, input_data, t, "pet", mat_pet)

                do col = 1, mesh%ncol
                    do row = 1, mesh%nrow

                        if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                        if (mat_prcp(row, col) .ge. 0._sp .and. mat_pet(row, col) .ge. 0._sp) then

                            call gr_interception(mat_prcp(row, col), mat_pet(row, col), cmax(i), &
                            & h(row, col), pn, en)
                            ei = mat_pet(row, col) - en

                        else

                            ei = 0._sp

                        end if

                        sub_daily_cumulated(row, col) = sub_daily_cumulated(row, col) + ei

                    end do
                end do

            end do

            diff(:, :, i) = abs(sub_daily_cumulated - daily_cumulated)

        end do

        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                ! Specify dim=1 to return an integer
                ind = minloc(diff(row, col, :), dim=1)

                ci(row, col) = cmax(ind)

            end do
        end do

    end subroutine adjust_interception_capacity

end module mw_interception_capacity
