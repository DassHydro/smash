!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - compute_interception_capacity

module mw_interception_capacity

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_sparse_matrix_manipulation, only: sparse_matrix_to_matrix
    use md_gr_operator, only: gr_interception
    use m_array_creation, only: arange

    implicit none

contains

    subroutine compute_interception_capacity(setup, mesh, input_data, day_index, nday, ci)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, dimension(setup%ntime_step), intent(in) :: day_index
        integer, intent(in) :: nday
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: ci

        real(sp), dimension(mesh%nrow, mesh%ncol, nday) :: daily_prcp, daily_pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet, h, daily_cumulated, sub_daily_cumulated
        real(sp), dimension(:), allocatable :: cmax
        real(sp), dimension(:, :, :), allocatable :: diff
        real(sp) :: stt, stp, step, ec, pth
        integer :: i, j, ind, row, col, n

        !% =========================================================================================================== %!
        !%   Daily aggregation of precipitation and evapotranspiration
        !% =========================================================================================================== %!

        daily_prcp = 0._sp
        daily_pet = 0._sp

        do i = 1, setup%ntime_step

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(i), matrix_prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(i), matrix_pet)

            else

                matrix_prcp = input_data%atmos_data%prcp(:, :, i)
                matrix_pet = input_data%atmos_data%pet(:, :, i)

            end if

            ind = day_index(i)

            daily_prcp(:, :, ind) = daily_prcp(:, :, ind) + matrix_prcp
            daily_pet(:, :, ind) = daily_pet(:, :, ind) + matrix_pet

        end do

        daily_cumulated = 0._sp

        do i = 1, nday

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    daily_cumulated(row, col) = daily_cumulated(row, col) + min(daily_prcp(row, col, i), daily_pet(row, col, i))

                end do

            end do

        end do

        !% =========================================================================================================== %!
        !%   Calculate interception storage
        !% =========================================================================================================== %!

        stt = 0.1_sp
        stp = 5._sp
        step = 0.1_sp
        n = ceiling((stp - stt)/step)

        allocate (cmax(n), diff(mesh%nrow, mesh%ncol, n))

        call arange(stt, stp, step, cmax)

        do i = 1, n

            h = 0._sp
            sub_daily_cumulated = 0._sp

            do j = 1, setup%ntime_step

                if (setup%sparse_storage) then

                    call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(j), matrix_prcp)
                    call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(j), matrix_pet)

                else

                    matrix_prcp = input_data%atmos_data%prcp(:, :, j)
                    matrix_pet = input_data%atmos_data%pet(:, :, j)

                end if

                do col = 1, mesh%ncol

                    do row = 1, mesh%nrow

                        if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                        call gr_interception(matrix_prcp(row, col), matrix_pet(row, col), cmax(i), &
                        & h(row, col), pth, ec)
                        sub_daily_cumulated(row, col) = sub_daily_cumulated(row, col) + ec

                    end do

                end do

            end do

            diff(:, :, i) = abs(sub_daily_cumulated - daily_cumulated)

        end do

        do row = 1, mesh%ncol

            do col = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                ! Specify dim=1 to return an integer
                ind = minloc(diff(row, col, :), dim=1)

                ci(row, col) = cmax(ind)

            end do

        end do

    end subroutine compute_interception_capacity

end module mw_interception_capacity
