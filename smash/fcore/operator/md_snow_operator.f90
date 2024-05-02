!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - simple_snow
!%      - ssn_time_step

module md_snow_operator

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_options !% only: OptionsDT
    use mwd_atmos_manipulation !% get_ac_atmos_data_time_step

    implicit none

contains

    subroutine simple_snow(snow, temp, kmlt, hs, mlt)

        implicit none

        real(sp), intent(in) :: snow, temp, kmlt
        real(sp), intent(inout) :: hs
        real(sp), intent(out) :: mlt

        hs = hs + snow

        mlt = max(0._sp, kmlt*temp)

        mlt = min(mlt, hs)

        hs = hs - mlt

    end subroutine simple_snow

    subroutine ssn_time_step(setup, mesh, input_data, options, time_step, ac_kmlt, ac_hs, ac_mlt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_kmlt
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hs
        real(sp), dimension(mesh%nac), intent(inout) :: ac_mlt

        integer :: row, col, k
        real(sp), dimension(mesh%nac) :: ac_snow, ac_temp

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "snow", ac_snow)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "temp", ac_temp)
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_snow, ac_temp, ac_kmlt, ac_hs, ac_mlt) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_snow(k) .ge. 0._sp .and. ac_temp(k) .gt. -99._sp) then

                    call simple_snow(ac_snow(k), ac_temp(k), ac_kmlt(k), ac_hs(k), ac_mlt(k))

                else

                    ac_mlt(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine ssn_time_step

end module md_snow_operator
