module md_snow_operator

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_options !% only: OptionsDT

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

    subroutine ssn_timestep(setup, mesh, options, snow, temp, kmlt, hs, mlt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: snow, temp
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: kmlt
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: hs
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: mlt

        integer :: row, col

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, snow, temp, kmlt, hs, mlt) &
        !$OMP& private(row, col)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (snow(row, col) .ge. 0._sp .and. temp(row, col) .gt. -99._sp) then

                    call simple_snow(snow(row, col), temp(row, col), kmlt(row, col), hs(row, col), mlt(row, col))

                else

                    mlt(row, col) = 0._sp

                end if

            end do
        end do
        !$OMP end parallel do

    end subroutine ssn_timestep

end module md_snow_operator
