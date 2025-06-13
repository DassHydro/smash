!%      (MW) Module Differentiated.
!%
!%      Function
!%      ----------
!%
!%      - prior_regularization
!%      - smoothing_regularization_spatial_loop
!%      - smoothing_regularization

module md_regularization

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_options !% only: OptionsDT
    use mwd_parameters_manipulation !% only: control_tfm, control_to_parameters

    implicit none

contains

    function prior_regularization(parameters) result(res)

        implicit none

        type(ParametersDT), intent(in) :: parameters
        real(sp) :: res

        real(sp), dimension(parameters%control%n) :: dbkg

        ! Tapenade needs to know the size somehow
        dbkg = parameters%control%x - parameters%control%x_bkg

        res = sum(dbkg*dbkg)

    end function prior_regularization

    function smoothing_regularization_spatial_loop(mesh, matrix) result(res)

        implicit none

        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix
        real(sp) :: res

        integer :: row, col, m1row, p1row, m1col, p1col

        res = 0._sp

        do col = 1, mesh%ncol

            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0) cycle

                m1row = max(1, row - 1)
                p1row = min(mesh%nrow, row + 1)
                m1col = max(1, col - 1)
                p1col = min(mesh%ncol, col + 1)

                if (mesh%active_cell(m1row, col) .eq. 0) m1row = row
                if (mesh%active_cell(p1row, col) .eq. 0) p1row = row
                if (mesh%active_cell(row, m1col) .eq. 0) m1col = col
                if (mesh%active_cell(row, p1col) .eq. 0) p1col = col

                res = res + (matrix(p1row, col) - 2._sp*matrix(row, col) + matrix(m1row, col))**2 &
                & + (matrix(row, p1col) - 2._sp*matrix(row, col) + matrix(row, m1col))**2

            end do

        end do

    end function smoothing_regularization_spatial_loop

    function smoothing_regularization(setup, mesh, input_data, parameters, options, hard) result(res)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OptionsDT), intent(in) :: options
        logical :: hard
        real(sp) :: res

        integer :: i
        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix
        type(ParametersDT) :: parameters_bkg

        res = 0._sp

        ! This allows to retrieve a parameters structure with background values
        parameters_bkg = parameters
        parameters_bkg%control%x = parameters%control%x_bkg
        parameters_bkg%control%l = parameters%control%l_bkg
        parameters_bkg%control%u = parameters%control%u_bkg

        call control_tfm(parameters_bkg, options)
        call control_to_parameters(setup, mesh, input_data, parameters_bkg, options)

        ! Loop on rr_parameters
        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            if (hard) then
                matrix = parameters%rr_parameters%values(:, :, i)

            else
                matrix = parameters%rr_parameters%values(:, :, i) - parameters_bkg%rr_parameters%values(:, :, i)

            end if

            res = res + smoothing_regularization_spatial_loop(mesh, matrix)

        end do

        ! Loop on rr_initial_states
        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            if (hard) then
                matrix = parameters%rr_initial_states%values(:, :, i)

            else
                matrix = parameters%rr_initial_states%values(:, :, i) - parameters_bkg%rr_initial_states%values(:, :, i)

            end if

            res = res + smoothing_regularization_spatial_loop(mesh, matrix)

        end do

    end function smoothing_regularization

end module md_regularization
