!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - roll_discharge
!%      - store_time_step
!%      - simulation_checkpoint
!%      - simulation

module md_simulation

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_rr_states !% only: RR_StatesDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use md_checkpoint_variable !% only: Checkpoint_VariableDT
    use md_snow_operator !% only: ssn_time_step
    use md_gr_operator !% only: gr4_time_step, gr5_time_step, grd_time_step, loieau_time_step
    use md_vic3l_operator !% only: vic3l_time_step
    use md_routing_operator !% only: lag0_time_step, lr_time_step, kw_time_step
    use mwd_sparse_matrix_manipulation !% only: matrix_to_ac_vector, &
    !& ac_vector_to_matrix

    implicit none

contains

    subroutine roll_discharge(ac_qz)

        implicit none

        real(sp), dimension(:, :), intent(inout) :: ac_qz

        integer :: i, nqz
        real(sp), dimension(size(ac_qz, 1)) :: tmp

        nqz = size(ac_qz, 2)

        do i = nqz, 2, -1

            tmp = ac_qz(:, nqz)
            ac_qz(:, nqz) = ac_qz(:, i - 1)
            ac_qz(:, i - 1) = tmp

        end do

    end subroutine roll_discharge

    subroutine store_time_step(setup, mesh, output, returns, checkpoint_variable, time_step)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OutputDT), intent(inout) :: output
        type(ReturnsDT), intent(inout) :: returns
        type(Checkpoint_VariableDT), intent(in) :: checkpoint_variable
        integer, intent(in) :: time_step

        integer :: i, k, time_step_returns

        do i = 1, mesh%ng
            k = mesh%rowcol_to_ind_ac(mesh%gauge_pos(i, 1), mesh%gauge_pos(i, 2))
            output%response%q(i, time_step) = checkpoint_variable%ac_qz(k, setup%nqz)

        end do

        !$AD start-exclude
        if (allocated(returns%mask_time_step)) then
            if (returns%mask_time_step(time_step)) then
                time_step_returns = returns%time_step_to_returns_time_step(time_step)

                !% Return states
                if (returns%rr_states_flag) then
                    do i = 1, setup%nrrs

                        call ac_vector_to_matrix(mesh, checkpoint_variable%ac_rr_states(:, i), &
                        & returns%rr_states(time_step_returns)%values(:, :, i))
                        returns%rr_states(time_step_returns)%keys = output%rr_final_states%keys

                    end do

                end if

                !% Return discharge grid
                if (returns%q_domain_flag) then
                    call ac_vector_to_matrix(mesh, checkpoint_variable%ac_qz(:, setup%nqz), &
                    & returns%q_domain(:, :, time_step_returns))
                end if

            end if
        end if
        !$AD end-exclude

    end subroutine store_time_step

    subroutine simulation_checkpoint(setup, mesh, input_data, parameters, output, options, returns, &
    & checkpoint_variable, start_time_step, end_time_step)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        type(Checkpoint_VariableDT), intent(inout) :: checkpoint_variable
        integer, intent(in) :: start_time_step, end_time_step

        integer :: t, rr_parameters_inc, rr_states_inc
        ! % Might add any number if needed
        real(sp), dimension(mesh%nac) :: h1, h2, h3, h4

        do t = start_time_step, end_time_step

            rr_parameters_inc = 0
            rr_states_inc = 0

            ! % Roll discharge buffer. Depending on the routing module, it is sometimes necessary to store
            ! % more than one discharge time step. Instead of storing all the time steps, we allocate an array
            ! % whose depth is equal to the depth of the time dependency, and then at each time step, we
            ! % overwrite the oldest time step by rolling the array.
            call roll_discharge(checkpoint_variable%ac_qtz)
            call roll_discharge(checkpoint_variable%ac_qz)

            ! Snow module
            select case (setup%snow_module)

                ! 'zero' module
            case ("zero")

                ! Nothing to do

                ! 'ssn' module
            case ("ssn")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hs

                call ssn_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % kmlt
                    h1, & ! % hs
                    checkpoint_variable%ac_mlt)

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1

                rr_parameters_inc = rr_parameters_inc + 1
                rr_states_inc = rr_states_inc + 1

            end select

            ! Hydrological module
            select case (setup%hydrological_module)

                ! 'gr4' module
            case ("gr4")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hi
                h2 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) ! % hp
                h3 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) ! % ht

                call gr4_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_mlt, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % ci
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % cp
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 3), & ! % ct
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 4), & ! % kexc
                    h1, & ! % hi
                    h2, & ! % hp
                    h3, & ! % ht
                    checkpoint_variable%ac_qtz(:, setup%nqz))

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) = h2
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) = h3

                rr_parameters_inc = rr_parameters_inc + 4
                rr_states_inc = rr_states_inc + 3

                ! 'gr5' module
            case ("gr5")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hi
                h2 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) ! % hp
                h3 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) ! % ht

                call gr5_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_mlt, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % ci
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % cp
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 3), & ! % ct
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 4), & ! % kexc
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 5), & ! % aexc
                    h1, & ! % hi
                    h2, & ! % hp
                    h3, & ! % ht
                    checkpoint_variable%ac_qtz(:, setup%nqz))

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) = h2
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) = h3

                rr_parameters_inc = rr_parameters_inc + 5
                rr_states_inc = rr_states_inc + 3

                ! 'grd' module
            case ("grd")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hp
                h2 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) ! % ht

                call grd_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_mlt, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % cp
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % ct
                    h1, & ! % hp
                    h2, & ! % ht
                    checkpoint_variable%ac_qtz(:, setup%nqz))

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) = h2

                rr_parameters_inc = rr_parameters_inc + 2
                rr_states_inc = rr_states_inc + 2

                ! 'loieau' module
            case ("loieau")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % ha
                h2 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) ! % hc

                call loieau_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_mlt, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % ca
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % cc
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 3), & ! % kb
                    h1, & ! % ha
                    h2, & ! % hc
                    checkpoint_variable%ac_qtz(:, setup%nqz))

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) = h2

                rr_parameters_inc = rr_parameters_inc + 3
                rr_states_inc = rr_states_inc + 2

                ! 'vic3l' module
            case ("vic3l")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hcl
                h2 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) ! % husl
                h3 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) ! % hmsl
                h4 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 4) ! % hbsl

                call vic3l_time_step( &
                    setup, &
                    mesh, &
                    input_data, &
                    options, &
                    t, &
                    checkpoint_variable%ac_mlt, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % b
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % cusl
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 3), & ! % cmsl
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 4), & ! % cbsl
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 5), & ! % ks
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 6), & ! % pbc
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 7), & ! % ds
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 8), & ! % dsm
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 9), & ! % ws
                    h1, & ! % hcl
                    h2, & ! % husl
                    h3, & ! % hmsl
                    h4, & ! % hbsl
                    checkpoint_variable%ac_qtz(:, setup%nqz))

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 2) = h2
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 3) = h3
                checkpoint_variable%ac_rr_states(:, rr_states_inc + 4) = h4

                rr_parameters_inc = rr_parameters_inc + 9
                rr_states_inc = rr_states_inc + 4

            end select

            ! Routing module
            select case (setup%routing_module)

                ! 'lag0' module
            case ("lag0")

                call lag0_time_step( &
                    setup, &
                    mesh, &
                    options, &
                    checkpoint_variable%ac_qtz, &
                    checkpoint_variable%ac_qz)

                ! 'lr' module
            case ("lr")

                ! % To avoid potential aliasing tapenade warning (DF02)
                h1 = checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) ! % hlr

                call lr_time_step( &
                    setup, &
                    mesh, &
                    options, &
                    checkpoint_variable%ac_qtz, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % llr
                    h1, & ! % hlr
                    checkpoint_variable%ac_qz)

                checkpoint_variable%ac_rr_states(:, rr_states_inc + 1) = h1

                rr_parameters_inc = rr_parameters_inc + 1
                rr_states_inc = rr_states_inc + 1

                ! 'kw' module
            case ("kw")

                call kw_time_step( &
                    setup, &
                    mesh, &
                    options, &
                    checkpoint_variable%ac_qtz, &
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 1), & ! % akw
                    checkpoint_variable%ac_rr_parameters(:, rr_parameters_inc + 2), & ! % bkw
                    checkpoint_variable%ac_qz)

                rr_parameters_inc = rr_parameters_inc + 1

            end select

            call store_time_step(setup, mesh, output, returns, checkpoint_variable, t)

        end do

    end subroutine simulation_checkpoint

    subroutine simulation(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        integer :: ncheckpoint, checkpoint_size, i, start_time_step, end_time_step
        type(Checkpoint_VariableDT) :: checkpoint_variable

        ! % We use checkpoints to reduce the maximum memory usage of the adjoint model.
        ! % Without checkpoints, the maximum memory required is equal to K * T, where K in [0, +inf] is the
        ! % memory used at each time step and T in [1, +inf] the total number of time steps.
        ! % With checkpoints, the maximum memory required is equal to (K * C) + (K * T/C), where C in [1, T]
        ! % is the number of checkpoints.
        ! % Finding out what value of C minimizes it, C must be equal to the square root of the
        ! % number of time steps T. (K * C) + (K * T/C) becomes 2K * √T.
        ! % Therefore, the memory gain is equivalent to M = 1 - 2/√T
        ! % T = [1, 4, 1e1, 1e2, 1e3, 1e4] -> M = [-1, 0, 0.37, 0.8, 0.93, 0.98]
        ncheckpoint = int(sqrt(real(setup%ntime_step, sp)))
        checkpoint_size = setup%ntime_step/ncheckpoint

        ! % Allocate checkpoint variables
        allocate (checkpoint_variable%ac_rr_parameters(mesh%nac, setup%nrrp))
        allocate (checkpoint_variable%ac_rr_states(mesh%nac, setup%nrrs))
        allocate (checkpoint_variable%ac_mlt(mesh%nac))
        allocate (checkpoint_variable%ac_qtz(mesh%nac, setup%nqz))
        allocate (checkpoint_variable%ac_qz(mesh%nac, setup%nqz))

        ! % Initialize checkpoint fluxes
        checkpoint_variable%ac_mlt = 0._sp
        checkpoint_variable%ac_qtz = 0._sp
        checkpoint_variable%ac_qz = 0._sp

        ! % Initialize checkpoint rainfall-runoff parameters
        do i = 1, setup%nrrp

            call matrix_to_ac_vector(mesh, parameters%rr_parameters%values(:, :, i), &
            & checkpoint_variable%ac_rr_parameters(:, i))

        end do

        ! % Initialize checkpoint rainfall-runoff states
        do i = 1, setup%nrrs

            call matrix_to_ac_vector(mesh, parameters%rr_initial_states%values(:, :, i), &
            & checkpoint_variable%ac_rr_states(:, i))

        end do

        ! % Checkpoints loop
        do i = 1, ncheckpoint

            start_time_step = (i - 1)*checkpoint_size + 1
            end_time_step = i*checkpoint_size

            if (i .eq. ncheckpoint) end_time_step = setup%ntime_step

            call simulation_checkpoint(setup, mesh, input_data, parameters, output, options, returns, &
            & checkpoint_variable, start_time_step, end_time_step)

        end do

        !$AD start-exclude
        ! % Store last rainfall-runoff states
        do i = 1, setup%nrrs

            call ac_vector_to_matrix(mesh, checkpoint_variable%ac_rr_states(:, i), &
            & output%rr_final_states%values(:, :, i))

        end do
        !$AD end-exclude

    end subroutine simulation

end module md_simulation
