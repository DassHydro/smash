subroutine base_forward_run(setup, mesh, input_data, parameters, output, options, returns)

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use mwd_parameters_manipulation !% only: control_to_parameters
    use md_forward_structure !% only: gr4_lr_forward, gr4_kw_forward, gr5_lr_forward, gr5_kw_forward, &
    !& loieau_lr_forward, grd_lr_forward
    use mwd_cost !% only: compute_cost

    implicit none

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(inout) :: parameters
    type(OutputDT), intent(inout) :: output
    type(OptionsDT), intent(in) :: options
    type(ReturnsDT), intent(inout) :: returns

    real(sp), dimension(mesh%nrow, mesh%ncol, setup%nos) :: rr_states_buffer_values

    !% Map control to parameters
    call control_to_parameters(setup, mesh, input_data, parameters, options)

    !% Save initial states
    rr_states_buffer_values = parameters%rr_initial_states%values

    !% Simulation
    select case (setup%structure)

    case ("gr4-lr")

        call gr4_lr_forward(setup, mesh, input_data, parameters, output, options, returns)

    case ("gr4-kw")

        call gr4_kw_forward(setup, mesh, input_data, parameters, output, options, returns)

    case ("gr5-lr")

        call gr5_lr_forward(setup, mesh, input_data, parameters, output, options, returns)

    case ("gr5-kw")

        call gr5_kw_forward(setup, mesh, input_data, parameters, output, options, returns)

    case ("loieau-lr")

        call loieau_lr_forward(setup, mesh, input_data, parameters, output, options, returns)

    case ("grd-lr")

        call grd_lr_forward(setup, mesh, input_data, parameters, output, options, returns)

    end select

    !% Assign final states and reset initial states
    output%rr_final_states%values = parameters%rr_initial_states%values
    parameters%rr_initial_states%values = rr_states_buffer_values

    !% Compute cost
    call compute_cost(setup, mesh, input_data, parameters, output, options, returns)

end subroutine base_forward_run
