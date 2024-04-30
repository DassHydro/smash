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
    use md_simulation !% only: simulation
    use mwd_cost !% only: compute_cost

    implicit none

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(inout) :: parameters
    type(OutputDT), intent(inout) :: output
    type(OptionsDT), intent(in) :: options
    type(ReturnsDT), intent(inout) :: returns

    !% Map control to parameters
    call control_to_parameters(setup, mesh, input_data, parameters, options)

    !% Simulation
    call simulation(setup, mesh, input_data, parameters, output, options, returns)

    !% Compute cost
    call compute_cost(setup, mesh, input_data, parameters, output, options, returns)

end subroutine base_forward_run
