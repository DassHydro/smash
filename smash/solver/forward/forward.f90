subroutine base_forward_run(setup, mesh, input_data, parameters, output)

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_opr_states !% only: Opr_StatesDT
    use mwd_output !% only: OutputDT
    use md_forward_structure !% only: gr_a_forward

    implicit none

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(inout) :: parameters
    type(OutputDT), intent(inout) :: output

    type(Opr_StatesDT) :: opr_states_imd

    opr_states_imd = parameters%opr_initial_states

    select case (setup%structure)

    case ("gr_a")

        call gr_a_forward(setup, mesh, input_data, parameters%opr_parameters%gr_a, parameters%opr_initial_states%gr_a, output)

    end select

    output%opr_final_states = parameters%opr_initial_states
    parameters%opr_initial_states = opr_states_imd

end subroutine base_forward_run
