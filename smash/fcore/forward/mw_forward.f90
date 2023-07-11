!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - forward_run
!%      - forward_run_b

module mw_forward

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_output, only: OutputDT
    use mwd_options, only: OptionsDT
    use mwd_returns, only: ReturnsDT

    implicit none

contains

    subroutine forward_run(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        call base_forward_run(setup, mesh, input_data, parameters, output, options, returns)

    end subroutine forward_run

    subroutine forward_run_b(setup, mesh, input_data, parameters, parameters_b, output, output_b, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters, parameters_b
        type(OutputDT), intent(inout) :: output, output_b
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        call base_forward_run_b(setup, mesh, input_data, parameters, parameters_b, output, output_b, options, returns)

    end subroutine forward_run_b

end module mw_forward
