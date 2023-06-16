!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%      
!%      - forward_run
!%

module mw_forward

    use md_constant, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_output, only: OutputDT

    implicit none

contains

    subroutine forward_run(setup, mesh, input_data, parameters, output)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output

        call base_forward_run(setup, mesh, input_data, parameters, output)

    end subroutine forward_run

end module mw_forward
