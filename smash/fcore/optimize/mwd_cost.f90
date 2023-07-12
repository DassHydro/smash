!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - compute_cost

module mwd_cost

    use md_constant !% only: sp
    use mwd_efficiency_metric !% any type
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT

    implicit none

contains

    subroutine compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        real(sp) :: jobs
        real(sp), dimension(setup%ntime_step) :: qo, qs

        jobs = 0._sp

        qo = input_data%obs_response%q(1, :)
        qs = output%sim_response%q(1, :)

        jobs = nse(qo, qs)

        output%cost = jobs

    end subroutine compute_cost

end module mwd_cost
