!%      This module `mw_forward` encapsulates all SMASH routine.
!%      This module is wrapped

module mw_forward
        
    use md_constant, only: sp, dp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT, Hyper_ParametersDT
    use mwd_states, only: StatesDT, Hyper_StatesDT
    use mwd_output, only: OutputDT
        
    implicit none
    
    contains
    
        subroutine forward(setup, mesh, input_data, parameters, &
        & parameters_bgd, states, states_bgd, output, cost)
        
            !% Notes
            !% -----
            !%
            !% Wrapping base_forward
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters, parameters_bgd
            type(StatesDT), intent(inout) :: states, states_bgd
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost

            call base_forward(setup, mesh, input_data, parameters, &
            & parameters_bgd, states, states_bgd, output, cost)
        
        end subroutine forward

        subroutine forward_b(setup, mesh, input_data, parameters, &
        & parameters_b, parameters_bgd, states, states_b, states_bgd, &
        & output, output_b, cost, cost_b)
        
            !% Notes
            !% -----
            !%
            !% Wrapping base_forward_b
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters, parameters_b, parameters_bgd
            type(StatesDT), intent(inout) :: states, states_b, states_bgd
            type(OutputDT), intent(inout) :: output, output_b
            real(sp), intent(inout) :: cost, cost_b
            
            call base_forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, parameters_bgd, states, states_b, states_bgd, output, &
            & output_b, cost, cost_b)
        
        end subroutine forward_b

        subroutine forward_d(setup, mesh, input_data, parameters, &
        & parameters_d, parameters_bgd, states, states_d, states_bgd, &
        & output, output_d, cost, cost_d)
            
            !% Notes
            !% -----
            !%
            !% Wrapping base_forward_d
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters, parameters_d, parameters_bgd
            type(StatesDT), intent(inout) :: states, states_d, states_bgd
            type(OutputDT), intent(inout) :: output, output_d
            real(sp), intent(inout) :: cost, cost_d
            
            call base_forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, parameters_bgd, states, states_d, states_bgd, output, &
            & output_d, cost, cost_d)
        
        end subroutine forward_d

        subroutine hyper_forward(setup, mesh, input_data, parameters, &
        & hyper_parameters, hyper_parameters_bgd, states, hyper_states, &
        & hyper_states_bgd, output, cost)
        
            !% Notes
            !% -----
            !%
            !% Wrapping base_hyper_forward
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_bgd
            type(StatesDT), intent(inout) :: states
            type(Hyper_StatesDT), intent(inout) :: hyper_states, hyper_states_bgd
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            
            call base_hyper_forward(setup, mesh, input_data, parameters, hyper_parameters, &
            & hyper_parameters_bgd, states, hyper_states, hyper_states_bgd, output, cost)
        
        end subroutine hyper_forward

        subroutine hyper_forward_b(setup, mesh, input_data, &
        & parameters, parameters_b, hyper_parameters, hyper_parameters_b, &
        & hyper_parameters_bgd, states, states_b, hyper_states, hyper_states_b, &
        & hyper_states_bgd, output, output_b, cost, cost_b)
        
            !% Notes
            !% -----
            !%
            !% Wrapping base_hyper_forward_b
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(inout) :: parameters, parameters_b
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_b, hyper_parameters_bgd
            type(StatesDT), intent(inout) :: states, states_b
            type(Hyper_StatesDT), intent(inout) :: hyper_states, hyper_states_b, hyper_states_bgd
            type(OutputDT), intent(inout) :: output, output_b
            real(sp), intent(inout) :: cost, cost_b
            
            call base_hyper_forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, hyper_parameters, hyper_parameters_b, &
            & hyper_parameters_bgd, states, states_b, hyper_states, hyper_states_b, &
            & hyper_states_bgd, output, output_b, cost, cost_b)
        
        end subroutine hyper_forward_b

        subroutine hyper_forward_d(setup, mesh, input_data, &
        & parameters, parameters_d, hyper_parameters, hyper_parameters_d, hyper_parameters_bgd, &
        & states, states_d, hyper_states, hyper_states_d, hyper_states_bgd, &
        & output, output_d, cost, cost_d)
        
            !% Notes
            !% -----
            !%
            !% Wrapping base_hyper_forward_d
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(inout) :: parameters, parameters_d
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_d, hyper_parameters_bgd
            type(StatesDT), intent(inout) :: states, states_d
            type(Hyper_StatesDT), intent(inout) :: hyper_states, hyper_states_d, hyper_states_bgd
            type(OutputDT), intent(inout) :: output, output_d
            real(sp), intent(inout) :: cost, cost_d
            
            call base_hyper_forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, hyper_parameters, hyper_parameters_d, &
            & hyper_parameters_bgd, states, states_d, hyper_states, hyper_states_d, &
            & hyper_states_bgd, output, output_d, cost, cost_d)
            
        end subroutine hyper_forward_d

end module mw_forward
