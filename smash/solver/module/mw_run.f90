!%    This module (wrap) `mw_run` encapsulates all SMASH run (type, subroutines, functions)
module mw_run
    
    use m_common !% only: sp, dp, lchar, np, np
    use mw_setup !% only: SetupDT
    use mw_mesh  !% only: MeshDT
    use mw_input_data !% only: Input_DataDT
    use mw_parameters !%  only: ParametersDT, parameters_derived_type_to_matrix
    use mw_states !% only: StatesDT
    use mw_output !% only: OutputDT
    
    implicit none
    
    contains
        
        !% Calling forward from forward/forward.f90
        subroutine forward_run(setup, mesh, input_data, parameters, states, output, cost)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            call forward(setup, mesh, input_data, parameters, states, output, cost)
        
        end subroutine forward_run
        
        !% Calling forward_b from forward/forward_b.f90
        subroutine adjoint_run(setup, mesh, input_data, parameters, states, output, cost)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            type(ParametersDT) :: parameters_b
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_b_matrix
            type(StatesDT) :: states_b
            type(OutputDT) :: output_b
            real(sp) :: cost_b
            
            call ParametersDT_initialise(parameters_b, setup, mesh)
            call StatesDT_initialise(states_b, setup, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            cost_b = 1._sp
            cost = 0._sp
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, states, states_b, output, output_b, cost, cost_b)
            
            call parameters_derived_type_to_matrix(parameters_b, parameters_b_matrix)
            
            output%parameters_gradient = parameters_b_matrix
            
        end subroutine adjoint_run
        
        !% Calling forward_d from forward/forward_d.f90
        !% TODO
        subroutine tangent_linear_run(setup, mesh, input_data, parameters, states, output, cost)
        
        implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            type(ParametersDT) :: parameters_d
            type(StatesDT) :: states_d
            type(OutputDT) :: output_d
            real(sp) :: cost_d
            
            call forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, states, states_d, output, output_d, cost, cost_d)
            
        end subroutine tangent_linear_run
        
end module mw_run
