!%      This module `mw_run` encapsulates all SMASH run.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1] forward_run
!%      [2] adjoint_run
!%      [3] tangent_linear_run

module mw_run
    
    use mwd_common !% only: sp, dp, lchar, np, np
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !%  only: ParametersDT, parameters_to_matrix
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    implicit none
    
    contains

!%      TODO comment
        !% Calling forward from forward/forward.f90
        subroutine forward_run(setup, mesh, input_data, parameters, states, output)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            real(sp) :: cost
            
            write(*,*) ">>> Forward Model M (k)"
            
            call forward(setup, mesh, input_data, parameters, states, output, cost)

        end subroutine forward_run
        
!%      TODO comment
        !% Calling forward_b from forward/forward_b.f90
        subroutine adjoint_run(setup, mesh, input_data, parameters, states, output)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_b
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_b_matrix
            type(StatesDT) :: states_b
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            
            call ParametersDT_initialise(parameters_b, setup, mesh)
            call StatesDT_initialise(states_b, setup, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            write(*,*) ">>> Adjoint Model (dM/dk)* (k)"
        
            cost_b = 1._sp
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, states, states_b, output, output_b, cost, cost_b)
            
            call parameters_to_matrix(parameters_b, parameters_b_matrix)
            
            if (.not. allocated(output%parameters_gradient)) then
            
                allocate(output%parameters_gradient(mesh%nrow, mesh%ncol, np))
                
            end if
            
            output%parameters_gradient = parameters_b_matrix
            
        end subroutine adjoint_run
        
!%      TODO comment and clean
        !% Calling forward_d from forward/forward_d.f90
        subroutine tangent_linear_run(setup, mesh, input_data, parameters, states, output)
        
        implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_d
            type(StatesDT) :: states_d
            type(OutputDT) :: output_d
            real(sp) :: cost, cost_d
            
            write(*,*) ">>> Tangent Linear Model (dM/dk) (k)"
            
            call forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, states, states_d, output, output_d, cost, cost_d)
            
        end subroutine tangent_linear_run
        
end module mw_run
