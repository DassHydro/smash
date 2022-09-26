!%      This module `mw_run` encapsulates all SMASH run.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1] forward_run
!%      [2] adjoint_run
!%      [3] tangent_linear_run

module mw_run
    
    use mwd_common, only: sp, np, ns
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT, ParametersDT_initialise, &
    & parameters_to_matrix, set0_parameters, set1_parameters
    use mwd_states, only: StatesDT, StatesDT_initialise, &
    & states_to_matrix, set0_states, set1_states
    use mwd_output, only: OutputDT, OutputDT_initialise
    
    implicit none
    
    contains

        subroutine forward_run(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, verbose)
        
            !% Notes
            !% -----
            !%
            !% Forward run subroutine
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, 
            !% ParametersDT_bgd, StatesDT, StatesDT_bgd, OutputDT and logical verbose value,
            !% it computes a forward run. Verbose argument allow to print:
            !% "</> Forward Model M (k)"
            !%
            !% Calling forward from forward/forward.f90  Y = M (k)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters, parameters_bgd
            type(StatesDT), intent(inout) :: states, states_bgd
            type(OutputDT), intent(inout) :: output
            logical, intent(in) :: verbose
            
            real(sp) :: cost
                
            if (verbose) write(*,'(a)') "</> Forward Model M (k)"
            
            call forward(setup, mesh, input_data, parameters, &
            & parameters_bgd, states, states_bgd, output, cost)
            
            states = states_bgd

        end subroutine forward_run
        
        !% WIP
        subroutine adjoint_run(setup, mesh, input_data, parameters, states, output)
        
            !% Notes
            !% -----
            !%
            !% Forward run subroutine
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it computes an adjoint run.
            !%
            !% Calling forward_b from forward/forward_b.f90 dk* = (dM/dk)* (k) . dY*
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd, parameters_b
            type(StatesDT) :: states_bgd
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_b_matrix
            type(StatesDT) :: states_b
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            
            write(*,'(a)') "</> Adjoint Model (dM/dk)* (k)"
            
            call ParametersDT_initialise(parameters_b, mesh)
            call StatesDT_initialise(states_b, mesh)
            call OutputDT_initialise(output_b, setup, mesh)

            cost_b = 1._sp
            parameters_bgd = parameters
            states_bgd = states
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, parameters_bgd, states, states_b, states_bgd, &
            & output, output_b, cost, cost_b)
            
            states = states_bgd
            
            call parameters_to_matrix(parameters_b, parameters_b_matrix)
            
            if (.not. allocated(output%parameters_gradient)) then
            
                allocate(output%parameters_gradient(mesh%nrow, mesh%ncol, np))
                
            end if
            
            output%parameters_gradient = parameters_b_matrix
            
        end subroutine adjoint_run
        
        !% WIP
        subroutine tangent_linear_run(setup, mesh, input_data, parameters, states, output)
        
            !% Notes
            !% -----
            !%
            !% Forward run subroutine
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it computes an tangent linear run.
            !%
            !% Calling forward_d from forward/forward_d.f90 dY = (dM/dk) (k) . dk
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd, parameters_d
            type(StatesDT) :: states_bgd, states_d
            type(OutputDT) :: output_d
            real(sp) :: cost, cost_d
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_d_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, ns) :: states_d_matrix
            
            write(*,'(a)') "</> Tangent Linear Model (dM/dk) (k)"
            
            call ParametersDT_initialise(parameters_d, mesh)
            call StatesDT_initialise(states_d, mesh)
            call OutputDT_initialise(output_d, setup, mesh)
            
            call set1_parameters(parameters_d)
            call set0_states(states_d)
            
            parameters_bgd = parameters
            states_bgd = states
            
            call forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, parameters_bgd, states, states_d, states_bgd, &
            & output, output_d, cost, cost_d)
            
            states = states_bgd
            
        end subroutine tangent_linear_run
        
end module mw_run
