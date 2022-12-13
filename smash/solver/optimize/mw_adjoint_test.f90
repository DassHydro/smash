!%      This module `mw_adjoint_test` encapsulates all SMASH adjoint_test.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1] scalar_product_test
!%      [2] gradient_test

module mw_adjoint_test
    
    use md_constant, only: sp, GNP, GNS
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT, ParametersDT_initialise
    use mwd_states, only: StatesDT, StatesDT_initialise
    use mwd_output, only: OutputDT, OutputDT_initialise
    use mw_forward, only: forward, forward_b, forward_d
    use mwd_parameters_manipulation, only: get_parameters, set_parameters
    use mwd_states_manipulation, only: get_states, set_states
    
    implicit none
    
    contains
    
        subroutine scalar_product_test(setup, mesh, input_data, parameters, states, output)
            
            implicit none
            
            !% Notes
            !% -----
            !%
            !% Scalar Product Test subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it returns the results of scalar product test.
            !% sp1 = <dY*, dY>
            !% sp2 = <dk*, dk>
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd, parameters_d, parameters_b
            type(StatesDT) ::  states_bgd, states_d, states_b
            type(OutputDT) ::  output_d, output_b
            real(sp) :: cost, cost_d, cost_b
            real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: parameters_d_matrix, parameters_b_matrix
            real(sp) :: sp1, sp2

            write(*,'(a)') "</> Scalar Product Test"
            
            parameters_bgd = parameters
            states_bgd = states
            
            call ParametersDT_initialise(parameters_d, mesh)
            call ParametersDT_initialise(parameters_b, mesh)
            
            call StatesDT_initialise(states_d, mesh)
            call StatesDT_initialise(states_b, mesh)
            
            call OutputDT_initialise(output_d, setup, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            call set_parameters(mesh, parameters_d, 1._sp)
            call set_states(mesh, states_d, 0._sp)
            
            write(*,'(4x,a)') "Tangent Linear Model dY  = (dM/dk) (k) . dk"
            
            call forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, parameters_bgd, states, states_d, states_bgd, &
            & output, output_d, cost, cost_d)
            
            cost_b = 1._sp
            
            write(*,'(4x,a)') "Adjoint Model        dk* = (dM/dk)* (k) . dY*"
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, parameters_bgd, states, states_b, states_bgd, &
            & output, output_b, cost, cost_b)
            
            call get_parameters(mesh, parameters_b, parameters_b_matrix)
            call get_parameters(mesh, parameters_d, parameters_d_matrix)
            
            ! No perturbation has been set to states_d
            sp1 = cost_b * cost_d
            
            sp2 = sum(parameters_b_matrix * parameters_d_matrix)
            
            write(*,'(4x,a,f12.8)') "<dY*, dY> (sp1) = ", sp1
            write(*,'(4x,a,f12.8)') "<dk*, dk> (sp2) = ", sp2
            write(*,'(4x,a,f12.8)') "Relative Error  = ", (sp1 - sp2) / sp1
        
        end subroutine scalar_product_test
        
        
        !% WIP
        subroutine gradient_test(setup, mesh, input_data, parameters, states, output)
        
            !% Notes
            !% -----
            !%
            !% Gradient Test subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it returns the results of gradient test.
            !% Ia = (Yadk - Y) / (a dk* . dk)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd, parameters_b
            type(StatesDT) :: states_bgd, states_b
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: parameters_matrix, parameters_b_matrix
            real(sp) :: yk, yadk, dk, an, ian
            integer :: n
            
            write(*,'(a)') "</> Gradient Test"
            
            parameters_bgd = parameters
            states_bgd = states
            dk = 1._sp
            
            call ParametersDT_initialise(parameters_b, mesh)
            call StatesDT_initialise(states_b, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            write(*,'(4x,a)') "Forward Model Y    = M (k)"
            
            call forward(setup, mesh, input_data, parameters, &
            & parameters_bgd, states, states_bgd, output, cost)
            
            yk = cost
            
            cost_b = 1._sp
            
            write(*,'(4x,a)') "Adjoint Model dk*  = (dM/dk)* (k) . dY*"
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, parameters_bgd, states, states_b, states_bgd, &
            & output, output_b, cost, cost_b)
            
            call get_parameters(mesh, parameters_b, parameters_b_matrix)
            
            write(*,'(4x,a)') "Forward Model Yadk = M (k + a dk)"
            
            do n=1, 16
            
                an = 2._sp ** (- (n - 1))
            
                call get_parameters(mesh, parameters_bgd, parameters_matrix)
                
                parameters_matrix = parameters_matrix + an * dk
                
                call set_parameters(mesh, parameters, parameters_matrix)
                
                call forward(setup, mesh, input_data, parameters, &
                & parameters_bgd, states, states_bgd, output, cost)
                
                yadk = cost
                
                ian = (yadk - yk) / (an * sum(parameters_b_matrix * dk)) 
                
                write(*,'(4x,a,f12.8,4x,a,f12.8)') "a = ", an, "|Ia - 1| = ", abs(ian - 1)
                
            end do
            
            write(*,'(4x,a)') "Ia = (Yadk - Y) / (a dk* . dk)"
        
        end subroutine gradient_test

end module mw_adjoint_test
