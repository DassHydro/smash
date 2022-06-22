module m_run

    use m_common, only: sp, dp, lchar, np, ns
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    use m_input_data, only: Input_DataDT
    use m_parameters, only: ParametersDT
    use m_states, only: StatesDT
    use m_output, only: OutputDT
    
    implicit none
    
    contains
    
        subroutine run_direct(setup, mesh, input_data, parameters, states, output, cost)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(in) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
        
            cost = sum(parameters%cp)
        
        end subroutine run_direct
    
!~     subroutine run_ADJ()
    
!~     end subroutine run_ADJ
    
!~     subroutine run_TLM()
    
!~     end subroutine run_TLM


end module m_run
