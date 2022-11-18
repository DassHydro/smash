!%      This module `mw_copy` encapsulates all SMASH copy.
!%      This module is wrapped

module mw_copy

    use md_constant, only: sp, dp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_states, only: StatesDT
    use mwd_output, only: OutputDT
    
    contains

        subroutine copy_setup(a, b)
        
            !% Notes
            !% -----
            !%
            !% SetupDT copy subroutine
        
            implicit none
            
            type(SetupDT), intent(in) :: a
            type(SetupDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_setup


        subroutine copy_mesh(a, b)
        
            !% Notes
            !% -----
            !%
            !% MeshDT copy subroutine
        
            implicit none
            
            type(MeshDT), intent(in) :: a
            type(MeshDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_mesh


        subroutine copy_input_data(a, b)
        
            !% Notes
            !% -----
            !%
            !% Input_DataDT copy subroutine
        
            implicit none
            
            type(Input_DataDT), intent(in) :: a
            type(Input_DataDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_input_data


        subroutine copy_parameters(a, b)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT copy subroutine
        
            implicit none
            
            type(ParametersDT), intent(in) :: a
            type(ParametersDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_parameters


        subroutine copy_states(a, b)
        
            !% Notes
            !% -----
            !%
            !% StatesDT copy subroutine
        
            implicit none
            
            type(StatesDT), intent(in) :: a
            type(StatesDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_states
        
        
        subroutine copy_output(a, b)
        
            !% Notes
            !% -----
            !%
            !% OutputDT copy subroutine
        
            implicit none
            
            type(OutputDT), intent(in) :: a
            type(OutputDT), intent(out) :: b
            
            b = a
        
        end subroutine copy_output


end module mw_copy
