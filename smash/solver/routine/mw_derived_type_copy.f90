!%      This module `mw_derived_type_copy` encapsulates all SMASH derived_type_copy.
!%      This module is wrapped

module mw_derived_type_copy

    use md_constant, only: sp, dp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_states, only: StatesDT
    use mwd_output, only: OutputDT
    
    implicit none
    
    contains

        subroutine copy_setup(this, copy)
        
            !% Notes
            !% -----
            !%
            !% SetupDT copy subroutine
        
            implicit none
            
            type(SetupDT), intent(in) :: this
            type(SetupDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_setup


        subroutine copy_mesh(this, copy)
        
            !% Notes
            !% -----
            !%
            !% MeshDT copy subroutine
        
            implicit none
            
            type(MeshDT), intent(in) :: this
            type(MeshDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_mesh


        subroutine copy_input_data(this, copy)
        
            !% Notes
            !% -----
            !%
            !% Input_DataDT copy subroutine
        
            implicit none
            
            type(Input_DataDT), intent(in) :: this
            type(Input_DataDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_input_data


        subroutine copy_parameters(this, copy)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT copy subroutine
        
            implicit none
            
            type(ParametersDT), intent(in) :: this
            type(ParametersDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_parameters


        subroutine copy_states(this, copy)
        
            !% Notes
            !% -----
            !%
            !% StatesDT copy subroutine
        
            implicit none
            
            type(StatesDT), intent(in) :: this
            type(StatesDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_states
        
        
        subroutine copy_output(this, copy)
        
            !% Notes
            !% -----
            !%
            !% OutputDT copy subroutine
        
            implicit none
            
            type(OutputDT), intent(in) :: this
            type(OutputDT), intent(out) :: copy
            
            copy = this
        
        end subroutine copy_output


end module mw_derived_type_copy
