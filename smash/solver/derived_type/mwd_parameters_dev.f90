!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!% 
!%      - ParametersDT_dev
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``control``                ControlDT_dev
!%          ``opr_parameters``         Opr_ParametersDT_dev
!%          ``opr_initial_states``     Opr_StatesDT_dev
!%
!§      Subroutine
!%      ----------
!%
!%      - ParametersDT_dev_initialise
!%      - ParametersDT_dev_copy

module mwd_parameters_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev
    use mwd_control_dev !% only: ControlDT_dev
    use mwd_opr_parameters_dev !% only: Opr_ParametersDT_dev, Opr_ParametersDT_dev_initialise
    use mwd_opr_states_dev !% only: Opr_StatesDT_dev, Opr_StatesDT_dev_initialise
    
    implicit none
   
    type ParametersDT_dev
    
        type(ControlDT_dev) :: control
        type(Opr_ParametersDT_dev) :: opr_parameters
        type(Opr_StatesDT_dev) :: opr_initial_states
    
    end type ParametersDT_dev

contains

    subroutine ParametersDT_dev_initialise(this, setup, mesh)
    
        implicit none
        
        type(ParametersDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
        call Opr_ParametersDT_dev_initialise(this%opr_parameters, setup, mesh)
        call Opr_StatesDT_dev_initialise(this%opr_initial_states, setup, mesh)
    
    end subroutine ParametersDT_dev_initialise
    
    subroutine ParametersDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(ParametersDT_dev), intent(in) :: this
        type(ParametersDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine ParametersDT_dev_copy
    
end module mwd_parameters_dev
