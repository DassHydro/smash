!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - OutputDT_dev
!%
!%      Subroutine
!%      ----------
!%
!%      - OutputDT_dev_initialise
!%      - OutputDT_dev_copy

module mwd_output_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev
    use mwd_response_dev !% only: ResponseDT_dev, ResponseDT_dev_initialise
    use mwd_opr_states_dev !% only: Opr_StatesDT_dev, Opr_StatesDT_dev_initialise
    
    implicit none
    
    type OutputDT_dev
        
        type(ResponseDT_dev) :: sim_response
        type(Opr_StatesDT_dev) :: opr_final_states
    
    end type OutputDT_dev
    
contains

    subroutine OutputDT_dev_initialise(this, setup, mesh)
    
        implicit none
        
        type(OutputDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
        call ResponseDT_dev_initialise(this%sim_response, setup, mesh)
        call Opr_StatesDT_dev_initialise(this%opr_final_states, setup, mesh)
    
    end subroutine OutputDT_dev_initialise
    
        subroutine OutputDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(OutputDT_dev), intent(in) :: this
        type(OutputDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine OutputDT_dev_copy

end module mwd_output_dev
