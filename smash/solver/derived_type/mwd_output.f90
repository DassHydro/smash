!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - OutputDT
!%
!%      Subroutine
!%      ----------
!%
!%      - OutputDT_initialise
!%      - OutputDT_copy

module mwd_output

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_response !% only: ResponseDT, ResponseDT_initialise
    use mwd_opr_states !% only: Opr_StatesDT, Opr_StatesDT_initialise
    
    implicit none
    
    type OutputDT
        
        type(ResponseDT) :: sim_response
        type(Opr_StatesDT) :: opr_final_states
    
    end type OutputDT
    
contains

    subroutine OutputDT_initialise(this, setup, mesh)
    
        implicit none
        
        type(OutputDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        
        call ResponseDT_initialise(this%sim_response, setup, mesh)
        call Opr_StatesDT_initialise(this%opr_final_states, setup, mesh)
    
    end subroutine OutputDT_initialise
    
        subroutine OutputDT_copy(this, this_copy)
    
        implicit none
        
        type(OutputDT), intent(in) :: this
        type(OutputDT), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine OutputDT_copy

end module mwd_output
