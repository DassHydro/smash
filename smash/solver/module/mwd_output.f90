!%      This module `mwd_output` encapsulates all SMASH output.
!%      This module is wrapped and differentiated.

!%      OutputDT type:
!%      
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``qsim``                 Simulated discharge at gauge            [m3/s]
!%      ``qsim_domain``          Simulated discharge whole domain        [m3/s]
!%      ``sparse_qsim_domain``   Sparse simulated discharge whole domain [m3/s]
!%      ``parameters_gradient``  Parameters gradients
!%      ======================== =======================================
module mwd_output
    
    use mwd_common !% only: sp, dp, lchar, np, ns
    use mwd_setup  !% only: SetupDT
    use mwd_mesh   !%only: MeshDT
    
    implicit none
    
    type :: OutputDT
    
        real(sp), dimension(:,:), allocatable :: qsim
        real(sp), dimension(:,:,:), allocatable :: qsim_domain
        real(sp), dimension(:,:), allocatable :: sparse_qsim_domain
        
        real(sp), dimension(:,:,:), allocatable :: parameters_gradient
        
    end type OutputDT
    
    contains
    
        subroutine OutputDT_initialise(output, setup, mesh)
        
            implicit none
            
            type(OutputDT), intent(inout) :: output
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            
            allocate(output%qsim(mesh%ng, setup%ntime_step))
            output%qsim = - 99._sp
            
            if (setup%sparse_storage) then
            
                allocate(output%sparse_qsim_domain(mesh%nac, &
                & setup%ntime_step))
                output%sparse_qsim_domain = - 99._sp
                
            else

                allocate(output%qsim_domain(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                output%qsim_domain = - 99._sp
            
            end if
        
        end subroutine OutputDT_initialise
        

!%      TODO comment 
        subroutine output_copy(output_in, &
        & output_out)
            
            implicit none
            
            type(OutputDT), intent(in) :: output_in
            type(OutputDT), intent(out) :: output_out
            
            output_out = output_in
        
        end subroutine output_copy

end module mwd_output
