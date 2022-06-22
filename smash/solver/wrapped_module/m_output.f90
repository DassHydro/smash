!%    This module `m_output` encapsulates all SMASH output
module m_output
    
    use m_common, only: sp, dp, lchar, np, ns
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    
    
    implicit none
    
    public :: OutputDT
    
    !%      OutputDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================
    !%      ====================    ==========================================================
    
    type :: OutputDT
    
        real(sp), dimension(:,:), allocatable :: qsim
        real(sp), dimension(:,:,:), allocatable :: qsim_grid
        
        real(sp), dimension(:,:,:), allocatable :: gradient_parameters
        
        real(sp) :: cost
        
    end type OutputDT
    
    contains
    
        subroutine OutputDT_initialise(output, setup, mesh)
        
            implicit none
            
            type(OutputDT), intent(inout) :: output
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            
            allocate(output%qsim(mesh%ng, setup%ntime_step))
            output%qsim = - 99._sp
            
            if (.not. setup%active_cell_only) then
            
                allocate(output%qsim_grid(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                output%qsim_grid = - 99._sp
            
            end if
            
            if (.not. setup%simulation_only) then
            
                allocate(output%gradient_parameters(mesh%nrow, &
                & mesh%ncol, np))
                output%gradient_parameters = 0._sp
            
            end if
            
            output%cost = - 99._sp
        
        end subroutine OutputDT_initialise

end module m_output
