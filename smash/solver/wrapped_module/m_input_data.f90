!%    This module `m_data` encapsulates all SMASH data (type, subroutines, functions)
module m_input_data

    use m_common, only: sp, dp, lchar
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    
    implicit none
    
    !%      Input_DataDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================

    !%      ====================    ==========================================================
    
    type Input_DataDT
    
        real(sp), dimension(:,:), allocatable :: qobs
        real(sp), dimension(:,:,:), allocatable :: prcp
        real(sp), dimension(:,:,:), allocatable :: pet
        
        real(sp), dimension(:,:), allocatable :: prcp_sparse
        real(sp), dimension(:,:), allocatable :: pet_sparse
    
    end type Input_DataDT
    
    contains
    
        subroutine Input_DataDT_initialise(input_data, setup, mesh)
        
            implicit none
            
            type(Input_DataDT), intent(inout) :: input_data
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            
            integer :: n

            if (.not. setup%simulation_only) then
            
                allocate(input_data%qobs(mesh%ng, setup%ntime_step))
                input_data%qobs = -99._sp
                
            end if
            
            if (setup%sparse_forcing) then
            
                n = count(mesh%global_active_cell .eq. 1)
            
                allocate(input_data%prcp_sparse(n, setup%ntime_step))
                input_data%prcp_sparse = -99._sp
                allocate(input_data%pet_sparse(n, setup%ntime_step))
                input_data%pet_sparse = -99._sp
                
            else
            
                allocate(input_data%prcp(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%prcp = -99._sp
                allocate(input_data%pet(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%pet = -99._sp
            
            end if
            
        end subroutine Input_DataDT_initialise
    
    

end module m_input_data
