!%    This module `m_data` encapsulates all SMASH data (type, subroutines, functions)
module m_input_data

    use m_common, only: dp, lchar
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
    
        real(dp), dimension(:,:), allocatable :: qobs
        real(dp), dimension(:,:,:), allocatable :: prcp
        real(dp), dimension(:,:,:), allocatable :: pet
        
        real(dp), dimension(:,:), allocatable :: prcp_sparse
        real(dp), dimension(:,:), allocatable :: pet_sparse
    
    end type Input_DataDT
    
    contains
    
        subroutine Input_DataDT_initialise(input_data, setup, mesh)
        
            implicit none
            
            type(Input_DataDT), intent(inout) :: input_data
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            
            integer :: n
            
            if (.not. setup%simulation_only) then
            
                allocate(input_data%qobs(mesh%ng, setup%nb_time_step))
                input_data%qobs = -99._dp
                
            end if
            
            if (setup%sparse_forcing) then
            
                n = count(mesh%global_active_cell .eq. 1)
            
                allocate(input_data%prcp_sparse(n, setup%nb_time_step))
                input_data%prcp_sparse = -99._dp
                allocate(input_data%pet_sparse(n, setup%nb_time_step))
                input_data%pet_sparse = -99._dp
                
            
            else
            
                allocate(input_data%prcp(mesh%nrow, mesh%ncol, &
                & setup%nb_time_step))
                input_data%prcp = -99._dp
                allocate(input_data%pet(mesh%nrow, mesh%ncol, &
                & setup%nb_time_step))
                input_data%pet = -99._dp
            
            end if
            
        end subroutine Input_DataDT_initialise
    
    

end module m_input_data
