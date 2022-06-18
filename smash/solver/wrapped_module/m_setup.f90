!%    This module `m_setup` encapsulates all SMASH setup
module m_setup
    
    use m_common, only: sp, dp, lchar
    
    implicit none
    
    public :: SetupDT
    
    !%      SetupDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================
    !%      ``start_time``          Start time of simulation in %Y%m%d%H%M format
    !%      ``end_time``            End time of simulation in %Y%m%d%H%M format
    !%      ``optim_start_time``    Start time of optimization in %Y%m%d%H%M format
    !%      ``dt``                  Time step in [s]
    !%      ``dx``                  Spatial step in [m] (square cell)
    !%      ``nb_time_step``        Number of time step
    !%      ``optim_start_step``    Optimization start step
    !%      ====================    ==========================================================
    
    type :: SetupDT
    
        real(sp) :: dt = 3600._sp
        real(sp) :: dx = 1000._sp
        
        character(lchar) :: start_time = "..."
        character(lchar) :: end_time = "..."
        character(lchar) :: optim_start_time = "..."
        
        integer :: ntime_step = 0
        integer :: optim_start_step = 1
        
        logical :: active_cell_only = .true.
        logical :: simulation_only = .false.
        logical :: sparse_storage = .false.
        
        logical :: read_qobs = .true.
        character(lchar) :: qobs_directory = "..."
        
        logical :: read_prcp = .true.
        character(lchar) :: prcp_format = "tiff"
        real(sp) :: prcp_conversion_factor = 1._sp
        character(lchar) :: prcp_directory = "..."
        
        logical :: read_pet = .true.
        character(lchar) :: pet_format = "tiff"
        real(sp) :: pet_conversion_factor = 1._sp
        character(lchar) :: pet_directory = "..."
        logical :: daily_interannual_pet = .false.
        
    end type SetupDT

end module m_setup
