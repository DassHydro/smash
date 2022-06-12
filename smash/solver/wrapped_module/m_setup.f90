!%    This module `m_setup` encapsulates all SMASH setup (type, subroutines, functions)
module m_setup
    
    use m_common, only: dp, lchar
    
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
    
        real(dp) :: dt = 3600._dp
        real(dp) :: dx = 1000._dp
        
        character(lchar) :: start_time = "..."
        character(lchar) :: end_time = "..."
        character(lchar) :: optim_start_time = "..."
        
        integer :: nb_time_step = 0
        integer :: optim_start_step = 1
        
        logical :: active_cell_only = .true.
        logical :: simulation_only = .false.
        logical :: sparse_forcing = .false.
        
        logical :: read_qobs = .true.
        character(lchar) :: qobs_directory = "..."
        
    end type SetupDT

end module m_setup
