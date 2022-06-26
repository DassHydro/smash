!%    This module `mw_setup` (wrap) encapsulates all SMASH setup
module mw_setup
    
    use m_common, only: sp, dp, lchar, np, ns
    
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
        
        logical :: mean_forcing = .false.
        
        real(sp), dimension(np) :: default_parameters = &
        
        & (/1._sp    ,& !% ci
        &   200._sp  ,& !% cp
        &   1000._sp ,& !% beta
        &   500._sp  ,& !% cft
        &   500._sp  ,& !% cst
        &   0.9_sp   ,& !% alpha
        &   0._sp    ,& !% exc
        &   5._sp/)     !% lr
        
        real(sp), dimension(ns) :: default_states = &
        
        & (/0.01_sp ,& !% hi
        &   0.01_sp ,& !% hp
        &   0.01_sp ,& !% hft
        &   0.01_sp ,& !% hst
        &   0.01_sp/)  !% hr
        
        integer :: interception_module = 0
        integer :: production_module = 0
        integer :: transfer_module = 0
        integer :: exchange_module = 0
        integer :: routing_module = 0
        
        integer, dimension(np) :: optim_parameters = 0
        
        real(sp), dimension(np) :: lb_parameters = &
        
        & (/1e-6_sp ,& !% ci
        &   1e-6_sp ,& !% cp
        &   1e-6_sp ,& !% beta
        &   1e-6_sp ,& !% cft
        &   1e-6_sp ,& !% cst
        &   1e-6_sp ,& !% alpha
        &   -50._sp ,& !% exc
        &   1e-6_sp/)  !% lr
        
        real(sp), dimension(np) :: ub_parameters = &
        
        & (/1e2_sp     ,& !% ci
        &   1e3_sp     ,& !% cp
        &   1e3_sp     ,& !% beta
        &   1e3_sp     ,& !% cft
        &   1e4_sp     ,& !% cst
        &   0.999999_sp ,& !% alpha
        &   50._sp      ,& !% exc
        &   1e3_sp/)      !% lr
        
        integer :: maxiter = 10
        
    end type SetupDT

end module mw_setup
