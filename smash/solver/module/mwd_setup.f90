!%      This module `mwd_setup` encapsulates all SMASH setup.
!%      This module is wrapped and differentiated.
!%
!%      SetupDT type:
!%      
!%      ========================== =====================================
!%      `Variables`                Description
!%      ========================== =====================================
!%      ``dt``                     Solver time step        [s]                            (default: 3600)
!%      ``start_time``             Simulation start time   [%Y%m%d%H%M]                   (default: '...')
!%      ``end_time``               Simulation end time     [%Y%m%d%H%M]                   (default: '...')
!%      ``optim_start_time``       Optimization start time [%Y%m%d%H%M]                   (default: '...')
!%      ``ntime_step``             Number of time step
!%      ``optim_start_step``       Indice start optimization
!%      ``active_cell_only``       Simulation on active cell                              (default: .true.)
!%      ``simulation_only``        Simulation only                                        (default: .false.)
!%      ``sparse_storage``         Forcing sparse storage                                 (default: .false.)
!%      ``read_qobs``              Read observed discharge                                (default: .true.)
!%      ``qobs_directory``         Observed discharge directory path                      (default: '...')
!%      ``read_prcp``              Read precipitation                                     (default: .true.)
!%      ``prcp_format``            Precipitation format                                   (default: 'tif')
!%      ``prcp_conversion_factor`` Precipitation conversion factor                        (default: 1)
!%      ``prcp_directory``         Precipiation directory path                            (default: '...')
!%      ``read_pet``               Reap potential evapotranspiration                      (default: .true.)
!%      ``pet_format``             Potential evapotranspiration format                    (default: 'tif')
!%      ``pet_conversion_factor``  Potential evapotranpisration conversion factor         (default: 1)
!%      ``pet_directory``          Potential evapotranspiration directory path            (default: '...')
!%      ``daily_interannual_pet``  Read daily interannual potential evapotranspiration    (default: .false.)
!%      ``mean_forcing``           Compute mean forcing                                   (default: .false.)
!%      ``interception_module``    Choice of interception module                          (default: 0)
!%      ``production_module``      Choice of production module                            (default: 0)
!%      ``transfer_module``        Choice of transfer module                              (default: 0)
!%      ``exchange_module``        Choise of exchange module                              (default: 0)
!%      ``routing_module``         Choise of routing module                               (default: 0)
!%      ``default_parameters``     Default SMASH parameters                               (default: see below)
!%      ``default_states``         Default SMASH states                                   (default: see below)
!%      ``optim_parameters``       Choice of optimized SMASH parameters                   (default: 0)
!%      ``lb_parameters``          Lower bounds of SMASH parameters                       (default: see below)
!%      ``ub_parameters``          Upper bounds of SMASH parameters                       (default: see below
!%      ``maxiter``                Maximum number of optimization iteration               (default: 100)
!%      ``save_qsim_domain``       Save simulated discharge on the domain                 (default: .false.)
!%      =========================  =====================================
!%
!%      contains
!%
!%      [1] setup_copy

module mwd_setup
    
    use mwd_common !% only: sp, dp, lchar, np, ns
    
    implicit none
    
    type :: SetupDT
    
        real(sp) :: dt = 3600._sp
        
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
        character(lchar) :: prcp_format = "tif"
        real(sp) :: prcp_conversion_factor = 1._sp
        character(lchar) :: prcp_directory = "..."
        
        logical :: read_pet = .true.
        character(lchar) :: pet_format = "tif"
        real(sp) :: pet_conversion_factor = 1._sp
        character(lchar) :: pet_directory = "..."
        logical :: daily_interannual_pet = .false.
        
        logical :: mean_forcing = .false.
        
        integer :: interception_module = 0
        integer :: production_module = 0
        integer :: transfer_module = 0
        integer :: exchange_module = 0
        integer :: routing_module = 0
        
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
        &   0.01_sp/)  !% hlr
        
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
        
        & (/1e2_sp      ,&  !% ci
        &   1e3_sp      ,&  !% cp
        &   1e3_sp      ,&  !% beta
        &   1e3_sp      ,&  !% cft
        &   1e4_sp      ,&  !% cst
        &   0.999999_sp ,&  !% alpha
        &   50._sp      ,&  !% exc
        &   1e3_sp/)        !% lr
        
        integer :: maxiter = 100
        
        logical :: save_qsim_domain = .false.
        
    end type SetupDT
    
    contains
    
!%      TODO comment
        subroutine setup_copy(setup_in, setup_out)
                
            implicit none
            
            type(SetupDT), intent(in) :: setup_in
            type(SetupDT), intent(out) :: setup_out
            
            setup_out = setup_in
            
        end subroutine setup_copy

end module mwd_setup
