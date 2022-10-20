!%      This module `mwd_setup` encapsulates all SMASH setup.
!%      This module is wrapped and differentiated.
!%
!%      SetupDT type:
!%
!%      </> Public
!%      ========================== =====================================
!%      `Variables`                Description
!%      ========================== =====================================
!%      ``dt``                     Solver time step        [s]                            (default: 3600)
!%      ``start_time``             Simulation start time   [%Y%m%d%H%M]                   (default: '...')
!%      ``end_time``               Simulation end time     [%Y%m%d%H%M]                   (default: '...')
!%      ``sparse_storage``         Forcing sparse storage                                 (default: .false.)
!%      ``read_qobs``              Read observed discharge                                (default: .false.)
!%      ``qobs_directory``         Observed discharge directory path                      (default: '...')
!%      ``read_prcp``              Read precipitation                                     (default: .false.)
!%      ``prcp_format``            Precipitation format                                   (default: 'tif')
!%      ``prcp_conversion_factor`` Precipitation conversion factor                        (default: 1)
!%      ``prcp_directory``         Precipiation directory path                            (default: '...')
!%      ``read_pet``               Reap potential evapotranspiration                      (default: .false.)
!%      ``pet_format``             Potential evapotranspiration format                    (default: 'tif')
!%      ``pet_conversion_factor``  Potential evapotranpisration conversion factor         (default: 1)
!%      ``pet_directory``          Potential evapotranspiration directory path            (default: '...')
!%      ``daily_interannual_pet``  Read daily interannual potential evapotranspiration    (default: .false.)
!%      ``mean_forcing``           Compute mean forcing                                   (default: .false.)
!%      ``prcp_indice``            Compute prcp indices                                   (default: .false.)
!%      ``read_descriptor``        Read descriptor map(s)                                 (default: .false.)
!%      ``descriptor_format``      Descriptor map(s) format                               (default: .false.)
!%      ``descriptor_directory``   Descriptor map(s) directory                            (default: "...")
!%      ``descriptor_name``        Descriptor map(s) names
!%      ``interception_module``    Choice of interception module                          (default: 0)
!%      ``production_module``      Choice of production module                            (default: 0)
!%      ``transfer_module``        Choice of transfer module                              (default: 0)
!%      ``exchange_module``        Choice of exchange module                              (default: 0)
!%      ``routing_module``         Choice of routing module                               (default: 0)
!%      ``save_qsim_domain``       Save simulated discharge on the domain                 (default: .false.)
!%      ``save_net_prcp_domain``   Save net precipitation on the domain                   (default: .false.)
!%
!%      </> Private
!%      ========================== =====================================
!%      `Variables`                Description
!%      ========================== =====================================
!%      ``ntime_step``             Number of time step
!%      ``nd``                     Number of descriptor map(s)
!%      ``algorithm``              Optimize Algorithm name
!%      ``jobs_fun``               Objective function name        (default: 'nse')
!%      ``jreg_fun``               Regularization name            (default: 'prior')
!%      ``wjreg``                  Regularization weight          (default: 0)
!%      ``optim_start_step``       Optimization start step        (default: 1)
!%      ``optim_parameters``       Optimized parameters array     (default: 0)
!%      ``optim_states``           Optimize states array          (default: 0)
!%      ``lb_parameters``          Parameters lower bounds        (default: see below)
!%      ``ub_parameters``          Parameters upper bounds        (default: see below)
!%      ``lb_states``              States lower bounds            (default: see below)
!%      ``ub_states``              States upper bounds            (default: see below)
!%      ``maxiter``                Maximum number of iteration    (default: 100)
!%      =========================  =====================================
!%
!%      contains
!%      
!%      [1] SetupDT_initialise

module mwd_setup
    
    use mwd_common !% only: sp, lchar, np, ns
    
    implicit none
    
    type :: SetupDT
    
        !% </> Public
        real(sp) :: dt = 3600._sp
        
        character(lchar) :: start_time = "..."
        character(lchar) :: end_time = "..."
        
        logical :: sparse_storage = .false.
        
        logical :: read_qobs = .false.
        character(lchar) :: qobs_directory = "..."
        
        logical :: read_prcp = .false.
        character(lchar) :: prcp_format = "tif"
        real(sp) :: prcp_conversion_factor = 1._sp
        character(lchar) :: prcp_directory = "..."
        
        logical :: read_pet = .false.
        character(lchar) :: pet_format = "tif"
        real(sp) :: pet_conversion_factor = 1._sp
        character(lchar) :: pet_directory = "..."
        logical :: daily_interannual_pet = .false.
        
        logical :: mean_forcing = .false.
        logical :: prcp_indice = .false.
        
        
        logical :: read_descriptor = .false.
        character(lchar) :: descriptor_format = "tif"
        character(lchar) :: descriptor_directory = "..."
        character(20), allocatable, dimension(:) :: descriptor_name
        
        integer :: interception_module = 0
        integer :: production_module = 0
        integer :: transfer_module = 0
        integer :: exchange_module = 0
        integer :: routing_module = 0
        
        logical :: save_qsim_domain = .false.
        logical :: save_net_prcp_domain = .false.
        
        !% </> Private
        integer :: ntime_step = 0 !>f90wrap private
        integer :: nd = 0 !>f90wrap private
        
        character(lchar) :: algorithm !>f90wrap private
        
        character(lchar) :: jobs_fun = "nse" !>f90wrap private
        character(lchar) :: jreg_fun = "prior" !>f90wrap private
        real(sp) :: wjreg = 0._sp !>f90wrap private

        integer :: optim_start_step = 1 !>f90wrap private
        
        integer, dimension(np) :: optim_parameters = 0 !>f90wrap private
        integer, dimension(ns) :: optim_states = 0 !>f90wrap private
        
        real(sp), dimension(np) :: lb_parameters = & !>f90wrap private
        
        & (/1e-6_sp ,& !% ci
        &   1e-6_sp ,& !% cp
        &   1e-6_sp ,& !% beta
        &   1e-6_sp ,& !% cft
        &   1e-6_sp ,& !% cst
        &   1e-6_sp ,& !% alpha
        &   -50._sp ,& !% exc
        &   1e-6_sp/)  !% lr
        
        real(sp), dimension(np) :: ub_parameters = & !>f90wrap private
        
        & (/1e2_sp      ,&  !% ci
        &   1e3_sp      ,&  !% cp
        &   1e3_sp      ,&  !% beta
        &   1e3_sp      ,&  !% cft
        &   1e4_sp      ,&  !% cst
        &   0.999999_sp ,&  !% alpha
        &   50._sp      ,&  !% exc
        &   1e3_sp/)        !% lr
        
        real(sp), dimension(ns) :: lb_states = & !>f90wrap private
        
        & (/1e-6_sp ,& !% hi
        &   1e-6_sp ,& !% hp
        &   1e-6_sp ,& !% hft
        &   1e-6_sp ,& !% hst
        &   1e-6_sp/)  !% hlr
        
        real(sp), dimension(ns) :: ub_states = & !>f90wrap private
        
        & (/0.999999_sp ,& !% hi
        &   0.999999_sp ,& !% hp
        &   0.999999_sp ,& !% hft
        &   0.999999_sp ,& !% hst
        &   10000._sp/)    !% hlr
        
        integer :: maxiter = 100 !>f90wrap private
        
    end type SetupDT
    
    contains
    

        subroutine SetupDT_initialise(setup, nd)
        
            !% Notes
            !% -----
            !%
            !% SetupDT initialisation subroutine
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            integer, intent(in) :: nd
            
            setup%nd = nd
            
            if (setup%nd .gt. 0) then
            
                allocate(setup%descriptor_name(setup%nd))
                setup%descriptor_name = "..."
            
            end if
        
        end subroutine SetupDT_initialise

end module mwd_setup
