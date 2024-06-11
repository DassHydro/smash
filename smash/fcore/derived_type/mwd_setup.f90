!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SetupDT
!%          All user setup informations. See default values in _constant.py DEFAULT_SETUP
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``snow_module``            Snow module
!%          ``hydrological_module``    Hydrological module
!%          ``routing_module``         Routing module
!%          ``serr_mu_mapping``        Mapping for structural error model
!%          ``serr_sigma_mapping``     Mapping for structural error model
!%          ``dt``                     Solver time step        [s]
!%          ``start_time``             Simulation start time   [%Y%m%d%H%M]
!%          ``end_time``               Simulation end time     [%Y%m%d%H%M]
!%          ``adjust_interception``    Adjust interception reservoir capacity
!%          ``compute_mean_atmos``     Compute mean atmospheric data for each gauge
!%          ``read_qobs``              Read observed discharge
!%          ``qobs_directory``         Observed discharge directory path
!%          ``read_prcp``              Read precipitation
!%          ``prcp_format``            Precipitation format
!%          ``prcp_conversion_factor`` Precipitation conversion factor
!%          ``prcp_directory``         Precipiation directory path
!%          ``prcp_access``            Precipiation access tree
!%          ``read_pet``               Read potential evapotranspiration
!%          ``pet_format``             Potential evapotranspiration format
!%          ``pet_conversion_factor``  Potential evapotranpisration conversion factor
!%          ``pet_directory``          Potential evapotranspiration directory path
!%          ``pet_access``             Potential evapotranspiration access tree
!%          ``daily_interannual_pet``  Read daily interannual potential evapotranspiration
!%          ``read_snow``              Read snow
!%          ``snow_format``            Snow format
!%          ``snow_conversion_factor`` Snow conversion factor
!%          ``snow_directory``         Snow directory path
!%          ``snow_access``            Snow access tree
!%          ``read_temp``              Read temperatur
!%          ``temp_format``            Temperature format
!%          ``temp_directory``         Temperature directory path
!%          ``temp_access``            Temperature access tree
!%          ``prcp_partitioning``      Precipitation partitioning
!%          ``sparse_storage``         Forcing sparse storage
!%          ``read_descriptor``        Read descriptor map(s)
!%          ``descriptor_format``      Descriptor maps format
!%          ``descriptor_directory``   Descriptor maps directory
!%          ``descriptor_name``        Descriptor maps names
!%          ``structure``              Structure combaining all modules
!%          ``snow_module_present``    Presence of snow module
!%          ``ntime_step``             Number of time steps
!%          ``nd``                     Number of descriptor maps
!%          ``nrrp``                   Number of rainfall-runoff parameters
!%          ``nrrs``                   Number of rainfall-runoff states
!%          ``nsep_mu``                Number of structural error parameters for mu
!%          ``nsep_sigma``             Number of structural error parameters for sigma
!%          ``nqz``                    Size of the temporal buffer for discharge grids
!%
!%      Subroutine
!%      ----------
!%
!%      - SetupDT_initialise
!%      - SetupDT_copy

module mwd_setup

    use md_constant !% only: sp, lchar

    implicit none

    type SetupDT

        !% Notes
        !% -----
        !% SetupDT Derived Type.

        ! User variables
        character(lchar) :: snow_module = "..." !$F90W char
        character(lchar) :: hydrological_module = "..." !$F90W char
        character(lchar) :: routing_module = "..."!$F90W char

        character(lchar) :: serr_mu_mapping = "..." !$F90W char
        character(lchar) :: serr_sigma_mapping = "..." !$F90W char

        real(sp) :: dt = -99._sp

        character(lchar) :: start_time = "..." !$F90W char
        character(lchar) :: end_time = "..." !$F90W char

        logical :: adjust_interception = .true.
        logical :: compute_mean_atmos = .true.

        logical :: read_qobs = .false.
        character(2*lchar) :: qobs_directory = "..." !$F90W char

        logical :: read_prcp = .false.
        character(lchar) :: prcp_format = "..." !$F90W char
        real(sp) :: prcp_conversion_factor = 1._sp
        character(2*lchar) :: prcp_directory = "..." !$F90W char
        character(lchar) :: prcp_access = "..." !$F90W char

        logical :: read_pet = .false.
        character(lchar) :: pet_format = "..." !$F90W char
        real(sp) :: pet_conversion_factor = 1._sp
        character(2*lchar) :: pet_directory = "..." !$F90W char
        character(lchar) :: pet_access = "..." !$F90W char
        logical :: daily_interannual_pet = .false.

        logical :: read_snow = .false.
        character(lchar) :: snow_format = "..." !$F90W char
        real(sp) :: snow_conversion_factor = 1._sp
        character(2*lchar) :: snow_directory = "..." !$F90W char
        character(lchar) :: snow_access = "..." !$F90W char

        logical :: read_temp = .false.
        character(lchar) :: temp_format = "..." !$F90W char
        character(2*lchar) :: temp_directory = "..." !$F90W char
        character(lchar) :: temp_access = "..." !$F90W char

        logical :: prcp_partitioning = .false.

        logical :: sparse_storage = .false.

        logical :: read_descriptor = .false.
        character(lchar) :: descriptor_format = "..." !$F90W char
        character(2*lchar) :: descriptor_directory = "..." !$F90W char
        character(lchar), allocatable, dimension(:) :: descriptor_name !$F90W char-array

        ! Post processed variables
        character(lchar) :: structure = "..." !$F90W char
        logical :: snow_module_present = .false.

        integer :: ntime_step = -99
        integer :: nd = -99
        integer :: nrrp = -99
        integer :: nrrs = -99
        integer :: nsep_mu = -99
        integer :: nsep_sigma = -99
        integer :: nqz = -99

    end type SetupDT

contains

    subroutine SetupDT_initialise(this, nd)

        !% Notes
        !% -----
        !% SetupDT initialisation subroutine

        implicit none

        type(SetupDT), intent(inout) :: this
        integer, intent(in) :: nd

        this%nd = nd

        allocate (this%descriptor_name(this%nd))
        this%descriptor_name = "..."

    end subroutine SetupDT_initialise

    subroutine SetupDT_copy(this, this_copy)

        implicit none

        type(SetupDT), intent(in) :: this
        type(SetupDT), intent(out) :: this_copy

        this_copy = this

    end subroutine SetupDT_copy

end module mwd_setup
