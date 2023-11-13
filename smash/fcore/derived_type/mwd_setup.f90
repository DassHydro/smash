!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SetupDT
!%          All user setup informations
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``structure``              Hydrological model structure                           (default: 'gr4-lr')
!%          ``serr_mu_mapping``        Mapping for structural error model                     (default: 'Zero')
!%          ``serr_sigma_mapping``     Mapping for structural error model                     (default: 'Linear')
!%          ``dt``                     Solver time step        [s]                            (default: 3600)
!%          ``start_time``             Simulation start time   [%Y%m%d%H%M]                   (default: '...')
!%          ``end_time``               Simulation end time     [%Y%m%d%H%M]                   (default: '...')
!%          ``adjust_interception``    Adjust interception reservoir capacity                 (default: .true.)
!%          ``read_qobs``              Read observed discharge                                (default: .false.)
!%          ``qobs_directory``         Observed discharge directory path                      (default: '...')
!%          ``read_prcp``              Read precipitation                                     (default: .false.)
!%          ``prcp_format``            Precipitation format                                   (default: 'tif')
!%          ``prcp_conversion_factor`` Precipitation conversion factor                        (default: 1)
!%          ``prcp_directory``         Precipiation directory path                            (default: '...')
!%          ``read_pet``               Read potential evapotranspiration                      (default: .false.)
!%          ``pet_format``             Potential evapotranspiration format                    (default: 'tif')
!%          ``pet_conversion_factor``  Potential evapotranpisration conversion factor         (default: 1)
!%          ``pet_directory``          Potential evapotranspiration directory path            (default: '...')
!%          ``daily_interannual_pet``  Read daily interannual potential evapotranspiration    (default: .false.)
!%          ``sparse_storage``         Forcing sparse storage                                 (default: .false.)
!%          ``read_descriptor``        Read descriptor map(s)                                 (default: .false.)
!%          ``descriptor_format``      Descriptor maps format                                 (default: .false.)
!%          ``descriptor_directory``   Descriptor maps directory                              (default: "...")
!%          ``descriptor_name``        Descriptor maps names
!%          ``hidden_neuron``          Number of neurons in hidden layers
!%          ``ntime_step``             Number of time steps                                   (default: -99)
!%          ``nd``                     Number of descriptor maps                              (default: -99)
!%          ``nop``                    Number of rainfall-runoff parameters                   (default: -99)
!%          ``nos``                    Number of rainfall-runoff states                       (default: -99)
!%          ``nsep_mu``                Number of structural error parameters for mu           (default: -99)
!%          ``nsep_sigma``             Number of structural error parameters for sigma        (default: -99)
!%          ``nhl``                    Number of hidden layers of the neural network          (default: -99)
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

        character(lchar) :: structure = "gr4-lr" !$F90W char
        character(lchar) :: serr_mu_mapping = "Zero" !$F90W char
        character(lchar) :: serr_sigma_mapping = "Linear" !$F90W char

        real(sp) :: dt = 3600._sp

        character(lchar) :: start_time = "..." !$F90W char
        character(lchar) :: end_time = "..." !$F90W char

        logical :: adjust_interception = .true.

        logical :: read_qobs = .false.
        character(lchar) :: qobs_directory = "..." !$F90W char

        logical :: read_prcp = .false.
        character(lchar) :: prcp_format = "tif" !$F90W char
        real(sp) :: prcp_conversion_factor = 1._sp
        character(lchar) :: prcp_directory = "..." !$F90W char

        logical :: read_pet = .false.
        character(lchar) :: pet_format = "tif" !$F90W char
        real(sp) :: pet_conversion_factor = 1._sp
        character(lchar) :: pet_directory = "..." !$F90W char
        logical :: daily_interannual_pet = .false.

        logical :: sparse_storage = .false.

        logical :: read_descriptor = .false.
        character(lchar) :: descriptor_format = "tif" !$F90W char
        character(lchar) :: descriptor_directory = "..." !$F90W char
        character(20), allocatable, dimension(:) :: descriptor_name !$F90W char-array
        integer, allocatable, dimension(:) :: hidden_neuron

        integer :: ntime_step = -99
        integer :: nd = -99
        integer :: nop = -99
        integer :: nos = -99
        integer :: nsep_mu = -99
        integer :: nsep_sigma = -99
        integer :: nhl = -99

    end type SetupDT

contains

    subroutine SetupDT_initialise(this, nd, nhl)

        !% Notes
        !% -----
        !% SetupDT initialisation subroutine

        implicit none

        type(SetupDT), intent(inout) :: this
        integer, intent(in) :: nd
        integer, intent(in) :: nhl

        this%nd = nd

        allocate (this%descriptor_name(nd))
        this%descriptor_name = "..."

        this%nhl = nhl

        allocate (this%hidden_neuron(max(0, nhl)))
        this%hidden_neuron = 16  ! TODO: Handle this with Python constant (when merging git)

    end subroutine SetupDT_initialise

    subroutine SetupDT_copy(this, this_copy)

        implicit none

        type(SetupDT), intent(in) :: this
        type(SetupDT), intent(out) :: this_copy

        this_copy = this

    end subroutine SetupDT_copy

end module mwd_setup
