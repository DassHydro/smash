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
!%      ``mean_forcing``           Compute mean forcing                                   (default: .true.)
!%      ``read_descriptor``        Read descriptor map(s)                                 (default: .false.)
!%      ``descriptor_format``      Descriptor map(s) format                               (default: .false.)
!%      ``descriptor_directory``   Descriptor map(s) directory                            (default: "...")
!%      ``descriptor_name``        Descriptor map(s) names
!%      ``save_qsim_domain``       Save simulated discharge on the domain                 (default: .false.)
!%      ``save_net_prcp_domain``   Save net precipitation on the domain                   (default: .false.)
!%
!%      </> Private
!%      ========================== =====================================
!%      `Variables`                Description
!%      ========================== =====================================
!%      ``optimize``               Optimize_SetupDT which contains all optimize options
!%      ``ntime_step``             Number of time step
!%      ``nd``                     Number of descriptor map(s)
!%      ``name_parameters``        Name of SMASH parameters       (default: see below)
!%      ``name_states``            Name of SMASH states           (default: see below)
!%      =========================  =====================================
!%
!%      contains
!%
!%      [1] SetupDT_initialise

module mwd_setup

    use md_constant !% only: sp, lchar, &
    !% & G_NP, G_NS, GPARAMETERS_NAME, GSTATES_NAME, &
    !% & GLB_PARAMETERS, GUB_PARAMETERS, GLB_STATES, GUB_STATES

    implicit none

    type Optimize_SetupDT

        !% Notes
        !% -----
        !% Optimize_SetupDT Derived Type.

        character(lchar) :: algorithm = "..." !>f90w-char

        character(20), dimension(:), allocatable :: jobs_fun !>f90w-char_array
        real(sp), dimension(:), allocatable :: wjobs_fun
        integer :: njf = 0

        logical :: verbose = .true.

        character(lchar) :: mapping = "..." !>f90w-char

        logical :: normalize_forward = .false.

        integer :: nhyper = 0

        character(lchar) :: jreg_fun = "prior" !>f90w-char
        real(sp) :: wjreg = 0._sp

        integer :: optimize_start_step = 1

        integer :: maxiter = 100

        integer, dimension(GNP) :: optim_parameters = 0
        integer, dimension(GNS) :: optim_states = 0

        real(sp), dimension(GNP) :: lb_parameters = GLB_PARAMETERS

        real(sp), dimension(GNP) :: ub_parameters = GUB_PARAMETERS

        real(sp), dimension(GNS) :: lb_states = GLB_STATES

        real(sp), dimension(GNS) :: ub_states = GUB_STATES

        real(sp), dimension(:), allocatable :: wgauge

        integer, dimension(:, :), allocatable :: mask_event

    end type Optimize_SetupDT

    type SetupDT

        !% Notes
        !% -----
        !% SetupDT Derived Type.

        character(lchar) :: structure = "gr-a" !>f90w-char

        real(sp) :: dt = 3600._sp

        character(lchar) :: start_time = "..." !>f90w-char
        character(lchar) :: end_time = "..." !>f90w-char

        logical :: sparse_storage = .false.

        logical :: read_qobs = .false.
        character(lchar) :: qobs_directory = "..." !>f90w-char

        logical :: read_prcp = .false.
        character(lchar) :: prcp_format = "tif" !>f90w-char
        real(sp) :: prcp_conversion_factor = 1._sp
        character(lchar) :: prcp_directory = "..." !>f90w-char

        logical :: read_pet = .false.
        character(lchar) :: pet_format = "tif" !>f90w-char
        real(sp) :: pet_conversion_factor = 1._sp
        character(lchar) :: pet_directory = "..." !>f90w-char
        logical :: daily_interannual_pet = .false.

        logical :: mean_forcing = .true.

        logical :: read_descriptor = .false.
        character(lchar) :: descriptor_format = "tif" !>f90w-char
        character(lchar) :: descriptor_directory = "..." !>f90w-char
        character(20), allocatable, dimension(:) :: descriptor_name !>f90w-char_array

        logical :: save_qsim_domain = .false.
        logical :: save_net_prcp_domain = .false.

        type(Optimize_SetupDT) :: optimize !>f90w-private

        integer :: ntime_step = 0 !>f90w-private
        integer :: nd = 0 !>f90w-private

        character(10), dimension(GNP) :: parameters_name = GPARAMETERS_NAME !>f90w-private f90w-char_array

        character(10), dimension(GNS) :: states_name = GSTATES_NAME !>f90w-private f90w-char_array

    end type SetupDT

contains

    subroutine Optimize_SetupDT_initialise(this, ntime_step, nd, ng, mapping, njf)

        !% Notes
        !% -----
        !% Optimize_SetupDT initialisation subroutine

        implicit none

        type(Optimize_SetupDT), intent(inout) :: this
        integer, intent(in) :: ntime_step, nd, ng, njf
        character(len=*), intent(in) :: mapping

        allocate (this%wgauge(ng))
        this%wgauge = 1._sp/ng

        this%mapping = mapping

        select case (trim(this%mapping))

        case ("hyper-linear")

            this%nhyper = (1 + nd)

        case ("hyper-polynomial")

            this%nhyper = (1 + 2*nd)

        end select

        this%njf = njf

        allocate (this%jobs_fun(this%njf))
        this%jobs_fun = "..."

        allocate (this%wjobs_fun(this%njf))
        this%wjobs_fun = 0._sp

        allocate (this%mask_event(ng, ntime_step))
        this%mask_event = 0

    end subroutine Optimize_SetupDT_initialise

    subroutine SetupDT_initialise(this, nd, ng)

        !% Notes
        !% -----
        !% SetupDT initialisation subroutine

        implicit none

        type(SetupDT), intent(inout) :: this
        integer, intent(in) :: nd, ng

        this%nd = nd

        if (this%nd .gt. 0) then

            allocate (this%descriptor_name(this%nd))
            this%descriptor_name = "..."

        end if

        call Optimize_SetupDT_initialise(this%optimize, this%ntime_step, this%nd, ng, "...", 0)

    end subroutine SetupDT_initialise

end module mwd_setup
