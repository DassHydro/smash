!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SetupDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``structure``              Solver structure                                       (default: 'gr-a')
!%          ``dt``                     Solver time step        [s]                            (default: 3600)
!%          ``start_time``             Simulation start time   [%Y%m%d%H%M]                   (default: '...')
!%          ``end_time``               Simulation end time     [%Y%m%d%H%M]                   (default: '...')
!%          ``read_qobs``              Read observed discharge                                (default: .false.)
!%          ``qobs_directory``         Observed discharge directory path                      (default: '...')
!%          ``read_prcp``              Read precipitation                                     (default: .false.)
!%          ``prcp_format``            Precipitation format                                   (default: 'tif')
!%          ``prcp_conversion_factor`` Precipitation conversion factor                        (default: 1)
!%          ``prcp_directory``         Precipiation directory path                            (default: '...')
!%          ``read_pet``               Reap potential evapotranspiration                      (default: .false.)
!%          ``pet_format``             Potential evapotranspiration format                    (default: 'tif')
!%          ``pet_conversion_factor``  Potential evapotranpisration conversion factor         (default: 1)
!%          ``pet_directory``          Potential evapotranspiration directory path            (default: '...')
!%          ``daily_interannual_pet``  Read daily interannual potential evapotranspiration    (default: .false.)
!%          ``sparse_storage``         Forcing sparse storage                                 (default: .false.)
!%          ``read_descriptor``        Read descriptor map(s)                                 (default: .false.)
!%          ``descriptor_format``      Descriptor maps format                                 (default: .false.)
!%          ``descriptor_directory``   Descriptor maps directory                              (default: "...")
!%          ``descriptor_name``        Descriptor maps names
!%          ``ntime_step``             Number of time steps                                   (default: 0)
!%          ``nd``                     Number of descriptor maps                              (default: 0)
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

        character(lchar) :: structure = "gr-a" !>f90w-char

        real(sp) :: dt = 3600._sp

        character(lchar) :: start_time = "..." !>f90w-char
        character(lchar) :: end_time = "..." !>f90w-char

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

        logical :: sparse_storage = .false.

        logical :: read_descriptor = .false.
        character(lchar) :: descriptor_format = "tif" !>f90w-char
        character(lchar) :: descriptor_directory = "..." !>f90w-char
        character(20), allocatable, dimension(:) :: descriptor_name !>f90w-char_array

        integer :: ntime_step = 0
        integer :: nd = 0

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
