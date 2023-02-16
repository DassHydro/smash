!%      This module `mwd_input_data` encapsulates all SMASH input_data.
!%      This module is wrapped and differentiated.
!%
!%      Input_DataDT type:
!%
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``qobs``                 Oberserved discharge at gauge               [m3/s]
!%      ``prcp``                 Precipitation field                         [mm]
!%      ``pet``                  Potential evapotranspiration field          [mm]
!%      ``descriptor``           Descriptor map(s) field                     [(descriptor dependent)]
!%      ``sparse_prcp``          Sparse precipitation field                  [mm]
!%      ``sparse_pet``           Spase potential evapotranspiration field    [mm]
!%      ``mean_prcp``            Mean precipitation at gauge                 [mm]
!%      ``mean_pet``             Mean potential evapotranspiration at gauge  [mm]
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] Input_DataDT_initialise

module mwd_input_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT

    implicit none

    type Input_DataDT

        !% Notes
        !% -----
        !% Input_DataDT Derived Type.

        real(sp), dimension(:, :), allocatable :: qobs
        real(sp), dimension(:, :, :), allocatable :: prcp
        real(sp), dimension(:, :, :), allocatable :: pet

        real(sp), dimension(:, :, :), allocatable :: descriptor

        real(sp), dimension(:, :), allocatable :: sparse_prcp
        real(sp), dimension(:, :), allocatable :: sparse_pet

        real(sp), dimension(:, :), allocatable :: mean_prcp
        real(sp), dimension(:, :), allocatable :: mean_pet

    end type Input_DataDT

contains

    subroutine Input_DataDT_initialise(this, setup, mesh)

        !% Notes
        !% -----
        !% Input_DataDT initialisation subroutine.

        implicit none

        type(Input_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        if (mesh%ng .gt. 0) then

            allocate (this%qobs(mesh%ng, setup%ntime_step))
            this%qobs = -99._sp

        end if

        if (setup%sparse_storage) then

            allocate (this%sparse_prcp(mesh%nac, &
            & setup%ntime_step))
            this%sparse_prcp = -99._sp
            allocate (this%sparse_pet(mesh%nac, &
            & setup%ntime_step))
            this%sparse_pet = -99._sp

        else

            allocate (this%prcp(mesh%nrow, mesh%ncol, &
            & setup%ntime_step))
            this%prcp = -99._sp
            allocate (this%pet(mesh%nrow, mesh%ncol, &
            & setup%ntime_step))
            this%pet = -99._sp

        end if

        if (setup%nd .gt. 0) then

            allocate (this%descriptor(mesh%nrow, mesh%ncol, &
            & setup%nd))

        end if

        if (setup%mean_forcing .and. mesh%ng .gt. 0) then

            allocate (this%mean_prcp(mesh%ng, &
            & setup%ntime_step))
            this%mean_prcp = -99._sp
            allocate (this%mean_pet(mesh%ng, setup%ntime_step))
            this%mean_pet = -99._sp

        end if

    end subroutine Input_DataDT_initialise

end module mwd_input_data
