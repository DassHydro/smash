!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Atmos_DataDT
!%           Atmospheric data used to force smash and derived quantities.
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``prcp``                 Precipitation field                         [mm]
!%          ``pet``                  Potential evapotranspiration field          [mm]
!%          ``snow``                 Snow field                                  [mm]
!%          ``temp``                 Temperature field                           [C]
!%          ``sparse_prcp``          Sparse precipitation field                  [mm]
!%          ``sparse_pet``           Sparse potential evapotranspiration field   [mm]
!%          ``sparse_snow``          Sparse snow field                           [mm]
!%          ``sparse_temp``          Sparse temperature field                    [C]
!%          ``mean_prcp``            Mean precipitation at gauge                 [mm]
!%          ``mean_pet``             Mean potential evapotranspiration at gauge  [mm]
!%          ``mean_snow``            Mean snow at gauge                          [mm]
!%          ``mean_temp``            Mean temperature at gauge                   [C]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Atmos_DataDT_initialise
!%      - Atmos_DataDT_copy

module mwd_atmos_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_sparse_matrix !% only: Sparse_MatrixDT, Sparse_MatrixDT_initialise_array

    implicit none

    type Atmos_DataDT

        real(sp), dimension(:, :, :), allocatable :: prcp
        real(sp), dimension(:, :, :), allocatable :: pet
        real(sp), dimension(:, :, :), allocatable :: snow
        real(sp), dimension(:, :, :), allocatable :: temp

        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_prcp
        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_pet
        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_snow
        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_temp

        real(sp), dimension(:, :), allocatable :: mean_prcp
        real(sp), dimension(:, :), allocatable :: mean_pet
        real(sp), dimension(:, :), allocatable :: mean_snow
        real(sp), dimension(:, :), allocatable :: mean_temp

    end type Atmos_DataDT

contains

    subroutine Atmos_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Atmos_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        if (setup%sparse_storage) then

            allocate (this%sparse_prcp(setup%ntime_step))
            call Sparse_MatrixDT_initialise_array(this%sparse_prcp, 0, .true., -99._sp)
            allocate (this%sparse_pet(setup%ntime_step))
            call Sparse_MatrixDT_initialise_array(this%sparse_pet, 0, .true., -99._sp)

            if (setup%snow_module_present) then
                allocate (this%sparse_snow(setup%ntime_step))
                call Sparse_MatrixDT_initialise_array(this%sparse_snow, 0, .true., -99._sp)
                allocate (this%sparse_temp(setup%ntime_step))
                call Sparse_MatrixDT_initialise_array(this%sparse_temp, 0, .true., -99._sp)
            end if

        else

            allocate (this%prcp(mesh%nrow, mesh%ncol, setup%ntime_step))
            this%prcp = -99._sp
            allocate (this%pet(mesh%nrow, mesh%ncol, setup%ntime_step))
            this%pet = -99._sp

            if (setup%snow_module_present) then
                allocate (this%snow(mesh%nrow, mesh%ncol, setup%ntime_step))
                this%snow = -99._sp
                allocate (this%temp(mesh%nrow, mesh%ncol, setup%ntime_step))
                this%temp = -99._sp
            end if

        end if

        allocate (this%mean_prcp(mesh%ng, setup%ntime_step))
        this%mean_prcp = -99._sp
        allocate (this%mean_pet(mesh%ng, setup%ntime_step))
        this%mean_pet = -99._sp

        if (setup%snow_module_present) then
            allocate (this%mean_snow(mesh%ng, setup%ntime_step))
            this%mean_snow = -99._sp
            allocate (this%mean_temp(mesh%ng, setup%ntime_step))
            this%mean_temp = -99._sp
        end if

    end subroutine Atmos_DataDT_initialise

    subroutine Atmos_DataDT_copy(this, this_copy)

        implicit none

        type(Atmos_DataDT), intent(in) :: this
        type(Atmos_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Atmos_DataDT_copy

end module mwd_atmos_data
