!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Atmos_DataDT
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``prcp``                 Precipitation field                         [mm]
!%          ``pet``                  Potential evapotranspiration field          [mm]
!%          ``sparse_prcp``          Sparse precipitation field                  [mm]
!%          ``sparse_pet``           Spase potential evapotranspiration field    [mm]
!%          ``mean_prcp``            Mean precipitation at gauge                 [mm]
!%          ``mean_pet``             Mean potential evapotranspiration at gauge  [mm]
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
    use mwd_sparse_matrix !% only: Sparse_MatrixDT

    implicit none

    type Atmos_DataDT

        real(sp), dimension(:, :, :), allocatable :: prcp
        real(sp), dimension(:, :, :), allocatable :: pet

        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_prcp
        type(Sparse_MatrixDT), dimension(:), allocatable :: sparse_pet

        real(sp), dimension(:, :), allocatable :: mean_prcp
        real(sp), dimension(:, :), allocatable :: mean_pet

    end type Atmos_DataDT

contains

    subroutine Atmos_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Atmos_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        if (setup%sparse_storage) then

            allocate (this%sparse_prcp(setup%ntime_step))
            allocate (this%sparse_pet(setup%ntime_step))

        else

            allocate (this%prcp(mesh%nrow, mesh%ncol, setup%ntime_step))
            this%prcp = -99._sp
            allocate (this%pet(mesh%nrow, mesh%ncol, setup%ntime_step))
            this%pet = -99._sp

        end if

        allocate (this%mean_prcp(mesh%ng, setup%ntime_step))
        this%mean_prcp = -99._sp
        allocate (this%mean_pet(mesh%ng, setup%ntime_step))
        this%mean_pet = -99._sp

    end subroutine Atmos_DataDT_initialise

    subroutine Atmos_DataDT_copy(this, this_copy)

        implicit none

        type(Atmos_DataDT), intent(in) :: this
        type(Atmos_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Atmos_DataDT_copy

end module mwd_atmos_data
