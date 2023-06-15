!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Atmos_DataDT_dev
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
!%      - Atmos_DataDT_dev_initialise
!%      - Atmos_DataDT_dev_copy

module mwd_atmos_data_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev

    implicit none
    
    type Atmos_DataDT_dev
    
        real(sp), dimension(:, :, :), allocatable :: prcp
        real(sp), dimension(:, :, :), allocatable :: pet
        
        real(sp), dimension(:, :), allocatable :: sparse_prcp
        real(sp), dimension(:, :), allocatable :: sparse_pet
        
        real(sp), dimension(:, :), allocatable :: mean_prcp
        real(sp), dimension(:, :), allocatable :: mean_pet
    
    end type Atmos_DataDT_dev

contains
    
    subroutine Atmos_DataDT_dev_initialise(this, setup, mesh)
    
        implicit none
        
        type(Atmos_DataDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
         if (setup%sparse_storage) then

            allocate (this%sparse_prcp(mesh%nac, setup%ntime_step))
            this%sparse_prcp = -99._sp
            allocate (this%sparse_pet(mesh%nac, setup%ntime_step))
            this%sparse_pet = -99._sp

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
    
    end subroutine Atmos_DataDT_dev_initialise
    
        subroutine Atmos_DataDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(Atmos_DataDT_dev), intent(in) :: this
        type(Atmos_DataDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine Atmos_DataDT_dev_copy

end module mwd_atmos_data_dev
