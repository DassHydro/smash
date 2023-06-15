!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Physio_DataDT_dev
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``descriptor``           Descriptor maps field                       [(descriptor dependent)]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Physio_DataDT_dev_initialise
!%      - Physio_DataDT_dev_copy

module mwd_physio_data_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev

    implicit none
    
    type Physio_DataDT_dev
        
        real(sp), dimension(:, :, :), allocatable :: descriptor
    
    end type Physio_DataDT_dev
    
contains
    
    subroutine Physio_DataDT_dev_initialise(this, setup, mesh)
        
        implicit none
        
        type(Physio_DataDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
        allocate (this%descriptor(mesh%nrow, mesh%ncol, setup%nd))
        this%descriptor = -99._sp
    
    end subroutine Physio_DataDT_dev_initialise
    
    subroutine Physio_DataDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(Physio_DataDT_dev), intent(in) :: this
        type(Physio_DataDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine Physio_DataDT_dev_copy

end module mwd_physio_data_dev
