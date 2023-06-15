!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ResponseDT_dev
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``q``                    Discharge at gauges              [m3/s]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - ResponseDT_dev_initialise
!%      - ResponseDT_dev_copy

module mwd_response_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev

    implicit none
    
    type ResponseDT_dev
    
        real(sp), dimension(:, :), allocatable :: q
    
    end type ResponseDT_dev

contains

    subroutine ResponseDT_dev_initialise(this, setup, mesh)
    
        implicit none
        
        type(ResponseDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
        allocate (this%q(mesh%ng, setup%ntime_step))
        this%q = -99._sp
    
    end subroutine ResponseDT_dev_initialise
    
    subroutine ResponseDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(ResponseDT_dev), intent(in) :: this
        type(ResponseDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine ResponseDT_dev_copy

end module mwd_response_dev
