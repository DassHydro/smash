!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Input_DataDT_dev
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``response_data``        Response_DataDT_dev
!%          ``physio_data``          Physio_DataDT_dev
!%          ``atmos_data``           Atmos_DataDT_dev
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Input_DataDT_dev_initialise
!%      - Input_DataDT_dev_copy

module mwd_input_data_dev

    use md_constant_dev !% only: sp
    use mwd_setup_dev !% only: SetupDT_dev
    use mwd_mesh_dev !% only: MeshDT_dev
    use mwd_response_dev !%: only: ResponseDT_dev, ResponseDT_dev_initialise
    use mwd_physio_data_dev !%: only: Physio_DataDT_dev, Physio_DataDT_dev_initialise
    use mwd_atmos_data_dev !%: only: Atmos_DataDT_dev, Atmos_DataDT_dev_initialise

    implicit none
    
    type Input_DataDT_dev

        !% Notes
        !% -----
        !% Input_DataDT_dev Derived Type.
        
        type(ResponseDT_dev) :: obs_response
        
        type(Physio_DataDT_dev) :: physio_data
        
        type(Atmos_DataDT_dev) :: atmos_data

    end type Input_DataDT_dev

contains

    subroutine Input_DataDT_dev_initialise(this, setup, mesh)

        !% Notes
        !% -----
        !% Input_DataDT initialisation subroutine.

        implicit none

        type(Input_DataDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(in) :: setup
        type(MeshDT_dev), intent(in) :: mesh
        
        call ResponseDT_dev_initialise(this%obs_response, setup, mesh)
        
        call Physio_DataDT_dev_initialise(this%physio_data, setup, mesh)
        
        call Atmos_DataDT_dev_initialise(this%atmos_data, setup, mesh)

    end subroutine Input_DataDT_dev_initialise
    
    subroutine Input_DataDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(Input_DataDT_dev), intent(in) :: this
        type(Input_DataDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine Input_DataDT_dev_copy

end module mwd_input_data_dev
