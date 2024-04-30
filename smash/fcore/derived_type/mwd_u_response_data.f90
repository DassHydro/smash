!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - U_Response_DataDT
!%          User-provided observation uncertainties for the hydrological model response variables
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``q_stdev``              Discharge uncertainty at gauges (standard deviation of independent error) [m3/s]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - U_Response_DataDT_initialise
!%      - U_Response_DataDT_copy

module mwd_u_response_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type U_Response_DataDT

        real(sp), dimension(:, :), allocatable :: q_stdev

    end type U_Response_DataDT

contains

    subroutine U_Response_DataDT_initialise(this, setup, mesh)

        implicit none

        type(U_Response_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%q_stdev(mesh%ng, setup%ntime_step))
        this%q_stdev = 0._sp

    end subroutine U_Response_DataDT_initialise

    subroutine U_Response_DataDT_copy(this, this_copy)

        implicit none

        type(U_Response_DataDT), intent(in) :: this
        type(U_Response_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine U_Response_DataDT_copy

end module mwd_u_response_data
