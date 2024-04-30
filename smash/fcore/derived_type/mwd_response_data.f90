!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Response_DataDT
!%          User-provided observation for the hydrological model response variables
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``q``                    Observed discharge at gauges              [m3/s]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Response_DataDT_initialise
!%      - Response_DataDT_copy

module mwd_response_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Response_DataDT

        real(sp), dimension(:, :), allocatable :: q

    end type Response_DataDT

contains

    subroutine Response_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Response_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%q(mesh%ng, setup%ntime_step))
        this%q = -99._sp

    end subroutine Response_DataDT_initialise

    subroutine Response_DataDT_copy(this, this_copy)

        implicit none

        type(Response_DataDT), intent(in) :: this
        type(Response_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Response_DataDT_copy

end module mwd_response_data
