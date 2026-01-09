!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ResponseDT
!%          Response simulated by the hydrological model.
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``q``                    Simulated discharge at gauges              [m3/s]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - ResponseDT_initialise
!%      - ResponseDT_copy

module mwd_response

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type ResponseDT

        real(sp), dimension(:, :), allocatable :: q
        real(sp), dimension(:, :), allocatable :: qt

    end type ResponseDT

contains

    subroutine ResponseDT_initialise(this, setup, mesh)

        implicit none

        type(ResponseDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%q(mesh%ng, setup%ntime_step))
        this%q = -99._sp
        
!~         When conditionning this allocatation, tapenade force 
!~         its value to zeros before calling SIMULATION_B...
        if (setup%routing_module == "zero") then
            allocate (this%qt(mesh%nac, setup%ntime_step))
            this%qt = -99._sp
        else
            !save memory
            allocate (this%qt(1, 1))
            this%qt = -99._sp
        end if

    end subroutine ResponseDT_initialise

    subroutine ResponseDT_copy(this, this_copy)

        implicit none

        type(ResponseDT), intent(in) :: this
        type(ResponseDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ResponseDT_copy

end module mwd_response
