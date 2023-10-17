!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - OutputDT
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``cost``                 Value of cost function
!%          ``response``             ResponseDT
!%          ``rr_final_states``      Rr_StatesDT
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - OutputDT_initialise
!%      - OutputDT_copy

module mwd_output

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_response !% only: ResponseDT, ResponseDT_initialise
    use mwd_rr_states !% only: Rr_StatesDT, Rr_StatesDT_initialise

    implicit none

    type OutputDT

        type(ResponseDT) :: response
        type(Rr_StatesDT) :: rr_final_states
        real(sp) :: cost

    end type OutputDT

contains

    subroutine OutputDT_initialise(this, setup, mesh)

        implicit none

        type(OutputDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call ResponseDT_initialise(this%response, setup, mesh)
        call Rr_StatesDT_initialise(this%rr_final_states, setup, mesh)

    end subroutine OutputDT_initialise

    subroutine OutputDT_copy(this, this_copy)

        implicit none

        type(OutputDT), intent(in) :: this
        type(OutputDT), intent(out) :: this_copy

        this_copy = this

    end subroutine OutputDT_copy

end module mwd_output
