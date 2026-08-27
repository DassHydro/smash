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
!%          ``ac_q``                 Active cell discharge
!%          ``response``             ResponseDT
!%          ``rr_final_states``      RR_StatesDT
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
    use mwd_rr_states !% only: RR_StatesDT, RR_StatesDT_initialise

    implicit none

    type OutputDT

        type(ResponseDT) :: response
        type(RR_StatesDT) :: rr_final_states
        real(sp) :: cost
        real(sp), dimension(:, :), allocatable :: ac_q
        logical :: ac_q_allocated = .false.

    end type OutputDT

contains

    subroutine OutputDT_initialise(this, setup, mesh)

        implicit none

        type(OutputDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call ResponseDT_initialise(this%response, setup, mesh)
        call RR_StatesDT_initialise(this%rr_final_states, setup, mesh)
        allocate (this%ac_q(1, 1))

    end subroutine OutputDT_initialise

    subroutine OutputDT_copy(this, this_copy)

        implicit none

        type(OutputDT), intent(in) :: this
        type(OutputDT), intent(out) :: this_copy

        this_copy = this

    end subroutine OutputDT_copy

    subroutine OutputDT_allocate_ac_q(this, setup, mesh)

        implicit none

        type(OutputDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        if (allocated(this%ac_q)) deallocate (this%ac_q)
        allocate (this%ac_q(mesh%nac, setup%ntime_step))
        this%ac_q_allocated = .true.

    end subroutine OutputDT_allocate_ac_q

    subroutine OutputDT_deallocate_ac_q(this)

        implicit none

        type(OutputDT), intent(inout) :: this

        if (allocated(this%ac_q)) deallocate (this%ac_q)
        allocate (this%ac_q(1, 1))
        this%ac_q_allocated = .false.

    end subroutine OutputDT_deallocate_ac_q

end module mwd_output
