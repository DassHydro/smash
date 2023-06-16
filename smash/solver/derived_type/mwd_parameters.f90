!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ParametersDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``control``                ControlDT
!%          ``opr_parameters``         Opr_ParametersDT
!%          ``opr_initial_states``     Opr_StatesDT
!%
!§      Subroutine
!%      ----------
!%
!%      - ParametersDT_initialise
!%      - ParametersDT_copy

module mwd_parameters

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_control !% only: ControlDT
    use mwd_opr_parameters !% only: Opr_ParametersDT, Opr_ParametersDT_initialise
    use mwd_opr_states !% only: Opr_StatesDT, Opr_StatesDT_initialise

    implicit none

    type ParametersDT

        type(ControlDT) :: control
        type(Opr_ParametersDT) :: opr_parameters
        type(Opr_StatesDT) :: opr_initial_states

    end type ParametersDT

contains

    subroutine ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call Opr_ParametersDT_initialise(this%opr_parameters, setup, mesh)
        call Opr_StatesDT_initialise(this%opr_initial_states, setup, mesh)

    end subroutine ParametersDT_initialise

    subroutine ParametersDT_copy(this, this_copy)

        implicit none

        type(ParametersDT), intent(in) :: this
        type(ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ParametersDT_copy

end module mwd_parameters
