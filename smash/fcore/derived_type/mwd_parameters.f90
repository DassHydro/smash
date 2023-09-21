!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ParametersDT
!%          Container for all parameters. The goal is to keep the control vector in sync with the spatial matrices
!%          of operator parameters and the hyper parameters for mu/sigma of structural erros
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``control``                ControlDT
!%          ``opr_parameters``         Opr_ParametersDT
!%          ``opr_initial_states``     Opr_StatesDT
!%          ``serr_mu_parameters``     Serr_Mu_ParametersDT
!%          ``serr_sigma_parameters``  Serr_Sigma_ParametersDT
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
    use mwd_serr_mu_parameters !% only: Serr_Mu_ParametersDT, Serr_Mu_ParametersDT_initialise
    use mwd_serr_sigma_parameters !% only: Serr_Sigma_ParametersDT, Serr_Sigma_ParametersDT_initialise

    implicit none

    type ParametersDT

        type(ControlDT) :: control
        type(Opr_ParametersDT) :: opr_parameters
        type(Opr_StatesDT) :: opr_initial_states
        type(Serr_Mu_ParametersDT) :: serr_mu_parameters
        type(Serr_Sigma_ParametersDT) :: serr_sigma_parameters

    end type ParametersDT

contains

    subroutine ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call Opr_ParametersDT_initialise(this%opr_parameters, setup, mesh)
        call Opr_StatesDT_initialise(this%opr_initial_states, setup, mesh)
        call Serr_Mu_ParametersDT_initialise(this%serr_mu_parameters, setup, mesh)
        call Serr_Sigma_ParametersDT_initialise(this%serr_sigma_parameters, setup, mesh)

    end subroutine ParametersDT_initialise

    subroutine ParametersDT_copy(this, this_copy)

        implicit none

        type(ParametersDT), intent(in) :: this
        type(ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ParametersDT_copy

end module mwd_parameters
