!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ParametersDT
!%          Container for all parameters. The goal is to keep the control vector in sync with the spatial matrices
!%          of rainfall-runoff parameters and the hyper parameters for mu/sigma of structural erros
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``control``                ControlDT
!%          ``rr_parameters``          Rr_ParametersDT
!%          ``rr_initial_states``      Rr_StatesDT
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
    use mwd_rr_parameters !% only: Rr_ParametersDT, Rr_ParametersDT_initialise
    use mwd_rr_states !% only: Rr_StatesDT, Rr_StatesDT_initialise
    use mwd_serr_mu_parameters !% only: Serr_Mu_ParametersDT, Serr_Mu_ParametersDT_initialise
    use mwd_serr_sigma_parameters !% only: Serr_Sigma_ParametersDT, Serr_Sigma_ParametersDT_initialise

    implicit none

    type ParametersDT

        type(ControlDT) :: control
        type(Rr_ParametersDT) :: rr_parameters
        type(Rr_StatesDT) :: rr_initial_states
        type(Serr_Mu_ParametersDT) :: serr_mu_parameters
        type(Serr_Sigma_ParametersDT) :: serr_sigma_parameters

    end type ParametersDT

contains

    subroutine ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call Rr_ParametersDT_initialise(this%rr_parameters, setup, mesh)
        call Rr_StatesDT_initialise(this%rr_initial_states, setup, mesh)
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
