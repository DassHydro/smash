!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SErr_Mu_ParametersDT
!%          Vectors containting hyper parameters of the temporalisation function for mu, the mean of structural errors
!%          (mg0, mg1, ...)
!%
!%          ======================== =============================================
!%          `Variables`              Description
!%          ======================== =============================================
!%          ``keys``                 Structural errors mu hyper parameters keys
!%          ``values``               Structural errors mu hyper parameters values
!%          ======================== =============================================
!%
!%      Subroutine
!%      ----------
!%
!%      - SErr_Mu_ParametersDT_initialise
!%      - SErr_Mu_ParametersDT_copy

module mwd_serr_mu_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type SErr_Mu_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :), allocatable :: values

    end type SErr_Mu_ParametersDT

contains

    subroutine SErr_Mu_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(SErr_Mu_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nsep_mu))
        this%keys = "..."

        allocate (this%values(mesh%ng, setup%nsep_mu))
        this%values = -99._sp

    end subroutine SErr_Mu_ParametersDT_initialise

    subroutine SErr_Mu_ParametersDT_copy(this, this_copy)

        implicit none

        type(SErr_Mu_ParametersDT), intent(in) :: this
        type(SErr_Mu_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine SErr_Mu_ParametersDT_copy

end module mwd_serr_mu_parameters
