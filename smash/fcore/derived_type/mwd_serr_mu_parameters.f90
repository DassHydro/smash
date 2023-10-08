!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Serr_Mu_ParametersDT
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
!%      - Serr_Mu_ParametersDT_initialise
!%      - Serr_Mu_ParametersDT_copy

module mwd_serr_mu_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Serr_Mu_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :), allocatable :: values

    end type Serr_Mu_ParametersDT

contains

    subroutine Serr_Mu_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(Serr_Mu_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nsep_mu))
        this%keys = "..."

        allocate (this%values(mesh%ng, setup%nsep_mu))
        this%values = -99._sp

    end subroutine Serr_Mu_ParametersDT_initialise

    subroutine Serr_Mu_ParametersDT_copy(this, this_copy)

        implicit none

        type(Serr_Mu_ParametersDT), intent(in) :: this
        type(Serr_Mu_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Serr_Mu_ParametersDT_copy

end module mwd_serr_mu_parameters
