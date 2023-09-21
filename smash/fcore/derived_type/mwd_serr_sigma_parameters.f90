!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Serr_Sigma_ParametersDT
!%          Vectors containting hyper parameters of the temporalisation function for sigma, the standard deviation for structural errors
!%          (sg0, sg1, sg2, ...)
!%
!%          ======================== =============================================
!%          `Variables`              Description
!%          ======================== =============================================
!%          ``keys``                 Structural errors sigma hyper parameters keys
!%          ``values``               Structural errors sigma hyper parameters values
!%          ======================== =============================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Serr_Sigma_ParametersDT_initialise
!%      - Serr_Sigma_ParametersDT_copy

module mwd_serr_sigma_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Serr_Sigma_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :), allocatable :: values

    end type Serr_Sigma_ParametersDT

contains

    subroutine Serr_Sigma_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(Serr_Sigma_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nsep_sigma))
        this%keys = "..."

        allocate (this%values(mesh%ng, setup%nsep_sigma))
        this%values = -99._sp

    end subroutine Serr_Sigma_ParametersDT_initialise

    subroutine Serr_Sigma_ParametersDT_copy(this, this_copy)

        implicit none

        type(Serr_Sigma_ParametersDT), intent(in) :: this
        type(Serr_Sigma_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Serr_Sigma_ParametersDT_copy

end module mwd_serr_sigma_parameters
