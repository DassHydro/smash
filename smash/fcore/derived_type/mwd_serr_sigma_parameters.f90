!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SErr_Sigma_ParametersDT
!%          Vectors containting hyper parameters of the temporalisation function for sigma, the standard deviation of structural errors
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
!%      - SErr_Sigma_ParametersDT_initialise
!%      - SErr_Sigma_ParametersDT_copy

module mwd_serr_sigma_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type SErr_Sigma_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :), allocatable :: values

    end type SErr_Sigma_ParametersDT

contains

    subroutine SErr_Sigma_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(SErr_Sigma_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nsep_sigma))
        this%keys = "..."

        allocate (this%values(mesh%ng, setup%nsep_sigma))
        this%values = -99._sp

    end subroutine SErr_Sigma_ParametersDT_initialise

    subroutine SErr_Sigma_ParametersDT_copy(this, this_copy)

        implicit none

        type(SErr_Sigma_ParametersDT), intent(in) :: this
        type(SErr_Sigma_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine SErr_Sigma_ParametersDT_copy

end module mwd_serr_sigma_parameters
