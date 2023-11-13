!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%
!%      - RR_ParametersDT
!%          Matrices containting spatialized parameters of hydrological operators.
!%          (reservoir max capacity, lag time ...)
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   Rainfall-runoff parameters keys
!%          ``values``                 Rainfall-runoff parameters values
!%
!%
!%      Subroutine
!%      ----------
!%
!%      - RR_ParametersDT_initialise
!%      - RR_ParametersDT_copy

module mwd_rr_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type RR_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type RR_ParametersDT

contains

    subroutine RR_ParametersDT_initialise(this, setup, mesh)
        !% Default parameters value will be handled in Python

        implicit none

        type(RR_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nrrp))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nrrp))
        this%values = -99._sp

    end subroutine RR_ParametersDT_initialise

    subroutine RR_ParametersDT_copy(this, this_copy)

        implicit none

        type(RR_ParametersDT), intent(in) :: this
        type(RR_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine RR_ParametersDT_copy

end module mwd_rr_parameters
