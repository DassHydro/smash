!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%
!%      - Rr_ParametersDT
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
!%      - Rr_ParametersDT_initialise
!%      - Rr_ParametersDT_copy

module mwd_rr_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Rr_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type Rr_ParametersDT

contains

    subroutine Rr_ParametersDT_initialise(this, setup, mesh)
        !% Default parameters value will be handled in Python

        implicit none

        type(Rr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nop))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nop))
        this%values = -99._sp

    end subroutine Rr_ParametersDT_initialise

    subroutine Rr_ParametersDT_copy(this, this_copy)

        implicit none

        type(Rr_ParametersDT), intent(in) :: this
        type(Rr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Rr_ParametersDT_copy

end module mwd_rr_parameters
