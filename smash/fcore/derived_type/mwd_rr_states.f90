!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Rr_StatesDT
!%        Matrices containting spatialized states of hydrological operators.
!%        (reservoir level ...) The matrices are updated at each time step.
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   Rainfall-runoff states keys
!%          ``values``                 Rainfall-runoff states values
!%
!§      Subroutine
!%      ----------
!%
!%      - Rr_StatesDT_initialise
!%      - Rr_StatesDT_copy

module mwd_rr_states

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Rr_StatesDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type Rr_StatesDT

contains

    subroutine Rr_StatesDT_initialise(this, setup, mesh)
        !% Default states value will be handled in Python

        implicit none

        type(Rr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nos))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nos))
        this%values = -99._sp

    end subroutine Rr_StatesDT_initialise

    subroutine Rr_StatesDT_copy(this, this_copy)

        implicit none

        type(Rr_StatesDT), intent(in) :: this
        type(Rr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Rr_StatesDT_copy

end module mwd_rr_states
