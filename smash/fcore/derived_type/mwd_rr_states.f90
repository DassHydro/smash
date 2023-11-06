!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - RR_StatesDT
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
!%      - RR_StatesDT_initialise
!%      - RR_StatesDT_copy

module mwd_rr_states

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type RR_StatesDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type RR_StatesDT

contains

    subroutine RR_StatesDT_initialise(this, setup, mesh)
        !% Default states value will be handled in Python

        implicit none

        type(RR_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nrrs))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nrrs))
        this%values = -99._sp

    end subroutine RR_StatesDT_initialise

    subroutine RR_StatesDT_copy(this, this_copy)

        implicit none

        type(RR_StatesDT), intent(in) :: this
        type(RR_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine RR_StatesDT_copy

end module mwd_rr_states
