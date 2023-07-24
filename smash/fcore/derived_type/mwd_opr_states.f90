!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Opr_StatesDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   Operator states keys
!%          ``values``                 Operator states values
!%
!§      Subroutine
!%      ----------
!%
!%      - Opr_StatesDT_initialise
!%      - Opr_StatesDT_copy

module mwd_opr_states

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Opr_StatesDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type Opr_StatesDT

contains

    subroutine Opr_StatesDT_initialise(this, setup, mesh)
        !% Default states value will be handled in Python

        implicit none

        type(Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nos))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nos))
        this%values = -99._sp

    end subroutine Opr_StatesDT_initialise

    subroutine Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(Opr_StatesDT), intent(in) :: this
        type(Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_StatesDT_copy

end module mwd_opr_states
