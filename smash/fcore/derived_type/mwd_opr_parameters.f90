!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%
!%      - Opr_ParametersDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   Operator parameters keys
!%          ``values``                 Operator parameters values
!%
!%
!%      Subroutine
!%      ----------
!%
!%      - Opr_ParametersDT_initialise
!%      - Opr_ParametersDT_copy

module mwd_opr_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Opr_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :), allocatable :: values

    end type Opr_ParametersDT

contains

    subroutine Opr_ParametersDT_initialise(this, setup, mesh)
        !% Default parameters value will be handled in Python

        implicit none

        type(Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nop))
        this%keys = "..."

        allocate (this%values(mesh%nrow, mesh%ncol, setup%nop))
        this%values = -99._sp

    end subroutine Opr_ParametersDT_initialise

    subroutine Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(Opr_ParametersDT), intent(in) :: this
        type(Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_ParametersDT_copy

end module mwd_opr_parameters
