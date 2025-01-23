!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%
!%      - HY_ParametersDT
!%          Matrices containting spatialized parameters of vectorial 1D network hydraulic model.
!%          (friction, bathymetry, hydraulic geoemtry)
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   Hydraulic model parameters keys
!%          ``values``                 Hydraulic model parameters values
!%
!%
!%      Subroutine
!%      ----------
!%
!%      - HY1D_ParametersDT_initialise
!%      - HY1D_ParametersDT_copy

module mwd_hy1d_parameters

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type HY1D_ParametersDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :), allocatable :: values

    end type HY1D_ParametersDT

contains

    subroutine HY1D_ParametersDT_initialise(this, setup, mesh)
        !% Default parameters value will be handled in Python

        implicit none

        type(HY1D_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nhy1dp))
        this%keys = "..."

        allocate (this%values(mesh%ncs, setup%nhy1dp))
        this%values = -99._sp

    end subroutine HY1D_ParametersDT_initialise

    subroutine HY1D_ParametersDT_copy(this, this_copy)

        implicit none

        type(HY1D_ParametersDT), intent(in) :: this
        type(HY1D_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine HY1D_ParametersDT_copy

end module mwd_hy1d_parameters
