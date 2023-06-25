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
!%          ``ci``                     GR interception capacity
!%          ``cp``                     GR production capacity
!%          ``cft``                    GR first transfer capacity
!%          ``cst``                    GR second transfer capacity
!%          ``kexc``                   GR exchange flux
!%          ``llr``                    Linear routing lag time
!%
!%
!%      Subroutine
!%      ----------
!%
!%      - Opr_ParametersDT_initialise
!%      - Opr_ParametersDT_copy

module mwd_opr_parameters

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Opr_ParametersDT

        real(sp), dimension(:, :), allocatable :: ci
        real(sp), dimension(:, :), allocatable :: cp
        real(sp), dimension(:, :), allocatable :: cft
        real(sp), dimension(:, :), allocatable :: cst
        real(sp), dimension(:, :), allocatable :: kexc
        real(sp), dimension(:, :), allocatable :: llr
        
    end type Opr_ParametersDT

contains

    subroutine Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%ci(mesh%nrow, mesh%ncol))
        this%ci = 1e-6_sp

        allocate (this%cp(mesh%nrow, mesh%ncol))
        this%cp = 200._sp

        allocate (this%cft(mesh%nrow, mesh%ncol))
        this%cft = 500._sp

        allocate (this%cst(mesh%nrow, mesh%ncol))
        this%cst = 500._sp

        allocate (this%kexc(mesh%nrow, mesh%ncol))
        this%kexc = 0._sp

        allocate (this%llr(mesh%nrow, mesh%ncol))
        this%llr = setup%dt*(5._sp/3600._sp)

    end subroutine Opr_ParametersDT_initialise

    subroutine Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(Opr_ParametersDT), intent(in) :: this
        type(Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_ParametersDT_copy

end module mwd_opr_parameters
