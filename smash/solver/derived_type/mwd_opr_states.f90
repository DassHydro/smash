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
!%          ``hi``                     GR interception state
!%          ``hp``                     GR production state
!%          ``hft``                    GR first transfer state
!%          ``hst``                    GR second transfer state
!%          ``hlr``                    Linear routing state
!%
!§      Subroutine
!%      ----------
!%
!%      - Opr_StatesDT_initialise
!%      - Opr_StatesDT_copy

module mwd_opr_states

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Opr_StatesDT

        real(sp), dimension(:, :), allocatable :: hi
        real(sp), dimension(:, :), allocatable :: hp
        real(sp), dimension(:, :), allocatable :: hft
        real(sp), dimension(:, :), allocatable :: hst
        real(sp), dimension(:, :), allocatable :: hlr

    end type Opr_StatesDT

contains

    subroutine Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%hi(mesh%nrow, mesh%ncol))
        this%hi = 1e-2_sp

        allocate (this%hp(mesh%nrow, mesh%ncol))
        this%hp = 1e-2_sp

        allocate (this%hft(mesh%nrow, mesh%ncol))
        this%hft = 1e-2_sp

        allocate (this%hst(mesh%nrow, mesh%ncol))
        this%hst = 1e-2_sp

        allocate (this%hlr(mesh%nrow, mesh%ncol))
        this%hlr = 1e-6_sp

    end subroutine Opr_StatesDT_initialise

    subroutine Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(Opr_StatesDT), intent(in) :: this
        type(Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_StatesDT_copy

end module mwd_opr_states
