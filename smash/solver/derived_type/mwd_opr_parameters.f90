!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----

!%      - GR_A_Opr_ParametersDT
!%
!%      - GR_B_Opr_ParametersDT
!%
!%      - GR_C_Opr_ParametersDT
!%
!%      - GR_D_Opr_ParametersDT
!%
!%      - Opr_ParametersDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``gr_a``                   GR_A_Opr_ParametersDT
!%          ``gr_b``                   GR_B_Opr_ParametersDT
!%          ``gr_c``                   GR_C_Opr_ParametersDT
!%          ``gr_d``                   GR_D_Opr_ParametersDT
!%
!§      Subroutine
!%      ----------
!%
!%      - GR_A_Opr_ParametersDT_initialise
!%      - GR_B_Opr_ParametersDT_initialise
!%      - GR_C_Opr_ParametersDT_initialise
!%      - GR_D_Opr_ParametersDT_initialise
!%      - Opr_ParametersDT_initialise
!%      - GR_A_Opr_ParametersDT_copy
!%      - GR_B_Opr_ParametersDT_copy
!%      - GR_C_Opr_ParametersDT_copy
!%      - GR_D_Opr_ParametersDT_copy
!%      - Opr_ParametersDT_copy

module mwd_opr_parameters

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type GR_A_Opr_ParametersDT

        real(sp), dimension(:, :), allocatable :: cp
        real(sp), dimension(:, :), allocatable :: cft
        real(sp), dimension(:, :), allocatable :: exc
        real(sp), dimension(:, :), allocatable :: lr

    end type GR_A_Opr_ParametersDT

    type GR_B_Opr_ParametersDT

        real(sp), dimension(:, :), allocatable :: ci
        real(sp), dimension(:, :), allocatable :: cp
        real(sp), dimension(:, :), allocatable :: cft
        real(sp), dimension(:, :), allocatable :: exc
        real(sp), dimension(:, :), allocatable :: lr

    end type GR_B_Opr_ParametersDT

    type GR_C_Opr_ParametersDT

        real(sp), dimension(:, :), allocatable :: ci
        real(sp), dimension(:, :), allocatable :: cp
        real(sp), dimension(:, :), allocatable :: cft
        real(sp), dimension(:, :), allocatable :: cst
        real(sp), dimension(:, :), allocatable :: exc
        real(sp), dimension(:, :), allocatable :: lr

    end type GR_C_Opr_ParametersDT

    type GR_D_Opr_ParametersDT

        real(sp), dimension(:, :), allocatable :: cp
        real(sp), dimension(:, :), allocatable :: cft
        real(sp), dimension(:, :), allocatable :: lr

    end type GR_D_Opr_ParametersDT

    type Opr_ParametersDT

        type(GR_A_Opr_ParametersDT) :: gr_a
        type(GR_B_Opr_ParametersDT) :: gr_b
        type(GR_C_Opr_ParametersDT) :: gr_c
        type(GR_D_Opr_ParametersDT) :: gr_d

    end type Opr_ParametersDT

contains

    subroutine GR_A_Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(GR_A_Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%cp(mesh%nrow, mesh%ncol))
        this%cp = 200._sp

        allocate (this%cft(mesh%nrow, mesh%ncol))
        this%cft = 500._sp

        allocate (this%exc(mesh%nrow, mesh%ncol))
        this%exc = 0._sp

        allocate (this%lr(mesh%nrow, mesh%ncol))
        this%lr = setup%dt*(5._sp/3600._sp)

    end subroutine GR_A_Opr_ParametersDT_initialise

    subroutine GR_B_Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(GR_B_Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%ci(mesh%nrow, mesh%ncol))
        this%ci = 1e-6_sp

        allocate (this%cp(mesh%nrow, mesh%ncol))
        this%cp = 200._sp

        allocate (this%cft(mesh%nrow, mesh%ncol))
        this%cft = 500._sp

        allocate (this%exc(mesh%nrow, mesh%ncol))
        this%exc = 0._sp

        allocate (this%lr(mesh%nrow, mesh%ncol))
        this%lr = setup%dt*(5._sp/3600._sp)

    end subroutine GR_B_Opr_ParametersDT_initialise

    subroutine GR_C_Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(GR_C_Opr_ParametersDT), intent(inout) :: this
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

        allocate (this%exc(mesh%nrow, mesh%ncol))
        this%exc = 0._sp

        allocate (this%lr(mesh%nrow, mesh%ncol))
        this%lr = setup%dt*(5._sp/3600._sp)

    end subroutine GR_C_Opr_ParametersDT_initialise

    subroutine GR_D_Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(GR_D_Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%cp(mesh%nrow, mesh%ncol))
        this%cp = 200._sp

        allocate (this%cft(mesh%nrow, mesh%ncol))
        this%cft = 500._sp

        allocate (this%lr(mesh%nrow, mesh%ncol))
        this%lr = setup%dt*(5._sp/3600._sp)

    end subroutine GR_D_Opr_ParametersDT_initialise

    subroutine Opr_ParametersDT_initialise(this, setup, mesh)

        implicit none

        type(Opr_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        select case (setup%structure)

        case ("gr_a")

            call GR_A_Opr_ParametersDT_initialise(this%gr_a, setup, mesh)

        case ("gr_b")

            call GR_B_Opr_ParametersDT_initialise(this%gr_b, setup, mesh)

        case ("gr_c")

            call GR_C_Opr_ParametersDT_initialise(this%gr_c, setup, mesh)

        case ("gr_d")

            call GR_D_Opr_ParametersDT_initialise(this%gr_d, setup, mesh)

        end select

    end subroutine Opr_ParametersDT_initialise

    subroutine GR_A_Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(GR_A_Opr_ParametersDT), intent(in) :: this
        type(GR_A_Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_A_Opr_ParametersDT_copy

    subroutine GR_B_Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(GR_B_Opr_ParametersDT), intent(in) :: this
        type(GR_B_Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_B_Opr_ParametersDT_copy

    subroutine GR_C_Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(GR_C_Opr_ParametersDT), intent(in) :: this
        type(GR_C_Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_C_Opr_ParametersDT_copy

    subroutine GR_D_Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(GR_D_Opr_ParametersDT), intent(in) :: this
        type(GR_D_Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_D_Opr_ParametersDT_copy

    subroutine Opr_ParametersDT_copy(this, this_copy)

        implicit none

        type(Opr_ParametersDT), intent(in) :: this
        type(Opr_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_ParametersDT_copy

end module mwd_opr_parameters
