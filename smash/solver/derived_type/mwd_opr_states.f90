!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - GR_A_Opr_StatesDT
!%
!%      - GR_B_Opr_StatesDT
!%
!%      - GR_C_Opr_StatesDT
!%
!%      - GR_D_Opr_StatesDT
!%
!%      - Opr_StatesDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``gr_a``                   GR_A_Opr_StatesDT
!%          ``gr_b``                   GR_B_Opr_StatesDT
!%          ``gr_c``                   GR_C_Opr_StatesDT
!%          ``gr_d``                   GR_D_Opr_StatesDT
!%
!§      Subroutine
!%      ----------
!%
!%      - GR_A_Opr_StatesDT_initialise
!%      - GR_B_Opr_StatesDT_initialise
!%      - GR_C_Opr_StatesDT_initialise
!%      - GR_D_Opr_StatesDT_initialise
!%      - Opr_StatesDT_initialise
!%      - GR_A_Opr_StatesDT_copy
!%      - GR_B_Opr_StatesDT_copy
!%      - GR_C_Opr_StatesDT_copy
!%      - GR_D_Opr_StatesDT_copy
!%      - Opr_StatesDT_copy

module mwd_opr_states

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type GR_A_Opr_StatesDT

        real(sp), dimension(:, :), allocatable :: hp
        real(sp), dimension(:, :), allocatable :: hft
        real(sp), dimension(:, :), allocatable :: hlr

    end type GR_A_Opr_StatesDT

    type GR_B_Opr_StatesDT

        real(sp), dimension(:, :), allocatable :: hi
        real(sp), dimension(:, :), allocatable :: hp
        real(sp), dimension(:, :), allocatable :: hft
        real(sp), dimension(:, :), allocatable :: hlr

    end type GR_B_Opr_StatesDT

    type GR_C_Opr_StatesDT

        real(sp), dimension(:, :), allocatable :: hi
        real(sp), dimension(:, :), allocatable :: hp
        real(sp), dimension(:, :), allocatable :: hft
        real(sp), dimension(:, :), allocatable :: hst
        real(sp), dimension(:, :), allocatable :: hlr

    end type GR_C_Opr_StatesDT

    type GR_D_Opr_StatesDT

        real(sp), dimension(:, :), allocatable :: hp
        real(sp), dimension(:, :), allocatable :: hft
        real(sp), dimension(:, :), allocatable :: hlr

    end type GR_D_Opr_StatesDT

    type Opr_StatesDT

        type(GR_A_Opr_StatesDT) :: gr_a
        type(GR_B_Opr_StatesDT) :: gr_b
        type(GR_C_Opr_StatesDT) :: gr_c
        type(GR_D_Opr_StatesDT) :: gr_d

    end type Opr_StatesDT

contains

    subroutine GR_A_Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(GR_A_Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%hp(mesh%nrow, mesh%ncol))
        this%hp = 1e-2_sp

        allocate (this%hft(mesh%nrow, mesh%ncol))
        this%hft = 1e-2_sp

        allocate (this%hlr(mesh%nrow, mesh%ncol))
        this%hlr = 1e-6_sp

    end subroutine GR_A_Opr_StatesDT_initialise

    subroutine GR_B_Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(GR_B_Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%hi(mesh%nrow, mesh%ncol))
        this%hi = 1e-2_sp

        allocate (this%hp(mesh%nrow, mesh%ncol))
        this%hp = 1e-2_sp

        allocate (this%hft(mesh%nrow, mesh%ncol))
        this%hft = 1e-2_sp

        allocate (this%hlr(mesh%nrow, mesh%ncol))
        this%hlr = 1e-6_sp

    end subroutine GR_B_Opr_StatesDT_initialise

    subroutine GR_C_Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(GR_C_Opr_StatesDT), intent(inout) :: this
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

    end subroutine GR_C_Opr_StatesDT_initialise

    subroutine GR_D_Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(GR_D_Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%hp(mesh%nrow, mesh%ncol))
        this%hp = 1e-2_sp

        allocate (this%hft(mesh%nrow, mesh%ncol))
        this%hft = 1e-2_sp

        allocate (this%hlr(mesh%nrow, mesh%ncol))
        this%hlr = 1e-6_sp

    end subroutine GR_D_Opr_StatesDT_initialise

    subroutine Opr_StatesDT_initialise(this, setup, mesh)

        implicit none

        type(Opr_StatesDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        select case (setup%structure)

        case ("gr_a")

            call GR_A_Opr_StatesDT_initialise(this%gr_a, setup, mesh)

        case ("gr_b")

            call GR_B_Opr_StatesDT_initialise(this%gr_b, setup, mesh)

        case ("gr_c")

            call GR_C_Opr_StatesDT_initialise(this%gr_c, setup, mesh)

        case ("gr_d")

            call GR_D_Opr_StatesDT_initialise(this%gr_d, setup, mesh)

        end select

    end subroutine Opr_StatesDT_initialise

    subroutine GR_A_Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(GR_A_Opr_StatesDT), intent(in) :: this
        type(GR_A_Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_A_Opr_StatesDT_copy

    subroutine GR_B_Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(GR_B_Opr_StatesDT), intent(in) :: this
        type(GR_B_Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_B_Opr_StatesDT_copy

    subroutine GR_C_Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(GR_C_Opr_StatesDT), intent(in) :: this
        type(GR_C_Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_C_Opr_StatesDT_copy

    subroutine GR_D_Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(GR_D_Opr_StatesDT), intent(in) :: this
        type(GR_D_Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine GR_D_Opr_StatesDT_copy

    subroutine Opr_StatesDT_copy(this, this_copy)

        implicit none

        type(Opr_StatesDT), intent(in) :: this
        type(Opr_StatesDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Opr_StatesDT_copy

end module mwd_opr_states
