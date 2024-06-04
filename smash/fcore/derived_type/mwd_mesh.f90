!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - MeshDT
!%          Meshing data
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``xres``                 X cell size derived from flwdir                               [m / degree]
!%          ``yres``                 Y cell size derived from flwdir                               [m / degree]
!%          ``xmin``                 X mininimum value derived from flwdir                         [m / degree]
!%          ``ymax``                 Y maximum value derived from flwdir                           [m / degree]
!%          ``nrow``                 Number of rows
!%          ``ncol``                 Number of columns
!%          ``dx``                   X cells size (meter approximation)                            [m]
!%          ``dy``                   Y cells size (meter approximation)                            [m]
!%          ``flwdir``               Flow directions
!%          ``flwacc``               Flow accumulation                                             [m2]
!%          ``flwdst``               Flow distances from main outlet(s)                            [m]
!%          ``npar``                 Number of partition
!%          ``ncpar``                Number of cells per partition
!%          ``cscpar``               Cumulative sum of cells per partition
!%          ``cpar_to_rowcol``       Matrix linking partition cell (c) to (row, col)
!%          ``flwpar``               Flow partitions
!%          ``nac``                  Number of active cell
!%          ``active_cell``          Mask of active cell
!%          ``ng``                   Number of gauge
!%          ``gauge_pos``            Gauge position
!%          ``code``                 Gauge code
!%          ``area``                 Drained area at gauge position                                [m2]
!%          ``area_dln``             Drained area at gauge position delineated                     [m2]
!%          ``rowcol_to_ind_ac``     Matrix linking (row, col) couple to active cell indice (k)
!%          ``local_active_cell``    Mask of local active cells
!%
!%      Subroutine
!%      ----------
!%
!%      - MeshDT_initialise
!%      - MeshDT_copy

module mwd_mesh

    use md_constant !%only: sp
    use mwd_setup !% only: SetupDT

    implicit none

    type MeshDT

        real(sp) :: xres
        real(sp) :: yres

        real(sp) :: xmin
        real(sp) :: ymax

        integer :: nrow
        integer :: ncol

        real(sp), dimension(:, :), allocatable :: dx
        real(sp), dimension(:, :), allocatable :: dy

        integer, dimension(:, :), allocatable :: flwdir
        real(sp), dimension(:, :), allocatable :: flwacc
        real(sp), dimension(:, :), allocatable :: flwdst

        integer :: npar
        integer, dimension(:), allocatable :: ncpar
        integer, dimension(:), allocatable :: cscpar
        integer, dimension(:, :), allocatable :: cpar_to_rowcol !$F90W index-array
        integer, dimension(:, :), allocatable :: flwpar

        integer :: nac
        integer, dimension(:, :), allocatable :: active_cell

        integer :: ng
        integer, dimension(:, :), allocatable :: gauge_pos !$F90W index-array
        character(lchar), dimension(:), allocatable :: code !$F90W char-array
        real(sp), dimension(:), allocatable :: area
        real(sp), dimension(:), allocatable :: area_dln

        integer, dimension(:, :), allocatable :: rowcol_to_ind_ac !$F90W index-array
        integer, dimension(:, :), allocatable :: local_active_cell

    end type MeshDT

contains

    subroutine MeshDT_initialise(this, setup, nrow, ncol, npar, ng)

        implicit none

        type(MeshDT), intent(inout) :: this
        type(SetupDT), intent(inout) :: setup
        integer, intent(in) :: nrow, ncol, npar, ng

        this%nrow = nrow
        this%ncol = ncol
        this%npar = npar
        this%ng = ng

        this%xres = -99._sp
        this%yres = -99._sp

        this%xmin = -99._sp
        this%ymax = -99._sp

        allocate (this%dx(this%nrow, this%ncol))
        this%dx = -99._sp

        allocate (this%dy(this%nrow, this%ncol))
        this%dy = -99._sp

        allocate (this%flwdir(this%nrow, this%ncol))
        this%flwdir = -99

        allocate (this%flwacc(this%nrow, this%ncol))
        this%flwacc = -99._sp

        allocate (this%flwdst(this%nrow, this%ncol))
        this%flwdst = -99._sp

        allocate (this%ncpar(this%npar))
        this%ncpar = -99

        allocate (this%cscpar(this%npar))
        this%cscpar = -99

        allocate (this%cpar_to_rowcol(this%nrow*this%ncol, 2))
        this%cpar_to_rowcol = -99

        allocate (this%flwpar(this%nrow, this%ncol))
        this%flwpar = -99

        allocate (this%active_cell(this%nrow, this%ncol))
        this%active_cell = -99

        allocate (this%gauge_pos(this%ng, 2))
        this%gauge_pos = -99

        allocate (this%code(this%ng))
        this%code = "..."

        allocate (this%area(this%ng))
        this%area = -99._sp

        allocate (this%area_dln(this%ng))
        this%area_dln = -99._sp

        allocate (this%rowcol_to_ind_ac(this%nrow, this%ncol))
        this%rowcol_to_ind_ac = -99

        allocate (this%local_active_cell(this%nrow, this%ncol))
        this%local_active_cell = -99

    end subroutine MeshDT_initialise

    subroutine MeshDT_copy(this, this_copy)

        implicit none

        type(MeshDT), intent(in) :: this
        type(MeshDT), intent(out) :: this_copy

        this_copy = this

    end subroutine MeshDT_copy

end module mwd_mesh
