!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - MeshDT
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
!%          ``nac``                  Number of active cell
!%          ``active_cell``          Mask of active cell
!%          ``path``                 Solver path
!%          ``ng``                   Number of gauge
!%          ``gauge_pos``            Gauge position 
!%          ``code``                 Gauge code
!%          ``area``                 Drained area at gauge position                                [m2]
!%          ``area_dln``             Drained area at gauge position delineated                     [m2]
!%          ``rowcol_to_ind_sparse`` Matrix linking (row, col) couple to sparse storage indice (k)
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

    type :: MeshDT

        real(sp) :: xres
        real(sp) :: yres
        
        real(sp) :: xmin
        real(sp) :: ymax
        
        integer :: nrow
        integer :: ncol
        
        real(sp), dimension(:,:), allocatable :: dx
        real(sp), dimension(:,:), allocatable :: dy
        
        integer, dimension(:,:), allocatable :: flwdir
        real(sp), dimension(:,:), allocatable :: flwacc
        real(sp), dimension(:,:), allocatable :: flwdst
        
        integer, dimension(:,:), allocatable :: path !>f90w-index
        
        integer :: nac
        integer, dimension(:,:), allocatable :: active_cell
        
        integer :: ng
        integer, dimension(:,:), allocatable :: gauge_pos !>f90w-index
        character(20), dimension(:), allocatable :: code !>f90w-char_array
        real(sp), dimension(:), allocatable :: area
        real(sp), dimension(:), allocatable :: area_dln
        
        integer, dimension(:,:), allocatable :: rowcol_to_ind_sparse
        integer, dimension(:,:), allocatable :: local_active_cell

    end type MeshDT

contains

    subroutine MeshDT_initialise(this, setup, nrow, ncol, ng)

        implicit none

        type(MeshDT), intent(inout) :: this
        type(SetupDT), intent(inout) :: setup
        integer, intent(in) :: nrow, ncol, ng

        this%nrow = nrow
        this%ncol = ncol
        this%ng = ng
        
        this%xres = 0._sp
        this%yres = 0._sp

        this%xmin = 0._sp
        this%ymax = 0._sp

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

        allocate (this%path(2, this%nrow*this%ncol))
        this%path = -99

        allocate (this%active_cell(this%nrow, this%ncol))
        this%active_cell = 1

        allocate (this%gauge_pos(this%ng, 2))

        allocate (this%code(this%ng))
        this%code = "..."

        allocate (this%area(this%ng))
        this%area = -99._sp
        
        allocate (this%area_dln(this%ng))
        this%area_dln = -99._sp

        if (setup%sparse_storage) then

            allocate (this%rowcol_to_ind_sparse(this%nrow, this%ncol))
            this%rowcol_to_ind_sparse = -99

        end if

        allocate (this%local_active_cell(this%nrow, this%ncol))
        this%local_active_cell = 1

    end subroutine MeshDT_initialise
    
    subroutine MeshDT_copy(this, this_copy)
    
        implicit none
        
        type(MeshDT), intent(in) :: this
        type(MeshDT), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine MeshDT_copy

end module mwd_mesh
