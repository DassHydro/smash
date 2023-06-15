!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - MeshDT_dev
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``dx``                   Solver spatial step                                            [m]
!%          ``nrow``                 Number of rows
!%          ``ncol``                 Number of columns
!%          ``ng``                   Number of gauges
!%          ``nac``                  Number of active cells
!%          ``xmin``                 CRS x mininimum value                                          [m]
!%          ``ymax``                 CRS y maximum value                                            [m]
!%          ``flwdir``               Flow directions
!%          ``flwacc``               Flow accumulation                                              [nb of cell]
!%          ``path``                 Solver path
!%          ``active_cell``          Mask of active cells
!%          ``flwdst``               Flow distances from main outlet(s)                             [m]
!%          ``gauge_pos``            Gauges position
!%          ``code``                 Gauges code
!%          ``area``                 Drained area at gauges position                                [m2]
!%          ``rowcol_to_ind_sparse`` Matrix linking (row, col) couple to sparse storage indice (k)
!%          ``local_active_cell``    Mask of local active cells
!%
!%      Subroutine
!%      ----------
!%
!%      - MeshDT_dev_initialise
!%      - MeshDT_dev_copy

module mwd_mesh_dev

    use md_constant_dev !%only: sp
    use mwd_setup_dev !% only: SetupDT_dev

    implicit none

    type :: MeshDT_dev

        !% Notes
        !% -----
        !% MeshDT_dev Derived Type.

        real(sp) :: dx
        integer :: nrow
        integer :: ncol
        integer :: ng
        integer :: nac
        real(sp) :: xmin
        real(sp) :: ymax

        integer, dimension(:, :), allocatable :: flwdir
        integer, dimension(:, :), allocatable :: flwacc
        integer, dimension(:, :), allocatable :: path !>f90w-index
        integer, dimension(:, :), allocatable :: active_cell

        real(sp), dimension(:, :), allocatable :: flwdst
        integer, dimension(:, :), allocatable :: gauge_pos !>f90w-index
        character(20), dimension(:), allocatable :: code !>f90w-char_array
        real(sp), dimension(:), allocatable :: area

        integer, dimension(:, :), allocatable :: rowcol_to_ind_sparse !>f90w-private
        integer, dimension(:, :), allocatable :: local_active_cell !>f90w-private

    end type MeshDT_dev

contains

    subroutine MeshDT_dev_initialise(this, setup, nrow, ncol, ng)

        !% Notes
        !% -----
        !% MeshDT initialisation subroutine.

        implicit none

        type(MeshDT_dev), intent(inout) :: this
        type(SetupDT_dev), intent(inout) :: setup
        integer, intent(in) :: nrow, ncol, ng

        this%nrow = nrow
        this%ncol = ncol
        this%ng = ng

        this%xmin = 0._sp
        this%ymax = 0._sp

        allocate (this%flwdir(this%nrow, this%ncol))
        this%flwdir = -99

        allocate (this%flwacc(this%nrow, this%ncol))
        this%flwacc = -99

        allocate (this%path(2, this%nrow*this%ncol))
        this%path = -99

        allocate (this%active_cell(this%nrow, this%ncol))
        this%active_cell = 1

        allocate (this%flwdst(this%nrow, this%ncol))
        this%flwdst = -99._sp

        allocate (this%gauge_pos(this%ng, 2))

        allocate (this%code(this%ng))
        this%code = "..."

        allocate (this%area(this%ng))
        this%area = -99._sp

        if (setup%sparse_storage) then

            allocate (this%rowcol_to_ind_sparse(this%nrow, this%ncol))
            this%rowcol_to_ind_sparse = -99

        end if

        allocate (this%local_active_cell(this%nrow, this%ncol))
        this%local_active_cell = 1

    end subroutine MeshDT_dev_initialise
    
    subroutine MeshDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(MeshDT_dev), intent(in) :: this
        type(MeshDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine MeshDT_dev_copy

end module mwd_mesh_dev
