!%      This module `mwd_mesh` encapsulates all SMASH mesh.
!%      This module is wrapped and differentiated.
!%
!%      MeshDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``dx``                   Solver spatial step                 [m]
!%      ``nrow``                 Number of row
!%      ``ncol``                 Number of column
!%      ``ng``                   Number of gauge
!%      ``nac``                  Number of active cell
!%      ``xmin``                 CRS x mininimum value               [m]
!%      ``ymax``                 CRS y maximum value                 [m]
!%      ``flwdir``               Flow directions
!%      ``flwacc``               Flow accumulation                   [nb of cell]
!%      ``path``                 Solver path 
!%      ``active_cell``          Mask of active cell
!%      ``flwdst``               Flow distances from main outlet(s)  [m]
!%      ``gauge_pos``            Gauge position 
!%      ``code``                 Gauge code
!%      ``area``                 Drained area at gauge position      [m2]
!%
!%      </> Private
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``rowcol_to_ind_sparse`` Matrix linking (row, col) couple to sparse storage indice (k)
!%      ``local_active_cell``    Mask of local active cell (\\in active_cell)
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] MeshDT_initialise

module mwd_mesh
    
    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    
    implicit none
    
    type :: MeshDT
        
        !% Notes
        !% -----
        !% MeshDT Derived Type.
        
        real(sp) :: dx
        integer :: nrow
        integer :: ncol
        integer :: ng
        integer :: nac
        integer :: xmin
        integer :: ymax
        
        integer, dimension(:,:), allocatable :: flwdir
        integer, dimension(:,:), allocatable :: flwacc
        integer, dimension(:,:), allocatable :: path !>f90w-index
        integer, dimension(:,:), allocatable :: active_cell
        
        real(sp), dimension(:,:), allocatable :: flwdst
        integer, dimension(:,:), allocatable :: gauge_pos !>f90w-index
        character(20), dimension(:), allocatable :: code !>f90w-char_array
        real(sp), dimension(:), allocatable :: area
        
        integer, dimension(:,:), allocatable :: rowcol_to_ind_sparse !>f90w-private
        integer, dimension(:,:), allocatable :: local_active_cell !>f90w-private

    end type MeshDT
    
    contains
    
        subroutine MeshDT_initialise(this, setup, nrow, ncol, ng)
        
            !% Notes
            !% -----
            !% MeshDT initialisation subroutine.
        
            implicit none
            
            type(MeshDT), intent(inout) :: this
            type(SetupDT), intent(inout) :: setup
            integer, intent(in) :: nrow, ncol, ng
            
            this%nrow = nrow
            this%ncol = ncol
            this%ng = ng
            
            this%xmin = 0
            this%ymax = 0
            
            allocate(this%flwdir(this%nrow, this%ncol)) 
            this%flwdir = -99
            
            allocate(this%flwacc(this%nrow, this%ncol)) 
            this%flwacc = -99
            
            allocate(this%path(2, this%nrow * this%ncol)) 
            this%path = -99
            
            allocate(this%active_cell(this%nrow, this%ncol))
            this%active_cell = 1
            
            if (this%ng .gt. 0) then
            
                allocate(this%flwdst(this%nrow, this%ncol))
                this%flwdst = -99._sp
            
                allocate(this%gauge_pos(this%ng, 2))
                
                allocate(this%code(this%ng))
                this%code = "..."
                
                allocate(this%area(this%ng))
                
            end if
            
            if (setup%sparse_storage) then
                
                allocate(this%rowcol_to_ind_sparse(this%nrow, this%ncol))
                
            end if
            
            allocate(this%local_active_cell(this%nrow, this%ncol))
            this%local_active_cell = 1
            
        end subroutine MeshDT_initialise

end module mwd_mesh
