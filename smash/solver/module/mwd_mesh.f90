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
!%      ``drained_area``         Drained area                        [nb of cell]
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
!%      ``wgauge``               Objective function gauge weight
!%      ``rowcol_to_ind_sparse`` Matrix linking (row, col) couple to sparse storage indice (k)
!%      ``local_active_cell``    Mask of local active cell (\in active_cell)
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] MeshDT_initialise

module mwd_mesh
    
    use mwd_common !% only: sp
    use mwd_setup !% only: SetupDT
    
    implicit none
    
    type :: MeshDT
        
        !% </> Public
        real(sp) :: dx
        integer :: nrow
        integer :: ncol
        integer :: ng
        integer :: nac
        integer :: xmin
        integer :: ymax
        
        integer, dimension(:,:), allocatable :: flwdir
        integer, dimension(:,:), allocatable :: drained_area
        integer, dimension(:,:), allocatable :: path
        integer, dimension(:,:), allocatable :: active_cell
        
        real(sp), dimension(:,:), allocatable :: flwdst
        integer, dimension(:,:), allocatable :: gauge_pos
        character(20), dimension(:), allocatable :: code
        real(sp), dimension(:), allocatable :: area
        
        !% </> Private
        real(sp), dimension(:), allocatable :: wgauge !>f90wrap private
        integer, dimension(:,:), allocatable :: rowcol_to_ind_sparse !>f90wrap private
        integer, dimension(:,:), allocatable :: local_active_cell !>f90wrap private

    end type MeshDT
    
    contains
    
        subroutine MeshDT_initialise(mesh, setup, nrow, ncol, ng)
        
            !% Notes
            !% -----
            !%
            !% MeshDT initialisation subroutine
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            integer, intent(in) :: nrow, ncol, ng
            
            mesh%nrow = nrow
            mesh%ncol = ncol
            mesh%ng = ng
            
            mesh%xmin = 0
            mesh%ymax = 0
            
            allocate(mesh%flwdir(mesh%nrow, mesh%ncol)) 
            mesh%flwdir = -99
            
            allocate(mesh%drained_area(mesh%nrow, mesh%ncol)) 
            mesh%drained_area = -99
            
            allocate(mesh%path(2, mesh%nrow * mesh%ncol)) 
            mesh%path = -99
            
            allocate(mesh%active_cell(mesh%nrow, mesh%ncol))
            mesh%active_cell = 1
            
            if (mesh%ng .gt. 0) then
            
                allocate(mesh%flwdst(mesh%nrow, mesh%ncol))
                mesh%flwdst = -99._sp
            
                allocate(mesh%gauge_pos(mesh%ng, 2))
                
                allocate(mesh%code(mesh%ng))
                mesh%code = "..."
                
                allocate(mesh%area(mesh%ng))
                
                allocate(mesh%wgauge(mesh%ng))
                mesh%wgauge = 1._sp
                
            end if
            
            if (setup%sparse_storage) then
                
                allocate(mesh%rowcol_to_ind_sparse(mesh%nrow, mesh%ncol))
                
            end if
            
            allocate(mesh%local_active_cell(mesh%nrow, mesh%ncol))
            mesh%local_active_cell = 1
            
        end subroutine MeshDT_initialise

end module mwd_mesh
