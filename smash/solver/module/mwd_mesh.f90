!%      This module `mwd_mesh` encapsulates all SMASH mesh.
!%      This module is wrapped and differentiated.

!%      mwd_mesh MeshDT type:
!%      
!%      ======================== =======================================
!%      `variables`              Description
!%      ======================== =======================================
!%      ``nrow``                 Number of row
!%      ``ncol``                 Number of column
!%      ``ng``                   Number of gauge
!%      ``nac``                  Number of active cell
!%      ``xmin``                 CRS x mininimum value [m] (x left)
!%      ``ymax``                 CRS y maximum value [m] (y upper)
!%      ``flow``                 Flow directions
!%      ``drained_area``         Drained area [nb of cell]
!%      ``path``                 Solver path (i.e. ascending sort of drained area)
!%      ``gauge_pos``            Gauge position (row, col)
!%      ``gauge_optim``          Gauge to optimize
!%      ``code``                 Gauge code
!%      ``area``                 Drained area at gauge position [km2]
!%      ``global_active_cell``   Mask of global active cell (i.e. cells that contribute to any gauge)
!%      ``local_active_cell``    Mask of local active cell (i.e.  equal to or a subset of global active cell) 
!%      ``rowcol_to_ind_sparse`` Matrix linking any (row, col) couple to his sparse storage indice
!%      ======================== =======================================

module mwd_mesh
    
    use md_common !% only: sp, dp, lchar
    use mwd_setup !% only: SetupDT
    
    implicit none
    
    type :: MeshDT
    
        integer :: nrow
        integer :: ncol
        integer :: ng
        integer :: nac
        integer :: xmin
        integer :: ymax
        
        integer, dimension(:,:), allocatable :: flow
        integer, dimension(:,:), allocatable :: drained_area
        integer, dimension(:,:), allocatable :: path
        
        integer, dimension(:,:), allocatable :: gauge_pos
        integer, dimension(:), allocatable :: gauge_optim
        character(20), dimension(:), allocatable :: code
        real(sp), dimension(:), allocatable :: area
        
        integer, dimension(:,:), allocatable :: global_active_cell
        integer, dimension(:,:), allocatable :: local_active_cell
        
        integer, dimension(:,:), allocatable :: rowcol_to_ind_sparse

    end type MeshDT
    
    contains
    
        subroutine MeshDT_initialise(mesh, setup, nrow, ncol, ng)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(inout) :: mesh
            integer, intent(in) :: nrow, ncol, ng
            
            mesh%nrow = nrow
            mesh%ncol = ncol
            mesh%ng = ng
            
            allocate(mesh%flow(mesh%nrow, mesh%ncol)) 
            mesh%flow = -99
            allocate(mesh%drained_area(mesh%nrow, mesh%ncol)) 
            mesh%drained_area = -99
            
            allocate(mesh%path(2, mesh%nrow * mesh%ncol)) 
            mesh%path = -99
            
            allocate(mesh%gauge_pos(2, mesh%ng))
            allocate(mesh%gauge_optim(mesh%ng))
            mesh%gauge_optim = 1
            
            allocate(mesh%code(mesh%ng))
            allocate(mesh%area(mesh%ng))
            
            allocate(mesh%global_active_cell(mesh%nrow, mesh%ncol))
            mesh%global_active_cell = 0
            allocate(mesh%local_active_cell(mesh%nrow, mesh%ncol))
            mesh%local_active_cell = 0
            
            if (setup%sparse_storage) then
                
                allocate(mesh%rowcol_to_ind_sparse(mesh%nrow, mesh%ncol))
                mesh%rowcol_to_ind_sparse = -99
                
            end if
            
        end subroutine MeshDT_initialise
        
        
end module mwd_mesh
