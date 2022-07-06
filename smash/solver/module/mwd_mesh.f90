!%      This module `mwd_mesh` encapsulates all SMASH mesh.
!%      This module is wrapped and differentiated.
!%
!%      MeshDT type:
!%      
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``dx``                   Solver spatial step             [m]
!%      ``nrow``                 Number of row
!%      ``ncol``                 Number of column
!%      ``ng``                   Number of gauge
!%      ``nac``                  Number of active cell
!%      ``xmin``                 CRS x mininimum value           [m]
!%      ``ymax``                 CRS y maximum value             [m]
!%      ``flow``                 Flow directions
!%      ``drained_area``         Drained area                    [nb of cell]
!%      ``path``                 Solver path 
!%      ``gauge_pos``            Gauge position 
!%      ``gauge_optim``          Gauge to optimize
!%      ``code``                 Gauge code
!%      ``area``                 Drained area at gauge position  [m2]
!%      ``global_active_cell``   Mask of global active cell
!%      ``local_active_cell``    Mask of local active cell
!%      ``rowcol_to_ind_sparse`` Matrix linking (row, col) couple to sparse storage indice
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] MeshDT_initialise
!%      [2] mesh_copy
!%      [3] compute_rowcol_to_ind_sparse
!%      [4] mask_upstream_cells
!%      [5] sparse_matrix_to_vector_r
!%      [6] sparse_matrix_to_vector_i
!%      [7] sparse_vector_to_matrix_r
!%      [8] sparse_vector_to_matrix_i

module mwd_mesh
    
    use mwd_common !% only: sp, dp, lchar
    use mwd_setup !% only: SetupDT
    
    implicit none
    
    type :: MeshDT
    
        real(sp) :: dx
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
        integer, dimension(:), allocatable :: optim_gauge
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
            allocate(mesh%optim_gauge(mesh%ng))
            mesh%optim_gauge = 1
            
            allocate(mesh%code(mesh%ng))
            mesh%code = "...................."
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
        

!%      TODO comment
        subroutine mesh_copy(mesh_in, mesh_out)
            
            implicit none
            
            type(MeshDT), intent(in) :: mesh_in
            type(MeshDT), intent(out) :: mesh_out
            
            mesh_out = mesh_in
        
        end subroutine mesh_copy
        
!%      TODO comment        
        subroutine compute_rowcol_to_ind_sparse(mesh)
        
            implicit none
            
            type(MeshDT), intent(inout) :: mesh
            
            integer :: i, row, col, k
            
            k = 0
            
            do i=1, mesh%nrow * mesh%ncol
            
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                        k = k + 1
                        mesh%rowcol_to_ind_sparse(row, col) = k
                        
                    end if
                
                end if
            
            end do
        
        end subroutine compute_rowcol_to_ind_sparse
    
    
!%      TODO comment
        recursive subroutine mask_upstream_cells(row, col, mesh, mask)
        
            implicit none
            
            integer, intent(in) :: row, col
            type(MeshDT), intent(in) :: mesh
            integer, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: mask
            
            integer :: i, row_imd, col_imd
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            mask(row, col) = 1
    
            do i=1, 8
                
                col_imd = col + dcol(i)
                row_imd = row + drow(i)
                
                if (col_imd .gt. 0 .and. col_imd .le. mesh%ncol .and. &
                &   row_imd .gt. 0 .and. row_imd .le. mesh%nrow) then
                
                    if (mesh%flow(row_imd, col_imd) .eq. dkind(i)) then
                        
                        call mask_upstream_cells(row_imd, col_imd, &
                        & mesh, mask)
                    
                    end if
                    
                end if
            
            end do
                    
        end subroutine mask_upstream_cells
        

!%      TODO comment        
        subroutine sparse_matrix_to_vector_r(mesh, matrix, vector)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) &
            & :: matrix
            real(sp), dimension(mesh%nac), intent(inout) :: vector
            
            integer :: i, row, col, k
            
            k = 0
            
            do i=1, mesh%nrow * mesh%ncol
            
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                        k = k + 1
                        vector(k) = matrix(row, col)
                        
                    end if
                
                end if
            
            end do
        
        end subroutine sparse_matrix_to_vector_r
        

!%      TODO comment        
        subroutine sparse_matrix_to_vector_i(mesh, matrix, vector)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(mesh%nrow, mesh%ncol), intent(in) &
            & :: matrix
            integer, dimension(mesh%nac), intent(inout) :: vector
            
            integer :: i, row, col, k
            
            k = 0
            
            do i=1, mesh%nrow * mesh%ncol
            
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                        k = k + 1
                        vector(k) = matrix(row, col)
                        
                    end if
                
                end if
            
            end do
        
        end subroutine sparse_matrix_to_vector_i
        
        
!%      TODO comment
        subroutine sparse_vector_to_matrix_r(mesh, vector, matrix, &
        & na_value)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nac), intent(in) :: vector
            real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: matrix
            real(sp), optional, intent(in) :: na_value
            
            integer :: i, row, col, k
            
            k = 0
            
            do i=1, mesh%nrow * mesh%ncol
                
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                        k = k + 1
                        matrix(row, col) = vector(k)
                        
                    else
                    
                         if (present(na_value)) then
                        
                            matrix(row, col) = na_value
                            
                        else
                        
                            matrix(row, col) = -99._sp
                        
                        end if
                    
                    end if
                    
                end if
                
            end do
        
        end subroutine sparse_vector_to_matrix_r
        

!%      TODO comment
        subroutine sparse_vector_to_matrix_i(mesh, vector, matrix, &
        & na_value)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(mesh%nac), intent(in) :: vector
            integer, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: matrix
            integer, optional, intent(in) :: na_value
            
            integer :: i, row, col, k
                
            k = 0
            
            do i=1, mesh%nrow * mesh%ncol
                
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                        k = k + 1
                        matrix(row, col) = vector(k)
                        
                    else
                    
                         if (present(na_value)) then
                        
                            matrix(row, col) = na_value
                            
                        else
                        
                            matrix(row, col) = -99
                            
                        end if
                    
                    end if
                    
                end if
                
            end do
        
        end subroutine sparse_vector_to_matrix_i


end module mwd_mesh
