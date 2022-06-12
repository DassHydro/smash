!%    This module `m_mesh` encapsulates all SMASH mesh (type, subroutines, functions)
module m_mesh
    
    use m_common, only: dp, lchar
    use m_setup, only: SetupDT
    
    implicit none
    
    public :: MeshDT, compute_mesh_path, compute_global_active_cell, &
    & mask_upstream_cells
    
    !%      MeshDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================

    !%      ====================    ==========================================================
    
    type :: MeshDT
    
        integer :: nrow
        integer :: ncol
        integer :: ng
        integer :: xll
        integer :: yll
        
        integer, dimension(:,:), allocatable :: flow
        integer, dimension(:,:), allocatable :: drained_area
        integer, dimension(:,:), allocatable :: path
        
        integer, dimension(:,:), allocatable :: gauge_pos
        integer, dimension(:), allocatable :: gauge_optim
        character(20), dimension(:), allocatable :: code
        real(dp), dimension(:), allocatable :: area
        integer, dimension(:,:), allocatable :: global_active_cell
        integer, dimension(:,:), allocatable :: local_active_cell
        
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
            
        end subroutine MeshDT_initialise
        
        ! Deprecated subroutine (see numpy argsort)
        subroutine compute_mesh_path(mesh)
        
            implicit none
            
            type(MeshDT), intent(inout) :: mesh
            
            logical, dimension(mesh%nrow, mesh%ncol) :: mask
            integer :: max_da, da, k, i, j 

            max_da = maxval(mesh%drained_area)
            da = 0
            k = 0
            
            do while (da .le. max_da)
            
                da = da + 1
                
                where (mesh%drained_area .eq. da)
                
                    mask = .true.
                    
                else where
                
                    mask = .false.
            
                end where
                
                if (any(mask)) then
                
                    do i=1, mesh%nrow
                    
                        do j=1, mesh%ncol
                        
                            if (mask(i, j)) then
                            
                                k = k + 1
                                mesh%path(:, k) = (/i, j/)
                                
                            end if
                        
                        end do
                        
                    end do
                
                end if
                
            end do
            
        end subroutine compute_mesh_path
        
        recursive subroutine mask_upstream_cells(row, col, mesh, mask)
        
            implicit none
            
            integer, intent(in) :: row, col
            type(MeshDT), intent(inout) :: mesh
            integer, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: mask
            
            integer :: i, row_s, col_s
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            mask(row, col) = 1
    
            do i=1, 8
            
                row_s = row + drow(i)
                col_s = col + dcol(i)
                
                if (col_s .gt. 0 .and. col_s .le. mesh%ncol .and. &
                &   row_s .gt. 0 .and. row_s .le. mesh%nrow) then
                
                    if (mesh%flow(row_s, col_s) .eq. dkind(i)) then
                        
                        call mask_upstream_cells(row_s, col_s, &
                        & mesh, mask)
                    
                    end if
                    
                end if
            
            end do
                    
        end subroutine mask_upstream_cells
        
        subroutine compute_global_active_cell(setup, mesh)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(inout) :: mesh
            
            integer :: i, row, col
            
            if (setup%active_cell_only) then
            
                do i=1, mesh%ng
                
                    row = mesh%gauge_pos(1, i)
                    col = mesh%gauge_pos(2, i)
                    
                    call mask_upstream_cells(row, col, mesh, &
                    & mesh%global_active_cell)
                    
                end do
            
            else
            
                where (mesh%flow .le. 0)
            
                    mesh%global_active_cell = 0
                
                else where
                    
                    mesh%global_active_cell = 1
                    
                end where

            end if
        
            mesh%local_active_cell = mesh%global_active_cell
            
        end subroutine compute_global_active_cell
        
end module m_mesh
