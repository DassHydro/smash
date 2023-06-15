!%      (MW) Module Wrapped.
!% 
!%      Subroutine
!%      ----------
!%
!%      - compute_rowcol_to_ind_sparse_dev
!%      - matrix_to_sparse_vector_i_dev
!%      - matrix_to_sparse_vector_r_dev
!%      - sparse_vector_to_matrix_i_dev
!%      - sparse_vector_to_matrix_r_dev

module mw_sparse_storage_dev

    use md_constant, only: sp
    use mwd_mesh_dev, only: MeshDT_dev

    implicit none

contains

    subroutine compute_rowcol_to_ind_sparse_dev(mesh)

        !% Notes
        !% -----

        implicit none

        type(MeshDT_dev), intent(inout) :: mesh

        integer :: i, row, col, ind

        ind = 0

        do i = 1, mesh%nrow*mesh%ncol

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0) cycle

            ind = ind + 1
            mesh%rowcol_to_ind_sparse(row, col) = ind

        end do

    end subroutine compute_rowcol_to_ind_sparse_dev
    
    subroutine matrix_to_sparse_vector_i_dev(mesh, matrix, vector)
    
        implicit none
        
        type(MeshDT_dev), intent(in) :: mesh
        integer, dimension(mesh%nrow, mesh%ncol), intent(in) :: matrix
        integer, dimension(mesh%nac), intent(inout) :: vector
        
        integer :: row, col, ind
        
        do col = 1, mesh%ncol
        
            do row = 1, mesh%nrow
                
                ind = mesh%rowcol_to_ind_sparse(row, col)
                
                if (ind .le. 0) cycle
                    
                vector(ind) = matrix(row, col)
                
            end do
        
        end do 
        
    end subroutine matrix_to_sparse_vector_i_dev
    
    subroutine matrix_to_sparse_vector_r_dev(mesh, matrix, vector)
    
        implicit none
        
        type(MeshDT_dev), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: matrix
        real(sp), dimension(mesh%nac), intent(inout) :: vector
        
        integer :: row, col, ind
        
        do col=1, mesh%ncol
        
            do row=1, mesh%nrow
                
                ind = mesh%rowcol_to_ind_sparse(row, col)
                
                if (ind .le. 0) cycle
                    
                vector(ind) = matrix(row, col)
                
            end do
        
        end do 
        
    end subroutine matrix_to_sparse_vector_r_dev
    
    subroutine sparse_vector_to_matrix_i_dev(mesh, vector, matrix, no_data)
    
        implicit none
        
        type(MeshDT_dev), intent(in) :: mesh
        integer, dimension(mesh%nac), intent(in) :: vector
        integer, dimension(mesh%nrow, mesh%ncol), intent(inout) :: matrix
        integer, optional, intent(in) :: no_data
        
        integer :: row, col, ind
        integer :: no_data_value
        
        if (present(no_data)) then
            no_data_value = no_data
            
        else
            no_data_value = -99
            
        end if
        
        do col=1, mesh%ncol
        
            do row=1, mesh%nrow
                
                ind = mesh%rowcol_to_ind_sparse(row, col)
                
                if (ind .gt. 0) then
                    matrix(row, col) = vector(ind)
                    
                else
                    matrix(row, col) = no_data_value
                
                end if
                
            end do
        
        end do 
        
    end subroutine sparse_vector_to_matrix_i_dev
    
    subroutine sparse_vector_to_matrix_r_dev(mesh, vector, matrix, no_data)
    
        implicit none
        
        type(MeshDT_dev), intent(in) :: mesh
        real(sp), dimension(mesh%nac), intent(in) :: vector
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: matrix
        real(sp), optional, intent(in) :: no_data
        
        integer :: row, col, ind
        real(sp) :: no_data_value
        
        if (present(no_data)) then
            no_data_value = no_data
            
        else
            no_data_value = -99._sp
            
        end if
        
        do col=1, mesh%ncol
        
            do row=1, mesh%nrow
                
                ind = mesh%rowcol_to_ind_sparse(row, col)
                
                if (ind .gt. 0) then
                    matrix(row, col) = vector(ind)
                    
                else
                    matrix(row, col) = no_data_value
                
                end if
                
            end do
        
        end do 
        
    end subroutine sparse_vector_to_matrix_r_dev

end module mw_sparse_storage_dev
