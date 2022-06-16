!%    This module `m_interface` encapsulates all SMASH mesh (type, subroutines, functions)
module m_interface
    
    use m_common, only: sp, dp, lchar
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    use m_input_data, only: Input_DataDT
    
    implicit none
    
    public :: mask_upstream_cells, sparse_mesh, &
    & sparse_vector_to_matrix_r, sparse_vector_to_matrix_i, &
    & sparse_matrix_to_vector_i, sparse_matrix_to_vector_r, &
    & setup_derived_type_copy, mesh_derived_type_copy, &
    & input_data_derived_type_copy
    
    contains
    
        recursive subroutine mask_upstream_cells(row, col, mesh, mask)
        
            implicit none
            
            integer, intent(in) :: row, col
            type(MeshDT), intent(inout) :: mesh
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
        
        subroutine sparse_mesh(mesh)
        
            implicit none
            
            type(MeshDT), intent(inout) :: mesh
            
            allocate(mesh%flow_sparse(mesh%nac))
            allocate(mesh%drained_area_sparse(mesh%nac))
            
            call sparse_matrix_to_vector_i(mesh, mesh%flow, &
            & mesh%flow_sparse)
            
            call sparse_matrix_to_vector_i(mesh, mesh%drained_area, &
            & mesh%drained_area_sparse)
            
            deallocate(mesh%flow)
            deallocate(mesh%drained_area)

        end subroutine sparse_mesh
        
        subroutine sparse_matrix_to_vector_r(mesh, matrix, vector)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) &
            & :: matrix
            real(sp), dimension(mesh%nac), intent(inout) :: vector
            
            integer :: i, j, k
            
            k = 0
            do i=1, mesh%ncol
            
                do j=1, mesh%nrow
                
                    if (mesh%global_active_cell(j, i) .eq. 1) then
                        
                        k = k + 1
                        vector(k) = matrix(j, i)

                    end if
                
                end do
            
            end do
        
        end subroutine sparse_matrix_to_vector_r
        
        subroutine sparse_matrix_to_vector_i(mesh, matrix, vector)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(mesh%nrow, mesh%ncol), intent(in) &
            & :: matrix
            integer, dimension(mesh%nac), intent(inout) :: vector
            
            integer :: i, j, k
            
            k = 0
            do i=1, mesh%ncol
            
                do j=1, mesh%nrow
                
                    if (mesh%global_active_cell(j, i) .eq. 1) then
                        
                        k = k + 1
                        vector(k) = matrix(j, i)

                    end if
                
                end do
            
            end do
        
        end subroutine sparse_matrix_to_vector_i
        
        subroutine sparse_vector_to_matrix_r(mesh, vector, matrix, &
        & na_value)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nac), intent(in) :: vector
            real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: matrix
            real(sp), optional, intent(in) :: na_value
            
            integer :: i, j, k
            
            k = 0
            
            do i=1, mesh%ncol
            
                do j=1, mesh%nrow
                
                    if (mesh%global_active_cell(j, i) .eq. 1) then
                    
                        k = k + 1
                        matrix(j, i) = vector(k)
                        
                    else
                    
                        if (present(na_value)) then
                        
                            matrix(j, i) = na_value
                            
                        else
                        
                            matrix(j, i) = -99._sp
                        
                        end if
                    
                    end if
                
                end do
            
            end do
        
        end subroutine sparse_vector_to_matrix_r
        
        subroutine sparse_vector_to_matrix_i(mesh, vector, matrix, &
        & na_value)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(mesh%nac), intent(in) :: vector
            integer, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: matrix
            integer, optional, intent(in) :: na_value
            
            integer :: i, j, k
            
            k = 0
            
            do i=1, mesh%ncol
            
                do j=1, mesh%nrow
                
                    if (mesh%global_active_cell(j, i) .eq. 1) then
                    
                        k = k + 1
                        matrix(j, i) = vector(k)
                        
                    else
                    
                        if (present(na_value)) then
                        
                            matrix(j, i) = na_value
                            
                        else
                        
                            matrix(j, i) = -99
                        
                        end if
                    
                    end if
                
                end do
            
            end do
        
        end subroutine sparse_vector_to_matrix_i
        
        subroutine setup_derived_type_copy(setup_in, setup_out)
            
            implicit none
            
            type(SetupDT), intent(in) :: setup_in
            type(SetupDT), intent(out) :: setup_out
            
            setup_out = setup_in
        
        end subroutine setup_derived_type_copy
        
        subroutine mesh_derived_type_copy(mesh_in, mesh_out)
            
            implicit none
            
            type(MeshDT), intent(in) :: mesh_in
            type(MeshDT), intent(out) :: mesh_out
            
            mesh_out = mesh_in
        
        end subroutine mesh_derived_type_copy
        
        subroutine input_data_derived_type_copy(input_data_in, &
        & input_data_out)
            
            implicit none
            
            type(Input_DataDT), intent(in) :: input_data_in
            type(Input_DataDT), intent(out) :: input_data_out
            
            input_data_out = input_data_in
        
        end subroutine input_data_derived_type_copy
        
end module m_interface
