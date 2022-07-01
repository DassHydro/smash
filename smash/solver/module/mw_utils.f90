!%    This module (wrap) `mw_utils` encapsulates all SMASH utils (type, subroutines, functions)
module mw_utils
    
    use md_common !% only: sp, dp, lchar, np, ns
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    implicit none
    
    contains
    
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
                        
                            matrix(row, col) = -99._sp
                            
                        end if
                    
                    end if
                    
                end if
                
            end do
        
        end subroutine sparse_vector_to_matrix_i

        
        subroutine compute_mean_forcing(setup, mesh, input_data)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            
            integer, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: &
            & mask_gauge
            real(sp), dimension(mesh%ng) :: cml_prcp, cml_pet
            integer :: t, col, row, g, k, n, i
            
            mask_gauge = 0
            
            do g=1, mesh%ng
            
                call mask_upstream_cells(mesh%gauge_pos(1, g), &
                & mesh%gauge_pos(2, g), mesh, mask_gauge(:, : ,g))
            
            end do

            do t=1, setup%ntime_step
            
                k = 0
                cml_prcp = 0._sp
                cml_pet = 0._sp
                
                do i=1, mesh%nrow * mesh%ncol
                
                    if (mesh%path(1, i) .gt. 0 .and. &
                    & mesh%path(2, i) .gt. 0) then
                    
                        row = mesh%path(1, i)
                        col = mesh%path(2, i)
                        
                        if (mesh%global_active_cell(row, col) .eq. &
                        & 1) then
                        
                            k = k + 1
                            
                            do g=1, mesh%ng
                                
                                if (mask_gauge(row, col, g) .eq. 1) then
                                
                                    if (setup%sparse_storage) then
                                        
                                        cml_prcp(g) = cml_prcp(g) + &
                                        & input_data%sparse_prcp(k, t)
                                        cml_pet(g) = cml_pet(g) + &
                                        & input_data%sparse_pet(k, t)
                                        
                                    else
                                    
                                        cml_prcp(g) = cml_prcp(g) + &
                                        & input_data%prcp(row, col, t)
                                        cml_pet(g) = cml_pet(g) + &
                                        & input_data%pet(row, col, t)
                                    
                                    end if
                                    
                                end if
                            
                            end do
                        
                        end if
                        
                    end if
                    
                end do
                    
                do g=1, mesh%ng
            
                    n = count(mask_gauge(:, :, g) .eq. 1)
                    
                    input_data%mean_prcp(g, t) = cml_prcp(g) / n
                    input_data%mean_pet(g, t) = cml_pet(g) / n
            
                end do
                
            end do
        
        end subroutine compute_mean_forcing
        
        
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
        
        
        subroutine parameters_derived_type_copy(parameters_in, &
        & parameters_out)
            
            implicit none
            
            type(ParametersDT), intent(in) :: parameters_in
            type(ParametersDT), intent(out) :: parameters_out
            
            parameters_out = parameters_in
        
        end subroutine parameters_derived_type_copy
        
        
        subroutine states_derived_type_copy(states_in, &
        & states_out)
            
            implicit none
            
            type(StatesDT), intent(in) :: states_in
            type(StatesDT), intent(out) :: states_out
            
            states_out = states_in
        
        end subroutine states_derived_type_copy
        
        
        subroutine output_derived_type_copy(output_in, &
        & output_out)
            
            implicit none
            
            type(OutputDT), intent(in) :: output_in
            type(OutputDT), intent(out) :: output_out
            
            output_out = output_in
        
        end subroutine output_derived_type_copy
        
end module mw_utils
