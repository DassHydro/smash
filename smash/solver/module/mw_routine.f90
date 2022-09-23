!%      This module `mw_routine` encapsulates all SMASH routine.
!%      This module is wrapped

module mw_routine

    use mwd_common, only: sp
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_states, only: StatesDT
    use mwd_output, only: OutputDT
    use m_statistic, only: quantile, flatten2d
    
    implicit none
    
    contains
    
        subroutine copy_setup(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% SetupDT copy subroutine
        
            implicit none
            
            type(SetupDT), intent(in) :: indt
            type(SetupDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_setup


        subroutine copy_mesh(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% MeshDT copy subroutine
        
            implicit none
            
            type(MeshDT), intent(in) :: indt
            type(MeshDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_mesh


        subroutine copy_input_data(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% Input_DataDT copy subroutine
        
            implicit none
            
            type(Input_DataDT), intent(in) :: indt
            type(Input_DataDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_input_data


        subroutine copy_parameters(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT copy subroutine
        
            implicit none
            
            type(ParametersDT), intent(in) :: indt
            type(ParametersDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_parameters


        subroutine copy_states(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% StatesDT copy subroutine
        
            implicit none
            
            type(StatesDT), intent(in) :: indt
            type(StatesDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_states
        
        
        subroutine copy_output(indt, outdt)
        
            !% Notes
            !% -----
            !%
            !% OutputDT copy subroutine
        
            implicit none
            
            type(OutputDT), intent(in) :: indt
            type(OutputDT), intent(out) :: outdt
            
            outdt = indt
        
        end subroutine copy_output
        
        
!%      TODO comment        
        subroutine compute_rowcol_to_ind_sparse(mesh)
        
            implicit none
            
            type(MeshDT), intent(inout) :: mesh
            
            integer :: i, row, col, ind
            
            ind = 0
            
            do i=1, mesh%nrow * mesh%ncol
            
                if (mesh%path(1, i) .gt. 0 .and. &
                & mesh%path(2, i) .gt. 0) then
                    
                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    
                    if (mesh%active_cell(row, col) .eq. 1) then
                        
                        ind = ind + 1
                        mesh%rowcol_to_ind_sparse(row, col) = ind
                        
                    end if
                
                end if
            
            end do
        
        end subroutine compute_rowcol_to_ind_sparse
    

!%      TODO comment
        recursive subroutine mask_upstream_cells(row, col, mesh, mask)
        
            implicit none
            
            integer, intent(in) :: row, col
            type(MeshDT), intent(in) :: mesh
            logical, dimension(mesh%nrow, mesh%ncol), intent(inout) &
            & :: mask
            
            integer :: i, col_imd, row_imd
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            mask(row, col) = .true.
    
            do i=1, 8
                
                col_imd = col + dcol(i)
                row_imd = row + drow(i)
                
                if (col_imd .gt. 0 .and. col_imd .le. mesh%ncol .and. &
                &   row_imd .gt. 0 .and. row_imd .le. mesh%nrow) then
                
                    if (mesh%flwdir(row_imd, col_imd) .eq. dkind(i)) then
                        
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
                    
                    if (mesh%active_cell(row, col) .eq. 1) then
                        
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
                    
                    if (mesh%active_cell(row, col) .eq. 1) then
                        
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
                    
                    if (mesh%active_cell(row, col) .eq. 1) then
                        
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
                    
                    if (mesh%active_cell(row, col) .eq. 1) then
                        
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
        
        
!%      TODO comment
        subroutine compute_mean_forcing(setup, mesh, input_data)
                
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            
            logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
            logical, dimension(mesh%nrow, mesh%ncol) :: mask_prcp, mask_pet
            real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet
            integer :: i, j
            
            mask_gauge = .false.
            
            do i=1, mesh%ng
            
                call mask_upstream_cells(mesh%gauge_pos(i, 1), &
                & mesh%gauge_pos(i, 2), mesh, mask_gauge(:,:,i))
            
            end do
            
            do i=1, setup%ntime_step
            
                if (setup%sparse_storage) then
                        
                    call sparse_vector_to_matrix_r(mesh, input_data%sparse_prcp(:,i), matrix_prcp)
                    call sparse_vector_to_matrix_r(mesh, input_data%sparse_pet(:,i), matrix_pet)
                    
                else
                
                    matrix_prcp = input_data%prcp(:,:,i)
                    matrix_pet = input_data%pet(:,:,i)
                    
                end if
            
                do j=1, mesh%ng
                
                    mask_prcp = (matrix_prcp .ge. 0._sp .and. mask_gauge(:,:,j))
                    mask_pet = (matrix_pet .ge. 0._sp .and. mask_gauge(:,:,j))
                    
                    input_data%mean_prcp(j, i) = sum(matrix_prcp, mask=mask_prcp) / count(mask_prcp)
                    input_data%mean_pet(j, i) = sum(matrix_pet, mask=mask_pet) / count(mask_pet)
                    
                end do
            
            end do

        end subroutine compute_mean_forcing
        
        
!%      TODO comment
        subroutine compute_prcp_indice(setup, mesh, input_data)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            
            logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
            real(sp), dimension(mesh%nrow, mesh%ncol, mesh%ng) :: flwdst_gauge
            logical, dimension(mesh%nrow, mesh%ncol) :: mask
            real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix
            integer :: i, j, k
            real(sp), dimension(11) :: qtl = (/(i / 100._sp, i=0, 100, 10)/), flwdst_qtl
            real(sp), dimension(:), allocatable :: flwdst1d
            real(sp) :: minv_n, sum_p, sum_p2, sum_d, sum_d2, sum_pd, &
            & sum_pd2, mean_p, p0, p1, p2, g1, g2, md1, md2, std, &
            & mean_subp, vg, lb_flwdst, ub_flwdst

            mask_gauge = .false.

            do i=1, mesh%ng
                
                call mask_upstream_cells(mesh%gauge_pos(i, 1), &
                & mesh%gauge_pos(i, 2), mesh, mask_gauge(:,:,i))
                
                flwdst_gauge(:,:,i) = mesh%flwdst - &
                & mesh%flwdst(mesh%gauge_pos(i, 1), mesh%gauge_pos(i, 2))
                
                call flatten2d(flwdst_gauge(:,:,i), flwdst1d, mask_gauge(:,:,i))

                !% tmp needed cause array(1, n) != array(n) 
                call quantile(flwdst1d, qtl, flwdst_qtl)
                input_data%prcp_indice%flwdst_qtl(i, :) = flwdst_qtl

                input_data%prcp_indice%wf(i, 1) = 1._sp
                
                do j=2, size(qtl)
                
                    input_data%prcp_indice%wf(i, j) = input_data%prcp_indice%wf(i, j - 1) + &
                    & real(count(flwdst1d .gt. input_data%prcp_indice%flwdst_qtl(i, j - 1) .and. &
                    & flwdst1d .le. input_data%prcp_indice%flwdst_qtl(i, j)), kind=sp)
                
                end do
            
            end do
            
            do i=1, setup%ntime_step
            
                if (setup%sparse_storage) then
                
                    call sparse_vector_to_matrix_r(mesh, input_data%sparse_prcp(:,i), matrix)
                    
                else
                
                    matrix = input_data%prcp(:,:,i)
                    
                end if
            
                do j=1, mesh%ng
                    
                    mask = (matrix .ge. 0._sp .and. mask_gauge(:,:,j))
                    
                    minv_n = 1._sp / count(mask)
                    
                    sum_p = sum(matrix, mask=mask)
                    
                    !% Do not compute indices if there is no precipitation
                    if (sum_p .gt. 0._sp) then
                        
                        sum_p2 = sum(matrix * matrix, mask=mask)
                        
                        sum_d = sum(flwdst_gauge(:,:,j), mask=mask)
                        
                        sum_d2 = sum(flwdst_gauge(:,:,j) * &
                        & flwdst_gauge(:,:,j), mask=mask)
                        
                        sum_pd = sum(matrix * &
                        & flwdst_gauge(:,:,j), mask=mask)
                        
                        sum_pd2 = sum(matrix * &
                        & flwdst_gauge(:,:,j) * flwdst_gauge(:,:,j), mask=mask)
                        
                        mean_p = minv_n * sum_p
                        
                        p0 = minv_n * sum_p
                        input_data%prcp_indice%p0(j, i) = p0
                        
                        p1 = minv_n * sum_pd
                        input_data%prcp_indice%p1(j, i) = p1
                        
                        p2 = minv_n * sum_pd2
                        input_data%prcp_indice%p2(j, i) = p2
                        
                        g1 = minv_n * sum_d
                        input_data%prcp_indice%g1(j) = g1
                        
                        g2 = minv_n * sum_d2
                        input_data%prcp_indice%g2(j) = g2
                        
                        md1 = p1 / (p0 * g1)
                        input_data%prcp_indice%md1(j, i) = md1
                    
                        md2 = (1._sp / (g2 - g1 * g1)) * ((p2 / p0) - (p1 / p0) * (p1 / p0))
                        input_data%prcp_indice%md2(j, i) = md2
                        
                        std = sqrt((minv_n * sum_p2) - (mean_p * mean_p))
                        input_data%prcp_indice%std(j, i) = std
                        
                        input_data%prcp_indice%pwf(j, i, 1) = &
                        & max(0._sp , matrix(mesh%gauge_pos(j, 1), mesh%gauge_pos(j, 1)) / sum_p)
                        
                        do k=2, size(qtl)
                        
                            lb_flwdst = input_data%prcp_indice%flwdst_qtl(j, k - 1)
                            ub_flwdst = input_data%prcp_indice%flwdst_qtl(j, k)
    
                            mask = (flwdst_gauge(:,:,j) .gt. lb_flwdst &
                            & .and. flwdst_gauge(:,:,j) .le. ub_flwdst)
                            
                            if (count(mask) .eq. 0) then
                            
                                mean_subp = 0._sp
                                
                            else
                            
                                mean_subp = sum(matrix, mask=mask) / count(mask)
                                
                            end if
                            
                            input_data%prcp_indice%pwf(j, i, k) = input_data%prcp_indice%pwf(j, i, k - 1) + &
                            & mean_subp / mean_p * input_data%prcp_indice%wf(j, k)
                        
                        end do
                        
                        vg = maxval(abs(input_data%prcp_indice%pwf(j, i, :) / input_data%prcp_indice%pwf(j, i, size(qtl)) - &
                        & input_data%prcp_indice%wf(j, :) / input_data%prcp_indice%wf(j, size(qtl))))
                        input_data%prcp_indice%vg(j, i) = vg
                        
                    end if
                
                end do
                
            end do
            
        end subroutine compute_prcp_indice
        
end module mw_routine
