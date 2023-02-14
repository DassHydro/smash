!%      This module `mw_sparse_storage` encapsulates all SMASH sparse storage routines.

module mw_sparse_storage

    use md_constant, only: sp, dp
    use mwd_mesh, only: MeshDT

    implicit none

contains

    subroutine compute_rowcol_to_ind_sparse(mesh)

        !% Notes
        !% -----
        !%
        !% (row, col) indices to (k) sparse indices subroutine
        !% Given MeshDT,
        !% it saves the link between (row, col) matrix to (k) vector indice
        !% in MeshDT%rowcol_to_sparse_indice

        implicit none

        type(MeshDT), intent(inout) :: mesh

        integer :: i, row, col, ind

        ind = 0

        do i = 1, mesh%nrow*mesh%ncol

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

    subroutine sparse_matrix_to_vector_r(mesh, matrix, vector)

        !% Notes
        !% -----
        !%
        !% Sparse matrix to vector subroutine
        !% Given MeshDT, a single precision matrix of dim(2) and size(mesh%nrow, mesh%ncol),
        !% it returns a single precision vector of dim(1) and size(mesh%nac)
        !% Flatten rule follow mesh%rowcol_to_ind_sparse
        !%
        !% See Also
        !% --------
        !% compute compute_rowcol_to_ind_sparse

        implicit none

        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) &
        & :: matrix
        real(sp), dimension(mesh%nac), intent(inout) :: vector

        integer :: i, row, col, k

        k = 0

        do i = 1, mesh%nrow*mesh%ncol

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

    subroutine sparse_matrix_to_vector_i(mesh, matrix, vector)

        !% Notes
        !% -----
        !%
        !% Sparse matrix to vector subroutine
        !% Given MeshDT, an integer matrix of dim(2) and size(mesh%nrow, mesh%ncol),
        !% it returns an integer vector of dim(1) and size(mesh%nac)
        !% Flatten rule follow mesh%rowcol_to_ind_sparse
        !%
        !% See Also
        !% --------
        !% compute compute_rowcol_to_ind_sparse

        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, dimension(mesh%nrow, mesh%ncol), intent(in) &
        & :: matrix
        integer, dimension(mesh%nac), intent(inout) :: vector

        integer :: i, row, col, k

        k = 0

        do i = 1, mesh%nrow*mesh%ncol

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

    subroutine sparse_vector_to_matrix_r(mesh, vector, matrix, &
    & na_value)

        !% Notes
        !% -----
        !%
        !% Sparse vector to matrix subroutine
        !% Given MeshDT, a single precision vector of dim(1) and size(mesh%nac),
        !% optionnaly a single precesion no-data value,
        !% it returns a single precision matrix of dim(2) and size(mesh%nrow, mesh%ncol)
        !% Unflatten rule follow mesh%rowcol_to_ind_sparse
        !%
        !% See Also
        !% --------
        !% compute compute_rowcol_to_ind_sparse

        implicit none

        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nac), intent(in) :: vector
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) &
        & :: matrix
        real(sp), optional, intent(in) :: na_value

        integer :: i, row, col, k

        k = 0

        do i = 1, mesh%nrow*mesh%ncol

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

    subroutine sparse_vector_to_matrix_i(mesh, vector, matrix, &
    & na_value)

        !% Notes
        !% -----
        !%
        !% Sparse vector to matrix subroutine
        !% Given MeshDT, an integer vector of dim(1) and size(mesh%nac),
        !% optionnaly an integer no-data value,
        !% it returns an integer matrix of dim(2) and size(mesh%nrow, mesh%ncol)
        !% Unflatten rule follow mesh%rowcol_to_ind_sparse
        !%
        !% See Also
        !% --------
        !% compute compute_rowcol_to_ind_sparse

        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, dimension(mesh%nac), intent(in) :: vector
        integer, dimension(mesh%nrow, mesh%ncol), intent(inout) &
        & :: matrix
        integer, optional, intent(in) :: na_value

        integer :: i, row, col, k

        k = 0

        do i = 1, mesh%nrow*mesh%ncol

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

end module mw_sparse_storage
