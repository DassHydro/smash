! Module mw_routine defined in file smash/solver/module/mw_routine.f90

subroutine f90wrap_copy_setup(indt, outdt)
    use mw_routine, only: copy_setup
    use mwd_setup, only: setupdt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type(setupdt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(setupdt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_setup(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_setup

subroutine f90wrap_copy_mesh(indt, outdt)
    use mwd_mesh, only: meshdt
    use mw_routine, only: copy_mesh
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(meshdt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_mesh(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_mesh

subroutine f90wrap_copy_input_data(indt, outdt)
    use mw_routine, only: copy_input_data
    use mwd_input_data, only: input_datadt
    implicit none
    
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type(input_datadt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(input_datadt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_input_data(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_input_data

subroutine f90wrap_copy_parameters(indt, outdt)
    use mw_routine, only: copy_parameters
    use mwd_parameters, only: parametersdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type(parametersdt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(parametersdt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_parameters(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_parameters

subroutine f90wrap_copy_states(indt, outdt)
    use mw_routine, only: copy_states
    use mwd_states, only: statesdt
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type(statesdt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(statesdt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_states(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_states

subroutine f90wrap_copy_output(indt, outdt)
    use mwd_output, only: outputdt
    use mw_routine, only: copy_output
    implicit none
    
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type(outputdt_ptr_type) :: indt_ptr
    integer, intent(in), dimension(2) :: indt
    type(outputdt_ptr_type) :: outdt_ptr
    integer, intent(out), dimension(2) :: outdt
    indt_ptr = transfer(indt, indt_ptr)
    allocate(outdt_ptr%p)
    call copy_output(indt=indt_ptr%p, outdt=outdt_ptr%p)
    outdt = transfer(outdt_ptr, outdt)
end subroutine f90wrap_copy_output

subroutine f90wrap_compute_rowcol_to_ind_sparse(mesh)
    use mw_routine, only: compute_rowcol_to_ind_sparse
    use mwd_mesh, only: meshdt
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    mesh_ptr = transfer(mesh, mesh_ptr)
    call compute_rowcol_to_ind_sparse(mesh=mesh_ptr%p)
end subroutine f90wrap_compute_rowcol_to_ind_sparse

subroutine f90wrap_mask_upstream_cells(row, col, mesh, mask, n0, n1)
    use mwd_mesh, only: meshdt
    use mw_routine, only: mask_upstream_cells
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    integer, intent(in) :: row
    integer, intent(in) :: col
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    logical, intent(inout), dimension(n0,n1) :: mask
    integer :: n0
    !f2py intent(hide), depend(mask) :: n0 = shape(mask,0)
    integer :: n1
    !f2py intent(hide), depend(mask) :: n1 = shape(mask,1)
    mesh_ptr = transfer(mesh, mesh_ptr)
    call mask_upstream_cells(row=row, col=col, mesh=mesh_ptr%p, mask=mask)
end subroutine f90wrap_mask_upstream_cells

subroutine f90wrap_sparse_matrix_to_vector_r(mesh, matrix, vector, n0, n1, n2)
    use mwd_mesh, only: meshdt
    use mw_routine, only: sparse_matrix_to_vector_r
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    real(4), intent(in), dimension(n0,n1) :: matrix
    real(4), intent(inout), dimension(n2) :: vector
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(vector) :: n2 = shape(vector,0)
    mesh_ptr = transfer(mesh, mesh_ptr)
    call sparse_matrix_to_vector_r(mesh=mesh_ptr%p, matrix=matrix, vector=vector)
end subroutine f90wrap_sparse_matrix_to_vector_r

subroutine f90wrap_sparse_matrix_to_vector_i(mesh, matrix, vector, n0, n1, n2)
    use mwd_mesh, only: meshdt
    use mw_routine, only: sparse_matrix_to_vector_i
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    integer, intent(in), dimension(n0,n1) :: matrix
    integer, intent(inout), dimension(n2) :: vector
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(vector) :: n2 = shape(vector,0)
    mesh_ptr = transfer(mesh, mesh_ptr)
    call sparse_matrix_to_vector_i(mesh=mesh_ptr%p, matrix=matrix, vector=vector)
end subroutine f90wrap_sparse_matrix_to_vector_i

subroutine f90wrap_sparse_vector_to_matrix_r(mesh, vector, matrix, na_value, n0, n1, n2)
    use mwd_mesh, only: meshdt
    use mw_routine, only: sparse_vector_to_matrix_r
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    real(4), intent(in), dimension(n0) :: vector
    real(4), intent(inout), dimension(n1,n2) :: matrix
    real(4), optional, intent(in) :: na_value
    integer :: n0
    !f2py intent(hide), depend(vector) :: n0 = shape(vector,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,0)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,1)
    mesh_ptr = transfer(mesh, mesh_ptr)
    call sparse_vector_to_matrix_r(mesh=mesh_ptr%p, vector=vector, matrix=matrix, na_value=na_value)
end subroutine f90wrap_sparse_vector_to_matrix_r

subroutine f90wrap_sparse_vector_to_matrix_i(mesh, vector, matrix, na_value, n0, n1, n2)
    use mwd_mesh, only: meshdt
    use mw_routine, only: sparse_vector_to_matrix_i
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    integer, intent(in), dimension(n0) :: vector
    integer, intent(inout), dimension(n1,n2) :: matrix
    integer, optional, intent(in) :: na_value
    integer :: n0
    !f2py intent(hide), depend(vector) :: n0 = shape(vector,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,0)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,1)
    mesh_ptr = transfer(mesh, mesh_ptr)
    call sparse_vector_to_matrix_i(mesh=mesh_ptr%p, vector=vector, matrix=matrix, na_value=na_value)
end subroutine f90wrap_sparse_vector_to_matrix_i

subroutine f90wrap_compute_mean_forcing(setup, mesh, input_data)
    use mw_routine, only: compute_mean_forcing
    use mwd_mesh, only: meshdt
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(in), dimension(2) :: input_data
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    call compute_mean_forcing(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p)
end subroutine f90wrap_compute_mean_forcing

subroutine f90wrap_compute_prcp_indice(setup, mesh, input_data)
    use mwd_mesh, only: meshdt
    use mw_routine, only: compute_prcp_indice
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    implicit none
    
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(in), dimension(2) :: input_data
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    call compute_prcp_indice(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p)
end subroutine f90wrap_compute_prcp_indice

! End of module mw_routine defined in file smash/solver/module/mw_routine.f90

