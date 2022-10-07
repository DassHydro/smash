! Module mwd_cost defined in file smash/solver/module/mwd_cost.f90

subroutine f90wrap_compute_jobs(setup, mesh, input_data, output, jobs)
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_output, only: outputdt
    use mwd_mesh, only: meshdt
    use mwd_cost, only: compute_jobs
    implicit none
    
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
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
    type(outputdt_ptr_type) :: output_ptr
    integer, intent(in), dimension(2) :: output
    real(4), intent(out) :: jobs
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    output_ptr = transfer(output, output_ptr)
    call compute_jobs(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, output=output_ptr%p, jobs=jobs)
end subroutine f90wrap_compute_jobs

subroutine f90wrap_compute_jreg(setup, mesh, parameters, parameters_bgd, states, states_bgd, jreg)
    use mwd_setup, only: setupdt
    use mwd_parameters, only: parametersdt
    use mwd_mesh, only: meshdt
    use mwd_cost, only: compute_jreg
    use mwd_states, only: statesdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    type(parametersdt_ptr_type) :: parameters_bgd_ptr
    integer, intent(in), dimension(2) :: parameters_bgd
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    type(statesdt_ptr_type) :: states_bgd_ptr
    integer, intent(in), dimension(2) :: states_bgd
    real(4), intent(inout) :: jreg
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    parameters_ptr = transfer(parameters, parameters_ptr)
    parameters_bgd_ptr = transfer(parameters_bgd, parameters_bgd_ptr)
    states_ptr = transfer(states, states_ptr)
    states_bgd_ptr = transfer(states_bgd, states_bgd_ptr)
    call compute_jreg(setup=setup_ptr%p, mesh=mesh_ptr%p, parameters=parameters_ptr%p, parameters_bgd=parameters_bgd_ptr%p, &
        states=states_ptr%p, states_bgd=states_bgd_ptr%p, jreg=jreg)
end subroutine f90wrap_compute_jreg

subroutine f90wrap_compute_cost(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_parameters, only: parametersdt
    use mwd_output, only: outputdt
    use mwd_mesh, only: meshdt
    use mwd_cost, only: compute_cost
    use mwd_states, only: statesdt
    implicit none
    
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(in), dimension(2) :: input_data
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    type(parametersdt_ptr_type) :: parameters_bgd_ptr
    integer, intent(in), dimension(2) :: parameters_bgd
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    type(statesdt_ptr_type) :: states_bgd_ptr
    integer, intent(in), dimension(2) :: states_bgd
    type(outputdt_ptr_type) :: output_ptr
    integer, intent(in), dimension(2) :: output
    real(4), intent(inout) :: cost
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    parameters_ptr = transfer(parameters, parameters_ptr)
    parameters_bgd_ptr = transfer(parameters_bgd, parameters_bgd_ptr)
    states_ptr = transfer(states, states_ptr)
    states_bgd_ptr = transfer(states_bgd, states_bgd_ptr)
    output_ptr = transfer(output, output_ptr)
    call compute_cost(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        parameters_bgd=parameters_bgd_ptr%p, states=states_ptr%p, states_bgd=states_bgd_ptr%p, output=output_ptr%p, &
        cost=cost)
end subroutine f90wrap_compute_cost

subroutine f90wrap_nse(x, ret_res, y, n0, n1)
    use mwd_cost, only: nse
    implicit none
    
    real(4), intent(in), dimension(n0) :: x
    real(4), intent(out) :: ret_res
    real(4), intent(in), dimension(n1) :: y
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    ret_res = nse(x=x, y=y)
end subroutine f90wrap_nse

subroutine f90wrap_kge_components(x, y, r, a, b, n0, n1)
    use mwd_cost, only: kge_components
    implicit none
    
    real, intent(in), dimension(n0) :: x
    real, intent(in), dimension(n1) :: y
    real, intent(inout) :: r
    real, intent(inout) :: a
    real, intent(inout) :: b
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    call kge_components(x=x, y=y, r=r, a=a, b=b)
end subroutine f90wrap_kge_components

subroutine f90wrap_kge(x, ret_res, y, n0, n1)
    use mwd_cost, only: kge
    implicit none
    
    real, intent(in), dimension(n0) :: x
    real, intent(out) :: ret_res
    real, intent(in), dimension(n1) :: y
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    ret_res = kge(x=x, y=y)
end subroutine f90wrap_kge

subroutine f90wrap_se(x, ret_res, y, n0, n1)
    use mwd_cost, only: se
    implicit none
    
    real, intent(in), dimension(n0) :: x
    real, intent(out) :: ret_res
    real, intent(in), dimension(n1) :: y
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    ret_res = se(x=x, y=y)
end subroutine f90wrap_se

subroutine f90wrap_rmse(x, ret_res, y, n0, n1)
    use mwd_cost, only: rmse
    implicit none
    
    real, intent(in), dimension(n0) :: x
    real, intent(out) :: ret_res
    real, intent(in), dimension(n1) :: y
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    ret_res = rmse(x=x, y=y)
end subroutine f90wrap_rmse

subroutine f90wrap_logarithmique(x, ret_res, y, n0, n1)
    use mwd_cost, only: logarithmique
    implicit none
    
    real, intent(in), dimension(n0) :: x
    real, intent(out) :: ret_res
    real, intent(in), dimension(n1) :: y
    integer :: n0
    !f2py intent(hide), depend(x) :: n0 = shape(x,0)
    integer :: n1
    !f2py intent(hide), depend(y) :: n1 = shape(y,0)
    ret_res = logarithmique(x=x, y=y)
end subroutine f90wrap_logarithmique

subroutine f90wrap_reg_prior(mesh, size_mat3, matrix, ret_res, matrix_bgd, n0, n1, n2, n3, n4, n5)
    use mwd_mesh, only: meshdt
    use mwd_cost, only: reg_prior
    implicit none
    
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    integer, intent(in) :: size_mat3
    real(4), intent(in), dimension(n0,n1,n2) :: matrix
    real(4), intent(out) :: ret_res
    real(4), intent(in), dimension(n3,n4,n5) :: matrix_bgd
    integer :: n0
    !f2py intent(hide), depend(matrix) :: n0 = shape(matrix,0)
    integer :: n1
    !f2py intent(hide), depend(matrix) :: n1 = shape(matrix,1)
    integer :: n2
    !f2py intent(hide), depend(matrix) :: n2 = shape(matrix,2)
    integer :: n3
    !f2py intent(hide), depend(matrix_bgd) :: n3 = shape(matrix_bgd,0)
    integer :: n4
    !f2py intent(hide), depend(matrix_bgd) :: n4 = shape(matrix_bgd,1)
    integer :: n5
    !f2py intent(hide), depend(matrix_bgd) :: n5 = shape(matrix_bgd,2)
    mesh_ptr = transfer(mesh, mesh_ptr)
    ret_res = reg_prior(mesh=mesh_ptr%p, size_mat3=size_mat3, matrix=matrix, matrix_bgd=matrix_bgd)
end subroutine f90wrap_reg_prior

! End of module mwd_cost defined in file smash/solver/module/mwd_cost.f90

