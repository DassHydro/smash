! Module mw_run defined in file smash/solver/module/mw_run.f90

subroutine f90wrap_forward_run(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, verbose)
    use mwd_mesh, only: meshdt
    use mwd_parameters, only: parametersdt
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_output, only: outputdt
    use mwd_states, only: statesdt
    use mw_run, only: forward_run
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
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
    logical, intent(in) :: verbose
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    parameters_ptr = transfer(parameters, parameters_ptr)
    parameters_bgd_ptr = transfer(parameters_bgd, parameters_bgd_ptr)
    states_ptr = transfer(states, states_ptr)
    states_bgd_ptr = transfer(states_bgd, states_bgd_ptr)
    output_ptr = transfer(output, output_ptr)
    call forward_run(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        parameters_bgd=parameters_bgd_ptr%p, states=states_ptr%p, states_bgd=states_bgd_ptr%p, output=output_ptr%p, &
        verbose=verbose)
end subroutine f90wrap_forward_run

subroutine f90wrap_adjoint_run(setup, mesh, input_data, parameters, states, output)
    use mwd_mesh, only: meshdt
    use mwd_parameters, only: parametersdt
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_output, only: outputdt
    use mwd_states, only: statesdt
    use mw_run, only: adjoint_run
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(in), dimension(2) :: input_data
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    type(outputdt_ptr_type) :: output_ptr
    integer, intent(in), dimension(2) :: output
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    parameters_ptr = transfer(parameters, parameters_ptr)
    states_ptr = transfer(states, states_ptr)
    output_ptr = transfer(output, output_ptr)
    call adjoint_run(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        states=states_ptr%p, output=output_ptr%p)
end subroutine f90wrap_adjoint_run

subroutine f90wrap_tangent_linear_run(setup, mesh, input_data, parameters, states, output)
    use mwd_mesh, only: meshdt
    use mwd_parameters, only: parametersdt
    use mwd_setup, only: setupdt
    use mw_run, only: tangent_linear_run
    use mwd_input_data, only: input_datadt
    use mwd_output, only: outputdt
    use mwd_states, only: statesdt
    implicit none
    
    type statesdt_ptr_type
        type(statesdt), pointer :: p => NULL()
    end type statesdt_ptr_type
    type input_datadt_ptr_type
        type(input_datadt), pointer :: p => NULL()
    end type input_datadt_ptr_type
    type parametersdt_ptr_type
        type(parametersdt), pointer :: p => NULL()
    end type parametersdt_ptr_type
    type meshdt_ptr_type
        type(meshdt), pointer :: p => NULL()
    end type meshdt_ptr_type
    type setupdt_ptr_type
        type(setupdt), pointer :: p => NULL()
    end type setupdt_ptr_type
    type outputdt_ptr_type
        type(outputdt), pointer :: p => NULL()
    end type outputdt_ptr_type
    type(setupdt_ptr_type) :: setup_ptr
    integer, intent(in), dimension(2) :: setup
    type(meshdt_ptr_type) :: mesh_ptr
    integer, intent(in), dimension(2) :: mesh
    type(input_datadt_ptr_type) :: input_data_ptr
    integer, intent(in), dimension(2) :: input_data
    type(parametersdt_ptr_type) :: parameters_ptr
    integer, intent(in), dimension(2) :: parameters
    type(statesdt_ptr_type) :: states_ptr
    integer, intent(in), dimension(2) :: states
    type(outputdt_ptr_type) :: output_ptr
    integer, intent(in), dimension(2) :: output
    setup_ptr = transfer(setup, setup_ptr)
    mesh_ptr = transfer(mesh, mesh_ptr)
    input_data_ptr = transfer(input_data, input_data_ptr)
    parameters_ptr = transfer(parameters, parameters_ptr)
    states_ptr = transfer(states, states_ptr)
    output_ptr = transfer(output, output_ptr)
    call tangent_linear_run(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        states=states_ptr%p, output=output_ptr%p)
end subroutine f90wrap_tangent_linear_run

! End of module mw_run defined in file smash/solver/module/mw_run.f90

