! Module mw_optimize defined in file smash/solver/module/mw_optimize.f90

subroutine f90wrap_optimize_sbs(setup, mesh, input_data, parameters, states, output)
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_parameters, only: parametersdt
    use mwd_output, only: outputdt
    use mwd_mesh, only: meshdt
    use mw_optimize, only: optimize_sbs
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
    call optimize_sbs(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        states=states_ptr%p, output=output_ptr%p)
end subroutine f90wrap_optimize_sbs

subroutine f90wrap_optimize_lbfgsb(setup, mesh, input_data, parameters, states, output)
    use mwd_setup, only: setupdt
    use mwd_input_data, only: input_datadt
    use mwd_parameters, only: parametersdt
    use mwd_output, only: outputdt
    use mwd_mesh, only: meshdt
    use mw_optimize, only: optimize_lbfgsb
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
    call optimize_lbfgsb(setup=setup_ptr%p, mesh=mesh_ptr%p, input_data=input_data_ptr%p, parameters=parameters_ptr%p, &
        states=states_ptr%p, output=output_ptr%p)
end subroutine f90wrap_optimize_lbfgsb

! End of module mw_optimize defined in file smash/solver/module/mw_optimize.f90

