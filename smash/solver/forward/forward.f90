subroutine base_forward(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)

   use md_constant !% only: sp
   use mwd_setup !% only: SetupDT
   use mwd_mesh !% only: MeshDT
   use mwd_input_data !% only: Input_DataDT
   use mwd_parameters !% only: ParametersDT
   use mwd_states !% only: StatesDT
   use mwd_output !% only: OutputDT
   use mwd_cost !% only: compute_cost
   use md_forward_structure !% only: gr_a_forward, gr_b_forward, gr_c_forward, gr_d_forward, vic_a_forward
   use mwd_parameters_manipulation !% only: denormalize_parameters
   use mwd_states_manipulation !% only: denormalize_states

   implicit none

   !% =================================================================================================================== %!
   !%   Derived Type Variables (shared)
   !% =================================================================================================================== %!

   type(SetupDT), intent(in) :: setup
   type(MeshDT), intent(in) :: mesh
   type(Input_DataDT), intent(in) :: input_data
   type(ParametersDT), intent(inout) :: parameters
   type(ParametersDT), intent(in) :: parameters_bgd
   type(StatesDT), intent(inout) :: states
   type(StatesDT), intent(in) :: states_bgd
   type(OutputDT), intent(inout) :: output
   real(sp), intent(inout) :: cost

   type(StatesDT) :: states_imd

   if (setup%optimize%normalize_forward) then

      call denormalize_parameters(setup, mesh, parameters)
      call denormalize_states(setup, mesh, states)

   end if

   cost = 0._sp
   states_imd = states

   select case (trim(setup%structure))

   case ("gr-a")

      call gr_a_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-b")

      call gr_b_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-c")

      call gr_c_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-d")

      call gr_d_forward(setup, mesh, input_data, parameters, states, output)

   case ("vic-a")

      call vic_a_forward(setup, mesh, input_data, parameters, states, output)

   end select

   !% =================================================================================================================== %!
   !%   Store states at final time step and reset states
   !% =================================================================================================================== %!

   output%fstates = states
   states = states_imd

   !% =================================================================================================================== %!
   !%   Compute J
   !% =================================================================================================================== %!

   call compute_cost(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)

end subroutine base_forward

subroutine base_hyper_forward(setup, mesh, input_data, parameters, hyper_parameters, &
& hyper_parameters_bgd, states, hyper_states, hyper_states_bgd, output, cost)

   use md_constant !% only: sp
   use mwd_setup !% only: SetupDT
   use mwd_mesh !% only: MeshDT
   use mwd_input_data !% only: Input_DataDT
   use mwd_parameters !% only: Hyper_ParametersDT
   use mwd_states !% only: Hyper_StatesDT
   use mwd_output !% only: OutputDT
   use mwd_cost !% only: compute_hyper_cost
   use md_forward_structure !% only: gr_a_forward, gr_b_forward, gr_c_forward, gr_d_forward, vic_a_forward
   use mwd_parameters_manipulation !% only: hyper_parameters_to_parameters
   use mwd_states_manipulation !% only: hyper_states_to_states

   implicit none

   !% =================================================================================================================== %!
   !%   Derived Type Variables (shared)
   !% =================================================================================================================== %!

   type(SetupDT), intent(in) :: setup
   type(MeshDT), intent(in) :: mesh
   type(Input_DataDT), intent(inout) :: input_data
   type(ParametersDT), intent(inout) :: parameters
   type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_bgd
   type(StatesDT), intent(inout) :: states
   type(Hyper_StatesDT), intent(in) :: hyper_states, hyper_states_bgd
   type(OutputDT), intent(inout) :: output
   real(sp), intent(inout) :: cost

   !% =================================================================================================================== %!
   !%   Local Variables (private)
   !% =================================================================================================================== %!

   call hyper_parameters_to_parameters(hyper_parameters, parameters, setup, mesh, input_data)
   call hyper_states_to_states(hyper_states, states, setup, mesh, input_data)

   select case (trim(setup%structure))

   case ("gr-a")

      call gr_a_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-b")

      call gr_b_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-c")

      call gr_c_forward(setup, mesh, input_data, parameters, states, output)

   case ("gr-d")

      call gr_d_forward(setup, mesh, input_data, parameters, states, output)

   case ("vic-a")

      call vic_a_forward(setup, mesh, input_data, parameters, states, output)

   end select

   !% =================================================================================================================== %!
   !%   Store states at final time step and reset states
   !% =================================================================================================================== %!

   output%fstates = states

   !% =================================================================================================================== %!
   !%   Compute J
   !% =================================================================================================================== %!

   call hyper_compute_cost(setup, mesh, input_data, hyper_parameters, &
   & hyper_parameters_bgd, hyper_states, hyper_states_bgd, output, cost)

end subroutine base_hyper_forward
