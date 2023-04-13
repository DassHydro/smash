!%      This module `mw_multiple_run` encapsulates all SMASH multiple run routines.
!%      This module is wrapped

module mw_multiple_run

    use md_constant, only: sp, GNP, GNS
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_states, only: StatesDT
    use mwd_output, only: OutputDT
    use mw_forward, only: forward
    use mwd_parameters_manipulation, only: get_parameters, set_parameters
    use mwd_states_manipulation, only: get_states, set_states
    use mwd_cost, only: nse

    implicit none

contains

    subroutine wait_bar_multiple_run(iter, niter)

        implicit none

        integer, intent(in) :: iter, niter
        integer :: per

        per = 100*iter/niter

        if (per /= 100*(iter - 1)/niter) then
            write (*, "(a,4x,a,1x,i0,a)", advance="no") &
            & achar(13), "Processing:", per, "%"
        end if

        if (iter == niter) write (*, "(a)") ""

    end subroutine wait_bar_multiple_run

    subroutine set_sample_to_parameters_states(mesh, parameters, states, ind_parameters_states, sample_arr)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        integer, dimension(:), intent(in) :: ind_parameters_states
        real(sp), dimension(:), intent(in) :: sample_arr

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        integer :: i

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        do i = 1, size(sample_arr)

            matrix(:, :, ind_parameters_states(i)) = sample_arr(i)

        end do

        call set_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call set_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

    end subroutine set_sample_to_parameters_states

    !% Only works with uniform parameters/states values
    subroutine compute_multiple_run(setup, mesh, input_data, parameters, states, output, &
    & sample, ind_parameters_states, res_cost, res_qsim)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output
        real(sp), dimension(:, :), intent(in) :: sample
        integer, dimension(:), intent(in) :: ind_parameters_states
        real(sp), dimension(:), intent(inout) :: res_cost
        real(sp), dimension(:, :, :), intent(inout) :: res_qsim

        type(ParametersDT) :: parameters_bgd, parameters_thread
        type(StatesDT) :: states_bgd, states_thread
        type(OutputDT) :: output_thread
        integer :: i, iter, niter
        real(sp) :: cost

        parameters_bgd = parameters
        states_bgd = states

        iter = 0
        niter = size(sample, 2)

        !$omp parallel do num_threads(setup%ncpu) default(shared), &
        !$omp& private(i, parameters_thread, states_thread, output_thread, cost)
        do i = 1, niter

            !$omp critical
            iter = iter + 1
            call wait_bar_multiple_run(iter, niter)
            !$omp end critical

            parameters_thread = parameters
            states_thread = states
            output_thread = output

            call set_sample_to_parameters_states(mesh, parameters_thread, states_thread, ind_parameters_states, sample(:, i))

            call forward(setup, mesh, input_data, parameters_thread, parameters_bgd, &
            & states_thread, states_bgd, output_thread, cost)

            res_cost(i) = cost

            if (size(res_qsim) .gt. 0) res_qsim(:, :, i) = output_thread%qsim

        end do
        !$omp end parallel do

    end subroutine compute_multiple_run

end module mw_multiple_run
