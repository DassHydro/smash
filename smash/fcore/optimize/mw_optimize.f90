!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - sbs_optimize
!%      - lbfgsb_optimize
!%      - optimize
!%      - multiple_optimize_sample_to_parameters
!%      - multiple_optimize_save_parameters
!%      - multiple_optimize

module mw_optimize

    use md_constant, only: sp, dp, lchar
    use m_screen_display, only: display_iteration_progress
    use m_array_manipulation, only: reallocate
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_output, only: OutputDT
    use mwd_options, only: OptionsDT
    use mwd_returns, only: ReturnsDT
    use mw_forward, only: forward_run, forward_run_b
    use mwd_parameters_manipulation, only: parameters_to_control, get_serr_mu, get_serr_sigma
    use mwd_control, only: ControlDT, ControlDT_finalise

    implicit none

    public :: optimize

contains

    subroutine sbs_optimize(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        character(lchar) :: task
        integer :: n, i, j, ia, iaa, iam, jf, jfa, jfaa, nfg, iter
        real(sp) :: gx, ga, clg, ddx, dxn
        real(sp), dimension(:), allocatable :: x_wa, y_wa, z_wa, l_wa, u_wa, sdx

        call parameters_to_control(setup, mesh, input_data, parameters, options)

        n = parameters%control%n

        allocate (x_wa(n), y_wa(n), z_wa(n), l_wa(n), u_wa(n), sdx(n))

        x_wa = parameters%control%x
        l_wa = parameters%control%l
        u_wa = parameters%control%u
        y_wa = x_wa
        z_wa = x_wa

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        gx = output%cost
        ga = gx
        clg = 0.7_sp**(1._sp/real(n, sp))
        sdx = 0._sp
        ddx = 0.64_sp
        dxn = ddx
        ia = 0
        iaa = 0
        iam = 0
        jfaa = 0
        nfg = 1

        task = "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"

        if (options%comm%verbose) then
            write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f14.6,4x,a,f5.2)') &
            & "At iterate", 0, "nfg = ", nfg, "J =", gx, "ddx =", ddx
        end if

        if (returns%iter_cost_flag) then
            allocate (returns%iter_cost(options%optimize%maxiter + 1))
            returns%iter_cost(1) = gx
        end if

        do iter = 1, options%optimize%maxiter*n

            if (dxn .gt. ddx) dxn = ddx
            if (ddx .gt. 2._sp) ddx = dxn

            do i = 1, n

                x_wa = y_wa

                do j = 1, 2

                    jf = 2*j - 3
                    if (i .eq. iaa .and. jf .eq. -jfaa) cycle
                    if (y_wa(i) .le. l_wa(i) .and. jf .lt. 0) cycle
                    if (y_wa(i) .ge. u_wa(i) .and. jf .gt. 0) cycle

                    x_wa(i) = y_wa(i) + jf*ddx
                    if (x_wa(i) .lt. l_wa(i)) x_wa(i) = l_wa(i)
                    if (x_wa(i) .gt. u_wa(i)) x_wa(i) = u_wa(i)

                    parameters%control%x = x_wa
                    parameters%control%l = l_wa
                    parameters%control%u = u_wa

                    call forward_run(setup, mesh, input_data, parameters, output, options, returns)
                    nfg = nfg + 1

                    if (output%cost .lt. gx) then

                        z_wa = x_wa
                        gx = output%cost
                        ia = i
                        jfa = jf

                    end if

                end do

            end do

            iaa = ia
            jfaa = jfa

            if (ia .ne. 0) then

                y_wa = z_wa

                sdx = clg*sdx
                sdx(ia) = (1._sp - clg)*real(jfa, sp)*ddx + clg*sdx(ia)

                iam = iam + 1

                if (iam .gt. 2*n) then

                    ddx = ddx*2._sp
                    iam = 0

                end if

                if (gx .lt. ga - 2) ga = gx

            else

                ddx = ddx/2._sp
                iam = 0

            end if

            if (iter .gt. 4*n) then

                do i = 1, n

                    x_wa(i) = y_wa(i) + sdx(i)
                    if (x_wa(i) .lt. l_wa(i)) x_wa(i) = l_wa(i)
                    if (x_wa(i) .gt. u_wa(i)) x_wa(i) = u_wa(i)

                end do

                parameters%control%x = x_wa
                parameters%control%l = l_wa
                parameters%control%u = u_wa

                call forward_run(setup, mesh, input_data, parameters, output, options, returns)
                nfg = nfg + 1

                if (output%cost .lt. gx) then

                    gx = output%cost
                    jfaa = 0
                    y_wa = x_wa
                    z_wa = x_wa

                    if (gx .lt. ga - 2) ga = gx

                end if

            end if

            ia = 0

            if (mod(iter, n) .eq. 0) then

                if (options%comm%verbose) then
                    write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f14.6,4x,a,f5.2)') &
                    & "At iterate", (iter/n), "nfg = ", nfg, "J =", gx, "ddx =", ddx
                end if

                if (returns%iter_cost_flag) returns%iter_cost(iter/n + 1) = gx

            end if

            if (ddx .lt. 0.01_sp) then
                task = "CONVERGENCE: DDX < 0.01"
                exit
            end if

            if (iter .eq. options%optimize%maxiter*n) then
                task = "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"
                exit
            end if

        end do

        parameters%control%x = z_wa
        parameters%control%l = l_wa
        parameters%control%u = u_wa

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        if (returns%control_vector_flag) then
            allocate (returns%control_vector(n))
            returns%control_vector = parameters%control%x
        end if

        if (returns%serr_mu_flag) call get_serr_mu(setup, mesh, parameters, output, returns%serr_mu)
        if (returns%serr_sigma_flag) call get_serr_sigma(setup, mesh, parameters, output, returns%serr_sigma)

        if (returns%iter_cost_flag) call reallocate(returns%iter_cost, iter/n + 1)

        call ControlDT_finalise(parameters%control)

        if (options%comm%verbose) write (*, '(4x,2a)') task, new_line("")

    end subroutine sbs_optimize

    subroutine lbfgsb_optimize(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        integer :: iprint, n, m
        real(dp) :: factr, pgtol, f, projg
        character(lchar) :: task, csave
        logical, dimension(4) :: lsave
        integer, dimension(44) :: isave
        real(dp), dimension(29) :: dsave
        integer, dimension(:), allocatable :: iwa
        real(dp), dimension(:), allocatable :: g, wa, x_wa, l_wa, u_wa
        type(ParametersDT) :: parameters_b
        type(OutputDT) :: output_b

        call parameters_to_control(setup, mesh, input_data, parameters, options)

        iprint = -1
        n = parameters%control%n
        m = 10
        factr = real(options%optimize%factr, dp)
        pgtol = real(options%optimize%pgtol, dp)

        allocate (g(n), x_wa(n), l_wa(n), u_wa(n))
        allocate (iwa(3*n))
        allocate (wa(2*m*n + 5*n + 11*m*m + 8*m))

        parameters_b = parameters
        output_b = output
        output_b%cost = 1._sp

        task = "START"

        if (returns%iter_cost_flag) allocate (returns%iter_cost(options%optimize%maxiter + 1))
        if (returns%iter_projg_flag) allocate (returns%iter_projg(options%optimize%maxiter + 1))

        x_wa = real(parameters%control%x, dp)
        l_wa = real(parameters%control%l, dp)
        u_wa = real(parameters%control%u, dp)

        do while (task(1:2) .eq. "FG" .or. task .eq. "NEW_X" .or. task .eq. "START")

            call setulb(n, m, x_wa, l_wa, u_wa, parameters%control%nbd, f, g, &
            & factr, pgtol, wa, iwa, task, iprint, csave, lsave, isave, dsave)

            parameters%control%x = real(x_wa, sp)
            parameters%control%l = real(l_wa, sp)
            parameters%control%u = real(u_wa, sp)

            if (task(1:2) .eq. "FG") then

                call forward_run_b(setup, mesh, input_data, parameters, parameters_b, &
                & output, output_b, options, returns)

                f = real(output%cost, dp)
                g = real(parameters_b%control%x, dp)

                if (task(4:8) .eq. "START") then

                    call projgr(n, l_wa, x_wa, parameters%control%nbd, x_wa, g, projg)
                    if (returns%iter_cost_flag) returns%iter_cost(1) = real(f, sp)
                    if (returns%iter_projg_flag) returns%iter_projg(1) = real(projg, sp)

                    if (options%comm%verbose) then

                        write (*, '(4x,a,4x,i3,4x,a,i5,3(4x,a,f14.6),4x,a,f10.6)') &
                        & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", projg

                    end if

                end if

            else if (task(1:5) .eq. "NEW_X") then

                if (returns%iter_cost_flag) returns%iter_cost(isave(30) + 1) = real(f, sp)
                if (returns%iter_projg_flag) returns%iter_projg(isave(30) + 1) = real(dsave(13), sp)

                if (options%comm%verbose) then

                    write (*, '(4x,a,4x,i3,4x,a,i5,3(4x,a,f14.6),4x,a,f10.6)') &
                    & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)

                end if

                if (isave(30) .ge. options%optimize%maxiter) task = "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"

                if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) task = "STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL"

            end if

        end do

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        if (returns%control_vector_flag) then
            allocate (returns%control_vector(n))
            returns%control_vector = parameters%control%x
        end if

        if (returns%serr_mu_flag) call get_serr_mu(setup, mesh, parameters, output, returns%serr_mu)
        if (returns%serr_sigma_flag) call get_serr_sigma(setup, mesh, parameters, output, returns%serr_sigma)

        if (returns%iter_cost_flag) call reallocate(returns%iter_cost, isave(30) + 1)
        if (returns%iter_projg_flag) call reallocate(returns%iter_projg, isave(30) + 1)

        call ControlDT_finalise(parameters%control)

        if (options%comm%verbose) write (*, '(4x,2a)') task, new_line("")

    end subroutine lbfgsb_optimize

    subroutine optimize(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        select case (options%optimize%optimizer)

        case ("sbs")

            call sbs_optimize(setup, mesh, input_data, parameters, output, options, returns)

        case ("lbfgsb")

            call lbfgsb_optimize(setup, mesh, input_data, parameters, output, options, returns)

        end select

    end subroutine optimize

    subroutine multiple_optimize_sample_to_parameters(sample, samples_kind, samples_ind, parameters)

        implicit none

        real(sp), dimension(:), intent(in) :: sample
        integer, dimension(size(sample)), intent(in) :: samples_kind, samples_ind
        type(ParametersDT), intent(inout) :: parameters

        integer :: i

        do i = 1, size(sample)

            select case (samples_kind(i))

            case (0)
                parameters%rr_parameters%values(:, :, samples_ind(i)) = sample(i)

            case (1)
                parameters%rr_initial_states%values(:, :, samples_ind(i)) = sample(i)

                ! Should be unreachable
            case default

            end select

        end do

    end subroutine multiple_optimize_sample_to_parameters

    subroutine multiple_optimize_save_parameters(setup, parameters, options, optimized_parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(ParametersDT), intent(in) :: parameters
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(:, :, :) :: optimized_parameters

        integer :: i, j

        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .ne. 1) cycle

            j = j + 1

            optimized_parameters(:, :, j) = parameters%rr_parameters%values(:, :, i)

        end do

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .ne. 1) cycle

            j = j + 1

            optimized_parameters(:, :, j) = parameters%rr_initial_states%values(:, :, i)

        end do

    end subroutine multiple_optimize_save_parameters

    subroutine multiple_optimize(setup, mesh, input_data, parameters, output, options, &
    & samples, samples_kind, samples_ind, cost, q, optimized_parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(inout) :: options
        real(sp), dimension(:, :), intent(in) :: samples
        integer, dimension(size(samples, 1)) :: samples_kind, samples_ind
        real(sp), dimension(size(samples, 2)), intent(inout) :: cost
        real(sp), dimension(mesh%ng, setup%ntime_step, size(samples, 2)), intent(inout) :: q
        real(sp), dimension(mesh%nrow, mesh%ncol, &
        & sum(options%optimize%rr_parameters) + sum(options%optimize%rr_initial_states), &
        & size(samples, 2)) :: optimized_parameters

        integer :: i, iter, niter, ncpu
        logical :: verbose
        character(lchar) :: task
        type(ParametersDT) :: parameters_thread
        type(OutputDT) :: output_thread
        type(ReturnsDT) :: returns

        niter = size(samples, 2)
        iter = 0
        task = "Optimize"

        ! Trigger parallel in multiple optimize and not inside optimize
        ncpu = options%comm%ncpu
        options%comm%ncpu = 1

        ! Deactivate other verbose
        verbose = options%comm%verbose
        options%comm%verbose = .false.

        if (verbose) call display_iteration_progress(iter, niter, task)
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(ncpu) &
        !$OMP& shared(setup, mesh, input_data, parameters, output, options, returns) &
        !$OMP& shared(samples, samples_kind, samples_ind, iter, niter, cost, q, optimized_parameters) &
        !$OMP& private(i, parameters_thread, output_thread)
#endif
        do i = 1, niter

            parameters_thread = parameters
            output_thread = output

            call multiple_optimize_sample_to_parameters(samples(:, i), samples_kind, samples_ind, parameters_thread)

            call optimize(setup, mesh, input_data, parameters_thread, output_thread, options, returns)
#ifdef _OPENMP
            !$OMP critical
#endif
            cost(i) = output_thread%cost
            q(:, :, i) = output_thread%response%q
            call multiple_optimize_save_parameters(setup, parameters_thread, options, optimized_parameters(:, :, :, i))

            iter = iter + 1
            if (verbose) call display_iteration_progress(iter, niter, task)
#ifdef _OPENMP
            !$OMP end critical
#endif
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine multiple_optimize

end module mw_optimize
