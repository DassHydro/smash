!%      (MWD) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - optimize_sbs
!%      - lbfgsb_optimize

module mw_optimize

    use md_constant, only: sp, dp, lchar
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT
    use mwd_output, only: OutputDT
    use mwd_options, only: OptionsDT
    use mwd_returns, only: ReturnsDT
    use mw_forward, only: forward_run, forward_run_b
    use mwd_parameters_manipulation, only: parameters_to_control
    use mwd_control, only: ControlDT, ControlDT_finalise

    implicit none

    public :: sbs_optimize, lbfgsb_optimize

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

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        call parameters_to_control(setup, mesh, input_data, parameters, options)

        n = size(parameters%control%x)

        allocate (x_wa(n), y_wa(n), z_wa(n), l_wa(n), u_wa(n), sdx(n))

        x_wa = parameters%control%x
        l_wa = parameters%control%l
        u_wa = parameters%control%u
        y_wa = x_wa
        z_wa = x_wa

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
            write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
            & "At iterate", 0, "nfg = ", nfg, "J =", gx, "ddx =", ddx
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
                    write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
                    & "At iterate", (iter/n), "nfg = ", nfg, "J =", gx, "ddx =", ddx
                end if

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
        real(dp) :: factr, pgtol, f
        character(lchar) :: task, csave
        logical, dimension(4) :: lsave
        integer, dimension(44) :: isave
        real(dp), dimension(29) :: dsave
        integer, dimension(:), allocatable :: iwa
        real(dp), dimension(:), allocatable :: g, wa, x_wa, l_wa, u_wa
        type(ParametersDT) :: parameters_b, parameters_bak
        type(OutputDT) :: output_b

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        call parameters_to_control(setup, mesh, input_data, parameters, options)

        iprint = -1
        n = size(parameters%control%x)
        m = 10
        factr = 1e6_dp
        pgtol = 1e-12_dp

        allocate (g(n), x_wa(n), l_wa(n), u_wa(n))
        allocate (iwa(3*n))
        allocate (wa(2*m*n + 5*n + 11*m*m + 8*m))

        parameters_bak = parameters
        parameters_b = parameters
        output_b = output
        output_b%cost = 1._sp
        output_b%sim_response%q = 0._sp

        task = "START"

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

                call forward_run_b(setup, mesh, input_data, parameters, &
                & parameters_b, output, output_b, options, returns)

                !% It's a Tapenade security, depending on the differentiation graph,
                !% some non-optimized parameters may be modified.
                parameters = parameters_bak

                f = real(output%cost, dp)
                g = real(parameters_b%control%x, dp)

                if (task(4:8) .eq. "START" .and. options%comm%verbose) then

                    write (*, '(4x,a,4x,i3,4x,a,i5,3(4x,a,f14.6),4x,a,f10.6)') &
                    & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", maxval(abs(g))

                end if

            else if (task(1:5) .eq. "NEW_X") then

                if (options%comm%verbose) then

                    write (*, '(4x,a,4x,i3,4x,a,i5,3(4x,a,f14.6),4x,a,f10.6)') &
                    & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)

                end if

                if (isave(30) .ge. options%optimize%maxiter) task = "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"

                if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) task = "STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL"

            end if

        end do

        call forward_run(setup, mesh, input_data, parameters, output, options, returns)

        call ControlDT_finalise(parameters%control)

        if (options%comm%verbose) write (*, '(4x,2a)') task, new_line("")

    end subroutine lbfgsb_optimize

end module mw_optimize
