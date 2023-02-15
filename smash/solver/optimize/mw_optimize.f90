!%      This module `mw_optimize` encapsulates all SMASH optimize.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1]  optimize_sbs
!%      [2]  transformation
!%      [3]  inv_transformation
!%      [4]  optimize_lbfgsb
!%      [5]  var_to_control
!%      [6]  control_to_var
!%      [7]  hyper_optimize_lbfgsb
!%      [8]  normalize_descritptor
!%      [9]  denormalize_descriptor
!%      [10] hyper_problem_initialise
!%      [11] hyper_var_to_control
!%      [12] hyper_control_to_var

module mw_optimize

    use md_constant, only: sp, dp, lchar, GNP, GNS
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mwd_parameters, only: ParametersDT, Hyper_ParametersDT, &
    & ParametersDT_initialise, Hyper_ParametersDT_initialise
    use mwd_states, only: StatesDT, Hyper_StatesDT, &
    & StatesDT_initialise, Hyper_StatesDT_initialise
    use mwd_output, only: OutputDT, OutputDT_initialise
    use mw_forward, only: forward, forward_b, hyper_forward, &
    & hyper_forward_b
    use mwd_parameters_manipulation, only: get_parameters, set_parameters, &
    & get_hyper_parameters, set_hyper_parameters, &
    & hyper_parameters_to_parameters, normalize_parameters
    use mwd_states_manipulation, only: get_states, set_states, &
    & get_hyper_states, set_hyper_states, &
    & hyper_states_to_states, normalize_states

    implicit none

    public :: optimize_sbs, optimize_lbfgsb, optimize_hyper_lbfgsb

    private :: bounds_initialise_sbs, &
    & var_to_control_sbs, control_to_var_sbs, &
    & transformation_sbs, inv_transformation_sbs, &
    & var_to_control_lbfgsb, control_to_var_lbfgsb, &
    & normalize_descriptor_hyper_lbfgsb, denormalize_descriptor_hyper_lbfgsb, &
    & problem_initialise_hyper_lbfgsb, &
    & var_to_control_hyper_lbfgsb, control_to_var_hyper_lbfgsb

contains

    subroutine optimize_sbs(setup, mesh, input_data, parameters, states, output)

        !% Notes
        !% -----
        !% Step By Step optimization subroutine.

        implicit none

        type(SetupDT), intent(inout) :: setup
        type(MeshDT), intent(inout) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        type(ParametersDT) :: parameters_bgd
        type(StatesDT) :: states_bgd
        integer :: n, i, j, iter, nfg, ia, iaa, iam, jf, jfa, jfaa
        real(sp), dimension(:), allocatable :: x, y, z, l, u, x_t, y_t, z_t, l_t, u_t, sdx
        real(sp) :: gx, ga, clg, ddx, dxn, f, cost

        ! =========================================================================================================== !
        !   Initialisation
        ! =========================================================================================================== !

        parameters_bgd = parameters
        states_bgd = states

        call forward(setup, mesh, input_data, parameters, &
        & parameters_bgd, states, states_bgd, output, cost)

        n = count(setup%optimize%optim_parameters .gt. 0) + &
        & count(setup%optimize%optim_states .gt. 0)

        allocate (x(n), y(n), z(n), l(n), u(n), x_t(n), y_t(n), z_t(n), l_t(n), u_t(n), sdx(n))

        call bounds_initialise_sbs(n, setup, l, u)

        call var_to_control_sbs(n, setup, mesh, parameters, states, x)

        call transformation_sbs(n, x, l, u, x_t)
        call transformation_sbs(n, l, l, u, l_t)
        call transformation_sbs(n, u, l, u, u_t)

        gx = cost
        ga = gx
        clg = 0.7_sp**(1._sp/real(n, kind(x)))
        z_t = x_t
        sdx = 0._sp
        ddx = 0.64_sp
        dxn = ddx
        ia = 0
        iaa = 0
        iam = 0
        jfaa = 0
        nfg = 1

        if (setup%optimize%verbose) then
            write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
            & "At iterate", 0, "nfg = ", nfg, "J =", gx, "ddx =", ddx
        end if

        do iter = 1, setup%optimize%maxiter*n

            ! ======================================================================================================= !
            !   Optimize
            ! ======================================================================================================= !

            if (dxn .gt. ddx) dxn = ddx
            if (ddx .gt. 2._sp) ddx = dxn

            do i = 1, n

                y_t = x_t

                do 7 j = 1, 2

                    jf = 2*j - 3
                    if (i .eq. iaa .and. jf .eq. -jfaa) goto 7
                    if (x_t(i) .le. l_t(i) .and. jf .lt. 0) goto 7
                    if (x_t(i) .ge. u_t(i) .and. jf .gt. 0) goto 7

                    y_t(i) = x_t(i) + jf*ddx
                    if (y_t(i) .lt. l_t(i)) y_t(i) = l_t(i)
                    if (y_t(i) .gt. u_t(i)) y_t(i) = u_t(i)

                    call inv_transformation_sbs(n, y, l, u, y_t)

                    call control_to_var_sbs(n, setup, mesh, parameters, states, y)

                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)

                    f = cost
                    nfg = nfg + 1

                    if (f .lt. gx) then

                        z_t = y_t
                        gx = f
                        ia = i
                        jfa = jf

                    end if

7               continue

            end do

            iaa = ia
            jfaa = jfa

            if (ia .ne. 0) then

                x_t = z_t

                call inv_transformation_sbs(n, x, l, u, x_t)

                call control_to_var_sbs(n, setup, mesh, parameters, states, x)

                sdx = clg*sdx
                sdx(ia) = (1._sp - clg)*real(jfa, kind(x))*ddx + clg*sdx(ia)

                iam = iam + 1

                if (iam .gt. 2*n) then

                    ddx = ddx*2._sp
                    iam = 0

                end if

                if (gx .lt. ga - 2) then

                    ga = gx

                end if

            else

                ddx = ddx/2._sp
                iam = 0

            end if

            if (iter .gt. 4*n) then

                do i = 1, n

                    y_t(i) = x_t(i) + sdx(i)
                    if (y_t(i) .lt. l_t(i)) y_t(i) = l_t(i)
                    if (y_t(i) .gt. u_t(i)) y_t(i) = u_t(i)

                end do

                call inv_transformation_sbs(n, y, l, u, y_t)

                call control_to_var_sbs(n, setup, mesh, parameters, states, y)

                call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                & states, states_bgd, output, cost)

                f = cost

                nfg = nfg + 1

                if (f .lt. gx) then

                    gx = f
                    jfaa = 0
                    x_t = y_t

                    call inv_transformation_sbs(n, x, l, u, x_t)

                    call control_to_var_sbs(n, setup, mesh, parameters, states, x)

                    if (gx .lt. ga - 2) then

                        ga = gx

                    end if

                end if

            end if

            ia = 0

            ! ======================================================================================================= !
            !   Iterate writting
            ! ======================================================================================================= !

            if (mod(iter, n) .eq. 0) then

                if (setup%optimize%verbose) then
                    write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
                    & "At iterate", (iter/n), "nfg = ", nfg, "J =", gx, "ddx =", ddx
                end if

            end if

            ! ======================================================================================================= !
            !   Convergence DDX < 0.01
            ! ======================================================================================================= !

            if (ddx .lt. 0.01_sp) then

                if (setup%optimize%verbose) then
                    write (*, '(4x,a)') "CONVERGENCE: DDX < 0.01"
                end if

                call control_to_var_sbs(n, setup, mesh, parameters, states, x)

                call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                & states, states_bgd, output, cost)

                exit

            end if

            ! ======================================================================================================= !
            !   Maximum Number of Iteration
            ! ======================================================================================================= !

            if (iter .eq. setup%optimize%maxiter*n) then

                if (setup%optimize%verbose) then
                    write (*, '(4x,a)') "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"
                end if

                call control_to_var_sbs(n, setup, mesh, parameters, states, x)

                call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                & states, states_bgd, output, cost)

                exit

            end if

        end do

    end subroutine optimize_sbs

    subroutine bounds_initialise_sbs(n, setup, l, u)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        real(sp), dimension(n), intent(inout) :: l, u

        real(sp), dimension(GNP + GNS) :: lb, ub
        integer, dimension(GNP + GNS) :: optim
        integer :: i, j

        lb(1:GNP) = setup%optimize%lb_parameters
        lb(GNP + 1:GNP + GNS) = setup%optimize%lb_states

        ub(1:GNP) = setup%optimize%ub_parameters
        ub(GNP + 1:GNP + GNS) = setup%optimize%ub_states

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        j = 0

        do i = 1, GNP + GNS

            if (optim(i) .gt. 0) then

                j = j + 1

                l(j) = lb(i)
                u(j) = ub(i)

            end if

        end do

    end subroutine bounds_initialise_sbs

    subroutine var_to_control_sbs(n, setup, mesh, parameters, states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(in) :: states
        real(sp), dimension(n), intent(inout) :: x

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        integer, dimension(GNP + GNS) :: optim
        integer, dimension(2) :: ind_ac
        integer :: i, j

        ind_ac = maxloc(mesh%active_cell)

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        j = 0

        do i = 1, GNP + GNS

            if (optim(i) .gt. 0) then

                j = j + 1

                x(j) = matrix(ind_ac(1), ind_ac(2), i)

            end if

        end do

    end subroutine var_to_control_sbs

    subroutine control_to_var_sbs(n, setup, mesh, parameters, states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        real(sp), dimension(n), intent(in) :: x

        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        integer, dimension(GNP + GNS) :: optim
        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        integer :: i, j

        mask = (mesh%active_cell .eq. 1)

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        j = 0

        do i = 1, GNP + GNS

            if (optim(i) .gt. 0) then

                j = j + 1

                where (mask)

                    matrix(:, :, i) = x(j)

                end where

            end if

        end do

        call set_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call set_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

    end subroutine control_to_var_sbs

!%      TODO comment
    subroutine transformation_sbs(n, x, l, u, x_t)

        implicit none

        integer, intent(in) :: n
        real(sp), dimension(n), intent(in) :: x, l, u
        real(sp), dimension(n), intent(inout) :: x_t

        integer :: i

        do i = 1, n

            if (l(i) .lt. 0._sp) then

                x_t(i) = asinh(x(i))

            else if (l(i) .ge. 0._sp .and. u(i) .le. 1._sp) then

                x_t(i) = log(x(i)/(1._sp - x(i)))

            else

                x_t(i) = log(x(i))

            end if

        end do

    end subroutine transformation_sbs

!%      TODO comment
    subroutine inv_transformation_sbs(n, x, l, u, x_t)

        implicit none

        integer, intent(in) :: n
        real(sp), dimension(n), intent(inout) :: x
        real(sp), dimension(n), intent(in) :: x_t, l, u

        integer :: i

        do i = 1, n

            if (l(i) .lt. 0._sp) then

                x(i) = sinh(x_t(i))

            else if (l(i) .ge. 0._sp .and. u(i) .le. 1._sp) then

                x(i) = exp(x_t(i))/(1._sp + exp(x_t(i)))

            else

                x(i) = exp(x_t(i))

            end if

        end do

    end subroutine inv_transformation_sbs

    subroutine optimize_lbfgsb(setup, mesh, input_data, parameters, states, output)

        !% Notes
        !% -----
        !% L-BFGS-B optimization subroutine.

        implicit none

        type(SetupDT), intent(inout) :: setup
        type(MeshDT), intent(inout) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        integer :: n, m, iprint
        integer, dimension(:), allocatable :: nbd, iwa
        real(dp) :: factr, pgtol, f
        real(dp), dimension(:), allocatable :: x, l, u, g, wa
        character(lchar) :: task, csave
        logical :: lsave(4)
        integer :: isave(44)
        real(dp) :: dsave(29)

        type(ParametersDT) :: parameters_b, parameters_bgd, parameters_bgd_b
        type(StatesDT) :: states_b, states_bgd, states_bgd_b
        type(OutputDT) :: output_b
        real(sp) :: cost, cost_b

        ! =========================================================================================================== !
        !   Initialize L-BFGS-B args (verbose, size, tol, bounds)
        ! =========================================================================================================== !

        iprint = -1

        n = mesh%nac*(count(setup%optimize%optim_parameters .gt. 0) + &
        & count(setup%optimize%optim_states .gt. 0))
        m = 10
        factr = 1.e6_dp
        pgtol = 1.e-12_dp

        allocate (nbd(n), x(n), l(n), u(n), g(n))
        allocate (iwa(3*n))
        allocate (wa(2*m*n + 5*n + 11*m*m + 8*m))

        ! =========================================================================================================== !
        !   Initialize forward_b args
        ! =========================================================================================================== !

        call ParametersDT_initialise(parameters_b, mesh)
        call ParametersDT_initialise(parameters_bgd_b, mesh)

        call StatesDT_initialise(states_b, mesh)
        call StatesDT_initialise(states_bgd_b, mesh)

        call OutputDT_initialise(output_b, setup, mesh)

        ! =========================================================================================================== !
        !   Initialize control (normalize and var to control)
        ! =========================================================================================================== !

        call normalize_parameters(setup, mesh, parameters)
        call normalize_states(setup, mesh, states)

        nbd = 2
        l = 0._dp
        u = 1._dp

        ! Background is normalize
        parameters_bgd = parameters
        states_bgd = states

        call var_to_control_lbfgsb(n, setup, mesh, parameters, states, x)

        ! Trigger the denormalization subroutine in forward
        setup%optimize%normalize_forward = .true.

        ! =========================================================================================================== !
        !   Start minimization
        ! =========================================================================================================== !

        task = 'START'
        do while ((task(1:2) .eq. 'FG' .or. task .eq. 'NEW_X' .or. &
                & task .eq. 'START'))

            call setulb(n, &   ! dimension of the problem
                        m, &   ! number of corrections of limited memory (approx. Hessian)
                        x, &   ! control
                        l, &   ! lower bound on control
                        u, &   ! upper bound on control
                        nbd, &   ! type of bounds
                        f, &   ! value of the (cost) function at x
                        g, &   ! value of the (cost) gradient at x
                        factr, &   ! tolerance iteration
                        pgtol, &   ! tolerance on projected gradient
                        wa, &   ! working array
                        iwa, &   ! working array
                        task, &   ! working string indicating current job in/out
                        iprint, &   ! verbose lbfgsb
                        csave, &   ! working array
                        lsave, &   ! working array
                        isave, &   ! working array
                        dsave)       ! working array

            call control_to_var_lbfgsb(n, setup, mesh, parameters, states, x)

            if (task(1:2) .eq. 'FG') then

                ! =================================================================================================== !
                !   Call forward_b (denormalization done in forward_b)
                ! =================================================================================================== !

                cost_b = 1._sp
                cost = 0._sp

                call forward_b(setup, mesh, input_data, parameters, &
                & parameters_b, parameters_bgd, parameters_bgd_b, states, &
                & states_b, states_bgd, states_bgd_b, output, output_b, &
                & cost, cost_b)

                call normalize_parameters(setup, mesh, parameters)
                call normalize_states(setup, mesh, states)

                f = real(cost, kind(f))

                call var_to_control_lbfgsb(n, setup, mesh, parameters_b, states_b, g)

                if (task(4:8) .eq. 'START') then

                    if (setup%optimize%verbose) then
                        write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", dsave(13)
                    end if

                end if

            end if

            if (task(1:5) .eq. 'NEW_X') then

                if (setup%optimize%verbose) then
                    write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)
                end if

                if (isave(30) .ge. setup%optimize%maxiter) then

                    task = 'STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT'

                end if

                if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) then

                    task = 'STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL'

                end if

            end if

        end do

        if (setup%optimize%verbose) then
            write (*, '(4x,a)') task
        end if

        ! =========================================================================================================== !
        !   End minimization
        ! =========================================================================================================== !

        call forward(setup, mesh, input_data, parameters, &
        & parameters_bgd, states, states_bgd, output, cost)

        ! Remove the denormalization subroutine in forward
        setup%optimize%normalize_forward = .false.

    end subroutine optimize_lbfgsb

    subroutine var_to_control_lbfgsb(n, setup, mesh, parameters, states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(in) :: states
        real(dp), dimension(n), intent(inout) :: x

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        integer, dimension(GNP + GNS) :: optim
        integer :: i, j, col, row

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        j = 0

        do i = 1, (GNP + GNS)

            if (optim(i) .gt. 0) then

                do col = 1, mesh%ncol

                    do row = 1, mesh%nrow

                        if (mesh%active_cell(row, col) .eq. 1) then

                            j = j + 1

                            x(j) = real(matrix(row, col, i), kind(x))

                        end if

                    end do

                end do

            end if

        end do

    end subroutine var_to_control_lbfgsb

    subroutine control_to_var_lbfgsb(n, setup, mesh, parameters, states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        real(dp), dimension(n), intent(in) :: x

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        integer, dimension(GNP + GNS) :: optim
        integer :: i, j, col, row

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        j = 0

        do i = 1, (GNP + GNS)

            if (optim(i) .gt. 0) then

                do col = 1, mesh%ncol

                    do row = 1, mesh%nrow

                        if (mesh%active_cell(row, col) .eq. 1) then

                            j = j + 1

                            matrix(row, col, i) = real(x(j), kind(matrix))

                        end if

                    end do

                end do

            end if

        end do

        call set_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call set_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

    end subroutine control_to_var_lbfgsb

    subroutine optimize_hyper_lbfgsb(setup, mesh, input_data, parameters, states, output)

        !% Notes
        !% -----
        !% L-BFGS-B hyper optimization subroutine.

        implicit none

        type(SetupDT), intent(inout) :: setup
        type(MeshDT), intent(inout) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        integer :: n, m, iprint
        integer, dimension(:), allocatable :: nbd, iwa
        real(dp) :: factr, pgtol, f
        real(dp), dimension(:), allocatable :: x, l, u, g, wa
        character(lchar) :: task, csave
        logical :: lsave(4)
        integer :: isave(44)
        real(dp) :: dsave(29)

        type(Hyper_ParametersDT) :: hyper_parameters, &
        & hyper_parameters_b, hyper_parameters_bgd
        type(Hyper_StatesDT) :: hyper_states, &
        & hyper_states_b, hyper_states_bgd
        type(ParametersDT) :: parameters_b
        type(StatesDT) :: states_b
        type(OutputDT) :: output_b
        real(sp) :: cost, cost_b
        real(sp), dimension(setup%nd) :: min_descriptor, &
        & max_descriptor

        ! =========================================================================================================== !
        !   Initialize L-BFGS-B args (verbose, size, tol, bounds)
        ! =========================================================================================================== !

        iprint = -1

        n = (count(setup%optimize%optim_parameters .gt. 0) &
        & + count(setup%optimize%optim_states .gt. 0))*setup%optimize%nhyper
        m = 10
        factr = 1.e6_dp
        pgtol = 1.e-12_dp

        allocate (nbd(n), x(n), l(n), u(n), g(n))
        allocate (iwa(3*n))
        allocate (wa(2*m*n + 5*n + 11*m*m + 8*m))

        ! =========================================================================================================== !
        !   Initialize hyper_forward_b args (normalize descriptors)
        ! =========================================================================================================== !

        call normalize_descriptor_hyper_lbfgsb(setup, input_data, min_descriptor, max_descriptor)

        call ParametersDT_initialise(parameters_b, mesh)
        call Hyper_ParametersDT_initialise(hyper_parameters, setup)
        call Hyper_ParametersDT_initialise(hyper_parameters_b, setup)

        call StatesDT_initialise(states_b, mesh)
        call Hyper_StatesDT_initialise(hyper_states, setup)
        call Hyper_StatesDT_initialise(hyper_states_b, setup)

        call OutputDT_initialise(output_b, setup, mesh)

        ! =========================================================================================================== !
        !   Initialize control (problem initialise and var to control)
        ! =========================================================================================================== !

        call problem_initialise_hyper_lbfgsb(n, setup, mesh, parameters, states, &
        & hyper_parameters, hyper_states, nbd, l, u)

        hyper_parameters_bgd = hyper_parameters
        hyper_states_bgd = hyper_states

        call var_to_control_hyper_lbfgsb(n, setup, hyper_parameters, hyper_states, x)

        ! =========================================================================================================== !
        !   Start minimization
        ! =========================================================================================================== !

        task = 'START'
        do while ((task(1:2) .eq. 'FG' .or. task .eq. 'NEW_X' .or. &
                & task .eq. 'START'))

            call setulb(n, &   ! dimension of the problem
                        m, &   ! number of corrections of limited memory (approx. Hessian)
                        x, &   ! control
                        l, &   ! lower bound on control
                        u, &   ! upper bound on control
                        nbd, &   ! type of bounds
                        f, &   ! value of the (cost) function at x
                        g, &   ! value of the (cost) gradient at x
                        factr, &   ! tolerance iteration
                        pgtol, &   ! tolerance on projected gradient
                        wa, &   ! working array
                        iwa, &   ! working array
                        task, &   ! working string indicating current job in/out
                        iprint, &   ! verbose lbfgsb
                        csave, &   ! working array
                        lsave, &   ! working array
                        isave, &   ! working array
                        dsave)       ! working array

            call control_to_var_hyper_lbfgsb(n, setup, hyper_parameters, hyper_states, x)

            if (task(1:2) .eq. 'FG') then

                ! =================================================================================================== !
                !   Call hyper_forward_b
                ! =================================================================================================== !

                cost_b = 1._sp
                cost = 0._sp

                call hyper_forward_b(setup, mesh, input_data, parameters, parameters_b, hyper_parameters, &
                & hyper_parameters_b, hyper_parameters_bgd, states, states_b, hyper_states, &
                & hyper_states_b, hyper_states_bgd, output, output_b, cost, cost_b)

                f = real(cost, kind(f))

                call var_to_control_hyper_lbfgsb(n, setup, hyper_parameters_b, hyper_states_b, g)

                if (task(4:8) .eq. 'START') then

                    if (setup%optimize%verbose) then
                        write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", dsave(13)
                    end if

                end if

            end if

            if (task(1:5) .eq. 'NEW_X') then

                if (setup%optimize%verbose) then
                    write (*, '(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)
                end if

                if (isave(30) .ge. setup%optimize%maxiter) then

                    task = 'STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT'

                end if

                if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) then

                    task = 'STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL'

                end if

            end if

        end do

        if (setup%optimize%verbose) then
            write (*, '(4x,a)') task
        end if

        ! =========================================================================================================== !
        !   End minimization (hyper_var to var and denormalize descriptors)
        ! =========================================================================================================== !

        call hyper_forward(setup, mesh, input_data, parameters, hyper_parameters, &
        & hyper_parameters_bgd, states, hyper_states, hyper_states_bgd, output, cost)

        call hyper_parameters_to_parameters(hyper_parameters, parameters, setup, mesh, input_data)
        call hyper_states_to_states(hyper_states, states, setup, mesh, input_data)

        call denormalize_descriptor_hyper_lbfgsb(setup, input_data, min_descriptor, max_descriptor)

    end subroutine optimize_hyper_lbfgsb

    subroutine normalize_descriptor_hyper_lbfgsb(setup, input_data, min_descriptor, max_descriptor)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Input_DataDT), intent(inout) :: input_data
        real(sp), dimension(setup%nd) :: min_descriptor, max_descriptor

        integer :: i

        do i = 1, setup%nd

            min_descriptor(i) = minval(input_data%descriptor(:, :, i))
            max_descriptor(i) = maxval(input_data%descriptor(:, :, i))

            input_data%descriptor(:, :, i) = &
            & (input_data%descriptor(:, :, i) - min_descriptor(i))/(max_descriptor(i) - min_descriptor(i))

        end do

    end subroutine normalize_descriptor_hyper_lbfgsb

    subroutine denormalize_descriptor_hyper_lbfgsb(setup, input_data, min_descriptor, max_descriptor)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Input_DataDT), intent(inout) :: input_data
        real(sp), dimension(setup%nd) :: min_descriptor, max_descriptor

        integer :: i

        do i = 1, setup%nd

            input_data%descriptor(:, :, i) = &
            & input_data%descriptor(:, :, i)*(max_descriptor(i) - min_descriptor(i)) + min_descriptor(i)

        end do

    end subroutine denormalize_descriptor_hyper_lbfgsb

    subroutine problem_initialise_hyper_lbfgsb(n, setup, mesh, parameters, states, hyper_parameters, hyper_states, nbd, l, u)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(in) :: states
        type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
        type(Hyper_StatesDT), intent(inout) :: hyper_states
        integer, dimension(n), intent(inout) :: nbd
        real(dp), dimension(n), intent(inout) :: l, u

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP + GNS) :: matrix
        real(sp), dimension(setup%optimize%nhyper, 1, GNP + GNS) :: hyper_matrix
        integer, dimension(GNP + GNS) :: optim
        real(sp), dimension(GNP + GNS) :: lb, ub
        integer, dimension(2) :: ind_ac
        integer :: i, j, k

        ind_ac = maxloc(mesh%active_cell)

        call get_parameters(mesh, parameters, matrix(:, :, 1:GNP))
        call get_states(mesh, states, matrix(:, :, GNP + 1:GNP + GNS))

        call set_hyper_parameters(setup, hyper_parameters, 0._sp)
        call set_hyper_states(setup, hyper_states, 0._sp)

        call get_hyper_parameters(setup, hyper_parameters, hyper_matrix(:, :, 1:GNP))
        call get_hyper_states(setup, hyper_states, hyper_matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        lb(1:GNP) = setup%optimize%lb_parameters
        lb(GNP + 1:GNP + GNS) = setup%optimize%lb_states

        ub(1:GNP) = setup%optimize%ub_parameters
        ub(GNP + 1:GNP + GNS) = setup%optimize%ub_states

        ! inverse sigmoid lambda = 1
        hyper_matrix(1, 1, :) = &
            log(max(1e-8_sp, (matrix(ind_ac(1), ind_ac(2), :) - lb))/ &
                max(1e-8_sp, (ub - matrix(ind_ac(1), ind_ac(2), :))) &
                )

        nbd = 0
        l = 0._dp
        u = 0._dp
        k = 0

        do i = 1, (GNP + GNS)

            if (optim(i) .gt. 0) then

                select case (trim(setup%optimize%mapping))

                case ("hyper-linear")

                    hyper_matrix(2:setup%optimize%nhyper, 1, i) = 0._sp

                case ("hyper-polynomial")

                    do j = 1, setup%optimize%nhyper - 1

                        if (mod(j + 1, 2) .eq. 0) then

                            hyper_matrix(j + 1, 1, i) = 0._sp

                            nbd(k + (j + 1)) = 0

                        else

                            hyper_matrix(j + 1, 1, i) = 1._sp

                            nbd(k + (j + 1)) = 2
                            l(k + (j + 1)) = 0.5_dp
                            u(k + (j + 1)) = 2._dp

                        end if

                    end do

                end select

                k = k + setup%optimize%nhyper

            end if

        end do

        call set_hyper_parameters(setup, hyper_parameters, hyper_matrix(:, :, 1:GNP))
        call set_hyper_states(setup, hyper_states, hyper_matrix(:, :, GNP + 1:GNP + GNS))

    end subroutine problem_initialise_hyper_lbfgsb

    subroutine var_to_control_hyper_lbfgsb(n, setup, hyper_parameters, hyper_states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(in) :: hyper_parameters
        type(Hyper_StatesDT), intent(in) :: hyper_states
        real(dp), dimension(n), intent(inout) :: x

        real(sp), dimension(setup%optimize%nhyper, 1, GNP + GNS) :: matrix
        integer, dimension(GNP + GNS) :: optim
        integer :: i, j, k

        call get_hyper_parameters(setup, hyper_parameters, matrix(:, :, 1:GNP))
        call get_hyper_states(setup, hyper_states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        k = 0

        do i = 1, (GNP + GNS)

            if (optim(i) .gt. 0) then

                do j = 1, setup%optimize%nhyper

                    k = k + 1
                    x(k) = real(matrix(j, 1, i), kind(x))

                end do

            end if

        end do

    end subroutine var_to_control_hyper_lbfgsb

    subroutine control_to_var_hyper_lbfgsb(n, setup, hyper_parameters, hyper_states, x)

        implicit none

        integer, intent(in) :: n
        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
        type(Hyper_StatesDT), intent(inout) :: hyper_states
        real(dp), dimension(n), intent(in) :: x

        real(sp), dimension(setup%optimize%nhyper, 1, GNP + GNS) :: matrix
        integer, dimension(GNP + GNS) :: optim
        integer :: i, j, k

        call get_hyper_parameters(setup, hyper_parameters, matrix(:, :, 1:GNP))
        call get_hyper_states(setup, hyper_states, matrix(:, :, GNP + 1:GNP + GNS))

        optim(1:GNP) = setup%optimize%optim_parameters
        optim(GNP + 1:GNP + GNS) = setup%optimize%optim_states

        k = 0

        do i = 1, (GNP + GNS)

            if (optim(i) .gt. 0) then

                do j = 1, setup%optimize%nhyper

                    k = k + 1
                    matrix(j, 1, i) = real(x(k), kind(matrix))

                end do

            end if

        end do

        call set_hyper_parameters(setup, hyper_parameters, matrix(:, :, 1:GNP))
        call set_hyper_states(setup, hyper_states, matrix(:, :, GNP + 1:GNP + GNS))

    end subroutine control_to_var_hyper_lbfgsb

end module mw_optimize
