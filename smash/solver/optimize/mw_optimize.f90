!%      This module `mw_optimize` encapsulates all SMASH optimize.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1] optimize_sbs
!%      [2] transformation
!%      [3] inv_transformation
!%      [4] optimize_lbfgsb
!%      [5] optimize_matrix_to_vector
!%      [6] optimize_vector_to_matrix
!%      [7] normalize_matrix
!%      [8] unnormalize_matrix
!%      [9] optimize_message

module mw_optimize
    
    use md_common, only: sp, dp, lchar, np, ns
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
    & hyper_parameters_to_parameters
    use mwd_states_manipulation, only: get_states, set_states, &
    & get_hyper_states, set_hyper_states, &
    & hyper_states_to_states
    
    implicit none
    
    public :: optimize_sbs, optimize_lbfgsb, hyper_optimize_lbfgsb
    
    private :: transformation, inv_transformation, & 
    & normalize_descriptor, denormalize_descriptor, &
    & hyper_problem_initialise, parameters_states_to_x, &
    & x_to_parameters_states, hyper_parameters_states_to_x, &
    & x_to_hyper_parameters_states
    
    contains
        
        subroutine optimize_sbs(setup, mesh, input_data, parameters, states, output)
        
            !% Notes
            !% -----
            !%
            !% Step By Step optimization subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it returns the result of a step by step optimization.
            !% argmin(theta) = J(theta)
            !%
            !% Calling forward from forward/forward.f90  Y  = M (k)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd
            type(StatesDT) :: states_bgd
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, ns) :: states_matrix
            integer, dimension(2) :: ind_ac
            integer :: nps, nops, iter, nfg, ia, iaa, iam, jf, jfa, jfaa, j, ps, p, s
            real(sp), dimension(np+ns) :: x, y, x_tf, y_tf, z_tf, lb, lb_tf, ub, ub_tf, sdx
            integer, dimension(np+ns) :: optim_ps
            real(sp) :: gx, ga, clg, ddx, dxn, f, cost
            
            !% =========================================================================================================== %!
            !%   Initialisation
            !% =========================================================================================================== %!
        
            ind_ac = maxloc(mesh%active_cell)
            
            parameters_bgd = parameters
            states_bgd = states
            
            call forward(setup, mesh, input_data, parameters, &
            & parameters_bgd, states, states_bgd, output, cost)
            
            call get_parameters(parameters, parameters_matrix)
            call get_states(states, states_matrix)
            
            nps = np + ns
            
            x(1:np) = parameters_matrix(ind_ac(1), ind_ac(2), :)
            x(np+1:nps) = states_matrix(ind_ac(1), ind_ac(2), :)
            
            lb(1:np) = setup%lb_parameters
            lb(np+1:nps) = setup%lb_states

            ub(1:np) = setup%ub_parameters
            ub(np+1:nps) = setup%ub_states
            
            optim_ps(1:np) = setup%optim_parameters
            optim_ps(np+1:nps) = setup%optim_states
            
            call transformation(x, lb, ub, x_tf)
            call transformation(lb, lb, ub, lb_tf)
            call transformation(ub, lb, ub, ub_tf)
            
            nops = count(optim_ps .eq. 1)
            gx = cost
            ga = gx
            clg = 0.7_sp ** (1._sp / real(nops))
            z_tf = x_tf
            sdx = 0._sp
            ddx = 0.64_sp
            dxn = ddx
            ia = 0
            iaa = 0
            iam = 0
            jfaa = 0
            nfg = 1
            
            write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
            & "At iterate", 0, "nfg = ", nfg, "J =", gx, "ddx =", ddx
            
            do iter=1, setup%maxiter * nops
                
                !% ======================================================================================================= %!
                !%   Optimize
                !% ======================================================================================================= %!

                if (dxn .gt. ddx) dxn = ddx
                if (ddx .gt. 2._sp) ddx = dxn
                
                do ps=1, nps
                
                    if (optim_ps(ps) .eq. 1) then
                    
                        y_tf = x_tf
                        
                        do 7 j=1, 2
                        
                            jf = 2 * j - 3
                            if (ps .eq. iaa .and. jf .eq. -jfaa) goto 7
                            if (x_tf(ps) .le. lb_tf(ps) .and. jf .lt. 0) goto 7
                            if (x_tf(ps) .ge. ub_tf(ps) .and. jf .gt. 0) goto 7
                            
                            y_tf(ps) = x_tf(ps) + jf * ddx
                            if (y_tf(ps) .lt. lb_tf(ps)) y_tf(ps) = lb_tf(ps)
                            if (y_tf(ps) .gt. ub_tf(ps)) y_tf(ps) = ub_tf(ps)
                            
                            call inv_transformation(y_tf, lb, ub, y)
                            
                            do p=1, np
                            
                                where (mesh%active_cell .eq. 1)
                                
                                    parameters_matrix(:,:,p) = y(p)
                                
                                end where
                                
                            end do
                            
                            do s=1, ns
                            
                                 where (mesh%active_cell .eq. 1)
                                
                                    states_matrix(:,:,s) = y(np+s)
                                
                                end where
                            
                            
                            end do
                            
                            call set_parameters(parameters, parameters_matrix)
                            call set_states(states, states_matrix)

                            call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                            & states, states_bgd, output, cost)

                            f = cost
                            nfg = nfg + 1
                            
                            if (f .lt. gx) then
                            
                                z_tf = y_tf
                                gx = f
                                ia = ps
                                jfa = jf
                            
                            end if
                            
                        7 continue

                    end if
                
                end do
                
                iaa = ia
                jfaa = jfa
                
                if (ia .ne. 0) then
                
                    x_tf = z_tf
                    call inv_transformation(x_tf, lb, ub, x)
                    
                    do p=1, np
                            
                        where (mesh%active_cell .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                            
                        where (mesh%active_cell .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call set_parameters(parameters, parameters_matrix)
                    call set_states(states, states_matrix)
                    
                    sdx = clg * sdx
                    sdx(ia) = (1._sp - clg) * real(jfa) * ddx + clg * sdx(ia)
                    
                    iam = iam + 1
                    
                    if (iam .gt. 2 * nops) then
                        
                        ddx = ddx * 2._sp
                        iam = 0
                    
                    end if
                    
                    if (gx .lt. ga - 2) then
                    
                        ga = gx
                    
                    end if
                    
                else
                
                    ddx = ddx / 2._sp
                    iam = 0
                
                end if
                
                if (iter .gt. 4 * nops) then
                
                    do ps=1, nps
                    
                        if (optim_ps(ps) .eq. 1) then
                        
                            y_tf(ps) = x_tf(ps) + sdx(ps)
                            if (y_tf(ps) .lt. lb_tf(ps)) y_tf(ps) = lb_tf(ps)
                            if (y_tf(ps) .gt. ub_tf(ps)) y_tf(ps) = ub_tf(ps)
                        
                        end if
                    
                    end do
                    
                    call inv_transformation(y_tf, lb, ub, y)
                    
                    do p=1, np
                    
                        where (mesh%active_cell .eq. 1)
                        
                            parameters_matrix(:,:,p) = y(p)
                        
                        end where
                    
                    end do
                    
                    do s=1, ns
                    
                        where (mesh%active_cell .eq. 1)
                        
                            states_matrix(:,:,s) = y(np+s)
                        
                        end where
                    
                    end do
                    
                    call set_parameters(parameters, parameters_matrix)
                    call set_states(states, states_matrix)
                    
                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    f = cost
                    
                    nfg = nfg + 1
                    
                    if (f .lt. gx) then
                    
                        gx = f
                        jfaa = 0
                        x_tf = y_tf
                        
                        call inv_transformation(x_tf, lb, ub, x)
                        
                        do p=1, np
                        
                            where (mesh%active_cell .eq. 1) 
                            
                                parameters_matrix(:,:,p) = x(p)
                            
                            end where
                        
                        end do
                        
                        do s=1, ns
                        
                            where (mesh%active_cell .eq. 1) 
                            
                                states_matrix(:,:,s) = x(np+s)
                            
                            end where
                        
                        end do
                        
                        call set_parameters(parameters, parameters_matrix)
                        call set_states(states, states_matrix)
                        
                        if (gx .lt. ga - 2) then
                        
                            ga = gx
                            
                        end if
                    
                    end if
                
                end if
                
                ia = 0
                
                !% ======================================================================================================= %!
                !%   Iterate writting
                !% ======================================================================================================= %!
            
                if (mod(iter, nops) .eq. 0) then
                    
                    write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f5.2)') &
                    & "At iterate", (iter / nops), "nfg = ", nfg, "J =", gx, "ddx =", ddx
                
                end if
                
                !% ======================================================================================================= %!
                !%   Convergence DDX < 0.01
                !% ======================================================================================================= %!
            
                if (ddx .lt. 0.01_sp) then
                
                    write(*,'(4x,a)') "CONVERGENCE: DDX < 0.01"

                    do p=1, np
                        
                        where (mesh%active_cell .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                        
                        where (mesh%active_cell .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call set_parameters(parameters, parameters_matrix)
                    call set_states(states, states_matrix)

                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    exit
                
                end if
                
                !% ======================================================================================================= %!
                !%   Maximum Number of Iteration
                !% ======================================================================================================= %!
                
                if (iter .eq. setup%maxiter * nops) then
                
                    write(*,'(4x,a)') "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"
            
                    do p=1, np
                        
                        where (mesh%active_cell .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                        
                        where (mesh%active_cell .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call set_parameters(parameters, parameters_matrix)
                    call set_states(states, states_matrix)

                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    exit
                
                end if
                
            end do
            
        end subroutine optimize_sbs


!%      TODO comment
        subroutine transformation(x, lb, ub, x_tf)
        
            implicit none
            
            real(sp), dimension(np+ns), intent(in) :: x, lb, ub
            real(sp), dimension(np+ns), intent(inout) :: x_tf
            
            integer :: i
            
            do i=1, np+ns
            
                if (lb(i) .lt. 0._sp) then
                    
                    x_tf(i) = sinh(x(i))
                    
                else if (lb(i) .ge. 0._sp .and. ub(i) .le. 1._sp) then
                    
                    x_tf(i) = log(x(i) / (1._sp - x(i)))
                    
                else
                
                    x_tf(i) = log(x(i))
                
                end if
 
            end do

        end subroutine transformation
        

!%      TODO comment
        subroutine inv_transformation(x_tf, lb, ub, x)
        
            implicit none
            
            real(sp), dimension(np+ns), intent(in) :: x_tf, lb, ub
            real(sp), dimension(np+ns), intent(inout) :: x
            
            integer :: i
            
            do i=1, np+ns
            
                if (lb(i) .lt. 0._sp) then
                    
                    x(i) = asinh(x_tf(i))
                    
                else if (lb(i) .ge. 0._sp .and. ub(i) .le. 1._sp) then
                    
                    x(i) = exp(x_tf(i)) / (1._sp + exp(x_tf(i)))
                    
                else
                
                    x(i) = exp(x_tf(i))
                
                end if
 
            end do
            
        end subroutine inv_transformation
        
        
        subroutine optimize_lbfgsb(setup, mesh, input_data, parameters, states, output)
        
            !% Notes
            !% -----
            !%
            !% L-BFGS-B optimization subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, StatesDT, OutputDT,
            !% it returns the result of a l-bfgs-b optimization.
            !% argmin(theta) = J(theta)
            !%
            !% Calling forward_b from forward/forward_b.f90  dk* = (dM/dk)* (k) . dY*
            !% Calling setulb from optimize/lbfgsb.f
            
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
            
            type(ParametersDT) :: parameters_bgd, parameters_b
            type(StatesDT) :: states_bgd, states_b
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            
            iprint = -1

            n = mesh%nac * (count(setup%optim_parameters .gt. 0) + &
            & count(setup%optim_states .gt. 0))
            m = 10
            factr = 1.e7_dp
            pgtol = 1.e-12_dp
            
            allocate(nbd(n), x(n), l(n), u(n), g(n))
            allocate(iwa(3 * n))
            allocate(wa(2 * m * n + 5 * n + 11 * m * m + 8 * m))
            
            nbd = 2
            l = 0._dp
            u = 1._dp
            
            call ParametersDT_initialise(parameters_b, mesh)
            
            call StatesDT_initialise(states_b, mesh)
            
            call OutputDT_initialise(output_b, setup, mesh)
            
            parameters_bgd = parameters
            states_bgd = states
            
            call parameters_states_to_x(parameters, states, x, setup, mesh, .true.)

            task = 'START'
            do while((task(1:2) .eq. 'FG' .or. task .eq. 'NEW_X' .or. &
                    & task .eq. 'START'))
                    
                call setulb(n       ,&   ! dimension of the problem
                            m       ,&   ! number of corrections of limited memory (approx. Hessian) 
                            x       ,&   ! control
                            l       ,&   ! lower bound on control
                            u       ,&   ! upper bound on control
                            nbd     ,&   ! type of bounds
                            f       ,&   ! value of the (cost) function at x
                            g       ,&   ! value of the (cost) gradient at x
                            factr   ,&   ! tolerance iteration 
                            pgtol   ,&   ! tolerance on projected gradient 
                            wa      ,&   ! working array
                            iwa     ,&   ! working array
                            task    ,&   ! working string indicating current job in/out
                            iprint  ,&   ! verbose lbfgsb
                            csave   ,&   ! working array
                            lsave   ,&   ! working array
                            isave   ,&   ! working array
                            dsave)       ! working array
                            
                call x_to_parameters_states(x, parameters, states, setup, mesh, .true.)
                
                if (task(1:2) .eq. 'FG') then
                
                    cost_b = 1._sp
                    cost = 0._sp
                    
                    call forward_b(setup, mesh, input_data, parameters, &
                    & parameters_b, parameters_bgd, states, states_b, states_bgd, &
                    & output, output_b, cost, cost_b)
        
                    f = real(cost, kind(f))
                    
                    call parameters_states_to_x(parameters_b, states_b, g, setup, mesh, .false.)
                    
                    if (task(4:8) .eq. 'START') then
                    
                        write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", dsave(13)
                    
                    end if
 
                end if
                
                if (task(1:5) .eq. 'NEW_X') then
                    
                    write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)
                
                    if (isave(30) .ge. setup%maxiter) then
                        
                        task='STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT'
                        
                    end if
                    
                    if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) then
                       
                        task='STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL'
                       
                    end if
                       
                end if
                    
            end do
            
        write(*, '(4x,a)') task
        
        call forward(setup, mesh, input_data, parameters, &
        & parameters_bgd, states, states_bgd, output, cost)

        end subroutine optimize_lbfgsb
        
        
        subroutine parameters_states_to_x(parameters, states, x, setup, mesh, normalize)
        
            implicit none
            
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(in) :: states
            real(dp), dimension(:), intent(inout) :: x
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            logical, intent(in) :: normalize
            
            real(sp), dimension(mesh%nrow, mesh%ncol, np+ns) :: matrix
            integer, dimension(np+ns) :: optim
            real(sp), dimension(np+ns) :: lb, ub
            integer :: i, j, col, row
            
            call get_parameters(parameters, matrix(:,:,1:np))
            call get_states(states, matrix(:,:,np+1:np+ns))
            
            optim(1:np) = setup%optim_parameters 
            optim(np+1:np+ns) = setup%optim_states
            
            lb(1:np) = setup%lb_parameters
            lb(np+1:np+ns) = setup%lb_states
            
            ub(1:np) = setup%ub_parameters
            ub(np+1:np+ns) = setup%ub_states

            j = 0
            
            do i=1, (np + ns)
            
                if (optim(i) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%active_cell(row, col) .eq. 1) then
                                
                                j = j + 1
                                
                                if (normalize) then
                                
                                    x(j) = real(&
                                    & (matrix(row, col, i) - lb(i)) / &
                                    & (ub(i) - lb(i)), kind(x))
                                    
                                else
                                
                                    x(j) = real(matrix(row, col, i), kind(x))
                                    
                                end if
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if

            end do

        end subroutine parameters_states_to_x
        
        
        subroutine x_to_parameters_states(x, parameters, states, setup, mesh, unnormalize)
        
            implicit none
            
            real(dp), dimension(:), intent(in) :: x
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            logical, intent(in) :: unnormalize
            
            real(sp), dimension(mesh%nrow, mesh%ncol, np+ns) :: matrix
            integer, dimension(np+ns) :: optim
            real(sp), dimension(np+ns) :: lb, ub
            integer :: i, j, col, row
            
            call get_parameters(parameters, matrix(:,:,1:np))
            call get_states(states, matrix(:,:,np+1:np+ns))
            
            optim(1:np) = setup%optim_parameters 
            optim(np+1:np+ns) = setup%optim_states
            
            lb(1:np) = setup%lb_parameters
            lb(np+1:np+ns) = setup%lb_states
            
            ub(1:np) = setup%ub_parameters
            ub(np+1:np+ns) = setup%ub_states
            
            j = 0
            
            do i=1, (np + ns)
            
                if (optim(i) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%active_cell(row, col) .eq. 1) then
                                
                                j = j + 1
                                
                                if (unnormalize) then
                                
                                    matrix(row, col, i) = real(x(j) * &
                                    & (ub(i) - lb(i)) + lb(i), & 
                                    & kind(matrix))
                                    
                                else
                                    
                                    matrix(row, col, i) = real(x(j), kind(matrix))
                                
                                end if
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if

            end do
            
            call set_parameters(parameters, matrix(:,:,1:np))
            call set_states(states, matrix(:,:,np+1:np+ns))

        end subroutine x_to_parameters_states


        subroutine hyper_optimize_lbfgsb(setup, mesh, input_data, parameters, states, output)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            integer :: n, c, ndc, m, iprint
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
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            real(sp), dimension(setup%nd) :: min_descriptor, &
            & max_descriptor
            
            iprint = -1
            
            c = count(setup%optim_parameters .gt. 0) + count(setup%optim_states .gt. 0)
            
            select case(trim(setup%mapping))
            
            case("hyper-linear")
            
                ndc = (1 + setup%nd)
                
            case("hyper-polynomial")
                
                ndc = (1 + 2 * setup%nd)
                
            end select
            
            n = c * ndc
            m = 10
            factr = 1.e7_dp
            pgtol = 1.e-12_dp
            
            allocate(nbd(n), x(n), l(n), u(n), g(n))
            allocate(iwa(3 * n))
            allocate(wa(2 * m * n + 5 * n + 11 * m * m + 8 * m))
            
            call normalize_descriptor(setup, input_data, min_descriptor, max_descriptor)
            
            call Hyper_ParametersDT_initialise(hyper_parameters, setup)
            call Hyper_ParametersDT_initialise(hyper_parameters_b, setup)
            
            call Hyper_StatesDT_initialise(hyper_states, setup)
            call Hyper_StatesDT_initialise(hyper_states_b, setup)
            
            call OutputDT_initialise(output_b, setup, mesh)
            
            call hyper_problem_initialise(hyper_parameters, &
            & hyper_states, nbd, l, u, setup, mesh, parameters, states, ndc)
            
            hyper_parameters_bgd = hyper_parameters
            hyper_states_bgd = hyper_states
            
            call hyper_parameters_states_to_x(hyper_parameters, hyper_states, x, setup, ndc)
            
            task = 'START'
            do while((task(1:2) .eq. 'FG' .or. task .eq. 'NEW_X' .or. &
                    & task .eq. 'START'))
                
                call setulb(n       ,&   ! dimension of the problem
                            m       ,&   ! number of corrections of limited memory (approx. Hessian) 
                            x       ,&   ! control
                            l       ,&   ! lower bound on control
                            u       ,&   ! upper bound on control
                            nbd     ,&   ! type of bounds
                            f       ,&   ! value of the (cost) function at x
                            g       ,&   ! value of the (cost) gradient at x
                            factr   ,&   ! tolerance iteration 
                            pgtol   ,&   ! tolerance on projected gradient 
                            wa      ,&   ! working array
                            iwa     ,&   ! working array
                            task    ,&   ! working string indicating current job in/out
                            iprint  ,&   ! verbose lbfgsb
                            csave   ,&   ! working array
                            lsave   ,&   ! working array
                            isave   ,&   ! working array
                            dsave)       ! working array
                            
                call x_to_hyper_parameters_states(x, hyper_parameters, hyper_states, setup, ndc)
                
                if (task(1:2) .eq. 'FG') then
                
                    cost_b = 1._sp
                    cost = 0._sp
                    
                    call hyper_forward_b(setup, mesh, input_data, hyper_parameters, &
                    & hyper_parameters_b, hyper_parameters_bgd, hyper_states, hyper_states_b, hyper_states_bgd, &
                    & output, output_b, cost, cost_b)
        
                    f = real(cost, kind(f))
                    
                    call hyper_parameters_states_to_x(hyper_parameters_b, hyper_states_b, g, setup, ndc)
                    
                    if (task(4:8) .eq. 'START') then
                        
                        write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", 0, "nfg = ", 1, "J =", f, "|proj g| =", dsave(13)
                    
                    end if
 
                end if
                
                if (task(1:5) .eq. 'NEW_X') then
                    
                    write(*,'(4x,a,4x,i3,4x,a,i5,4x,a,f10.6,4x,a,f10.6)') &
                        & "At iterate", isave(30), "nfg = ", isave(34), "J =", f, "|proj g| =", dsave(13)
                
                    if (isave(30) .ge. setup%maxiter) then
                        
                        task='STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT'
                        
                    end if
                    
                    if (dsave(13) .le. 1.d-10*(1.0d0 + abs(f))) then
                       
                        task='STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL'
                       
                    end if
                       
                end if
                    
            end do
            
        write(*, '(4x,a)') task

        call hyper_forward(setup, mesh, input_data, hyper_parameters, &
        & hyper_parameters_bgd, hyper_states, hyper_states_bgd, output, cost)
        
        call hyper_parameters_to_parameters(hyper_parameters, parameters, setup, input_data)
        call hyper_states_to_states(hyper_states, states, setup, input_data)
        
        call denormalize_descriptor(setup, input_data, min_descriptor, max_descriptor)
        
        end subroutine hyper_optimize_lbfgsb
        
        
        subroutine normalize_descriptor(setup, input_data, min_descriptor, max_descriptor)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(Input_DataDT), intent(inout) :: input_data
            real(sp), dimension(setup%nd) :: min_descriptor, max_descriptor
            
            integer :: i
            
            do i=1, setup%nd
            
                min_descriptor(i) = minval(input_data%descriptor(:,:,i))
                max_descriptor(i) = maxval(input_data%descriptor(:,:,i))
                
                input_data%descriptor(:,:,i) = &
                & (input_data%descriptor(:,:,i) - min_descriptor(i)) / (max_descriptor(i) - min_descriptor(i))
            
            end do
        
        end subroutine normalize_descriptor
        
        
        subroutine denormalize_descriptor(setup, input_data, min_descriptor, max_descriptor)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(Input_DataDT), intent(inout) :: input_data
            real(sp), dimension(setup%nd) :: min_descriptor, max_descriptor
            
            integer :: i
            
            do i=1, setup%nd
            
                input_data%descriptor(:,:,i) = &
                & input_data%descriptor(:,:,i) * (max_descriptor(i) - min_descriptor(i)) + min_descriptor(i)
            
            end do
        
        end subroutine denormalize_descriptor
        

        subroutine hyper_problem_initialise(hyper_parameters, hyper_states, nbd, l, u, setup, mesh, parameters, states, ndc)
            
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            integer, dimension(:), intent(inout) :: nbd
            real(dp), dimension(:), intent(inout) :: l, u
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(in) :: states
            integer, intent(in) :: ndc
            
            real(sp), dimension(mesh%nrow, mesh%ncol, np+ns) :: matrix
            real(sp), dimension(ndc, 1, np+ns) :: hyper_matrix
            integer, dimension(np+ns) :: optim
            real(sp), dimension(np+ns) :: lb, ub
            integer, dimension(2) :: ind_ac
            integer :: i, j, k
            
            ind_ac = maxloc(mesh%active_cell)
            
            call get_parameters(parameters, matrix(:,:,1:np))
            call get_states(states, matrix(:,:,np+1:np+ns))
            
            call set_hyper_parameters(hyper_parameters, 0._sp)
            call set_hyper_states(hyper_states, 0._sp)
            
            call get_hyper_parameters(hyper_parameters, hyper_matrix(:,:,1:np))
            call get_hyper_states(hyper_states, hyper_matrix(:,:,np+1:np+ns))
            
            optim(1:np) = setup%optim_parameters 
            optim(np+1:np+ns) = setup%optim_states
            
            lb(1:np) = setup%lb_parameters
            lb(np+1:np+ns) = setup%lb_states
            
            ub(1:np) = setup%ub_parameters
            ub(np+1:np+ns) = setup%ub_states
            
            !% inverse sigmoid lambda = 1
            hyper_matrix(1, 1, :) = log((matrix(ind_ac(1), ind_ac(2), :) - lb) / (ub - matrix(ind_ac(1), ind_ac(2), :)))
            
            nbd = 0
            l = 0._dp
            u = 0._dp
            k = 0
            
            do i=1, (np + ns)
            
                if (optim(i) .gt. 0) then
                
                    select case(trim(setup%mapping))
                    
                    case("hyper-linear")
                    
                        hyper_matrix(2: ndc, 1, i) = 0._sp
                    
                    case("hyper-polynomial")
                    
                        do j=1, 2 * setup%nd
                        
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
                    
                    k = k + ndc
                
                end if

            end do
            
            call set_hyper_parameters(hyper_parameters, hyper_matrix(:,:,1:np))
            call set_hyper_states(hyper_states, hyper_matrix(:,:,np+1:np+ns))
        
        end subroutine hyper_problem_initialise


        subroutine hyper_parameters_states_to_x(hyper_parameters, hyper_states, x, setup, ndc)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters
            type(Hyper_StatesDT), intent(in) :: hyper_states
            real(dp), dimension(:), intent(inout) :: x
            type(SetupDT), intent(in) :: setup
            integer, intent(in) :: ndc
            
            real(sp), dimension(ndc, 1, np+ns) :: matrix
            integer, dimension(np+ns) :: optim
            integer :: i, j, k
            
            call get_hyper_parameters(hyper_parameters, matrix(:,:,1:np))
            call get_hyper_states(hyper_states, matrix(:,:,np+1:np+ns))
            
            optim(1:np) = setup%optim_parameters 
            optim(np+1:np+ns) = setup%optim_states
            
            k = 0
            
            do i=1, (np + ns)
            
                if (optim(i) .gt. 0) then
                
                    do j=1, ndc
                        
                        k = k + 1
                        x(k) = real(matrix(j, 1, i), kind(x))
                    
                    end do
                
                end if
        
            end do
        
        end subroutine hyper_parameters_states_to_x
        
        
        subroutine x_to_hyper_parameters_states(x, hyper_parameters, hyper_states, setup, ndc)
        
            implicit none
            
            real(dp), dimension(:), intent(in) :: x
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            type(SetupDT), intent(in) :: setup
            integer, intent(in) :: ndc
            
            real(sp), dimension(ndc, 1, np+ns) :: matrix
            integer, dimension(np+ns) :: optim
            integer :: i, j, k
            
            call get_hyper_parameters(hyper_parameters, matrix(:,:,1:np))
            call get_hyper_states(hyper_states, matrix(:,:,np+1:np+ns))
            
            optim(1:np) = setup%optim_parameters 
            optim(np+1:np+ns) = setup%optim_states
            
            k = 0
            
            do i=1, (np + ns)
            
                if (optim(i) .gt. 0) then
                
                    do j=1, ndc
                        
                        k = k + 1
                        matrix(j, 1, i) = real(x(k), kind(matrix))
                    
                    end do
                
                end if
        
            end do
            
            call set_hyper_parameters(hyper_parameters, matrix(:,:,1:np))
            call set_hyper_states(hyper_states, matrix(:,:,np+1:np+ns))
        
        end subroutine x_to_hyper_parameters_states
        
end module mw_optimize
