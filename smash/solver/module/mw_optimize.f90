!%      This module `mw_optimize` encapsulates all SMASH optimize.
!%      This module is wrapped.
!%
!%      contains
!%
!%      [1] optimize_sbs
!%      [2] optimize_lbfgsb
!%      [3] transformation
!%      [4] inv_transformation
!%      [5] normalize_matrix
!%      [6] unnormalize_matrix
!%      [7] optimize_message

module mw_optimize
    
    use mwd_common !% only: sp, dp, lchar, np, ns
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT, parameters_to_matrix, &
    ! & matrix_to_parameters, states_to_matrix, matrix_to_states
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    implicit none
    
    public :: optimize_sbs, optimize_lbfgsb
    
    private :: transformation, inv_transformation, &
    & optimize_matrix_to_vector, optimize_vector_to_matrix, &
    & normalize_matrix, unnormalize_matrix, optimize_message
    
    contains
        
                
        !% Calling forward from forward/forward.f90
        subroutine optimize_sbs(setup, mesh, input_data, parameters, states, output)
    
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            type(ParametersDT) :: parameters_bgd
            type(StatesDT) :: states_bgd, states_imd
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, ns) :: states_matrix
            integer, dimension(mesh%nrow, mesh%ncol) :: mask_ac
            integer, dimension(2) :: loc_ac
            integer :: nps, nops, iter, nfg, ia, iaa, iam, jf, jfa, jfaa, j, ps, p, s
            real(sp), dimension(np+ns) :: x, y, x_tf, y_tf, z_tf, lb, lb_tf, ub, ub_tf, sdx
            integer, dimension(np+ns) :: optim_ps
            real(sp) :: gx, ga, clg, ddx, dxn, f, cost
            
            call optimize_message(setup, mesh, 1)
            
            !% =========================================================================================================== %!
            !%   Initialisation
            !% =========================================================================================================== %!
            
            where (mesh%global_active_cell .eq. 1)
            
                mask_ac = 1
            
            else where
            
                mask_ac = 0
            
            end where
        
            loc_ac = maxloc(mask_ac)
            
            parameters_bgd = parameters
            states_bgd = states
            
            call forward(setup, mesh, input_data, parameters, &
            & parameters_bgd, states, states_bgd, output, cost)
            
            states = states_bgd
            
            call parameters_to_matrix(parameters, parameters_matrix)
            call states_to_matrix(states, states_matrix)
            
            nps = np + ns
            
            x(1:np) = parameters_matrix(loc_ac(1), loc_ac(2), :)
            x(np+1:nps) = states_matrix(loc_ac(1), loc_ac(2), :)
            
            lb(1:np) = setup%lb_parameters
            lb(np+1:nps) = setup%lb_states

            ub(1:np) = setup%ub_parameters
            ub(np+1:nps) = setup%ub_states
            
            optim_ps(1:np) = setup%optim_parameters
            optim_ps(np+1:nps) = setup%optim_states
            
            x_tf = transformation(x, lb, ub)
            lb_tf = transformation(lb, lb, ub)
            ub_tf = transformation(ub, lb, ub)
            
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
                            
                            y = inv_transformation(y_tf, lb, ub)
                            
                            do p=1, np
                            
                                where (mask_ac .eq. 1)
                                
                                    parameters_matrix(:,:,p) = y(p)
                                
                                end where
                                
                            end do
                            
                            do s=1, ns
                            
                                 where (mask_ac .eq. 1)
                                
                                    states_matrix(:,:,s) = y(np+s)
                                
                                end where
                            
                            
                            end do
                            
                            call matrix_to_parameters(parameters_matrix, parameters)
                            call matrix_to_states(states_matrix, states)
                            
                            states_imd = states
                            
                            call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                            & states, states_bgd, output, cost)
                            
                            states = states_imd
                            
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
                    x = inv_transformation(x_tf, lb, ub)
                    
                    do p=1, np
                            
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                            
                        where (mask_ac .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    call matrix_to_states(states_matrix, states)
                    
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
                    
                    y = inv_transformation(y_tf, lb, ub)
                    
                    do p=1, np
                    
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = y(p)
                        
                        end where
                    
                    end do
                    
                    do s=1, ns
                    
                        where (mask_ac .eq. 1)
                        
                            states_matrix(:,:,s) = y(np+s)
                        
                        end where
                    
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    call matrix_to_states(states_matrix, states)
                    
                    states_imd = states
                    
                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    states = states_imd
                    
                    f = cost
                    
                    nfg = nfg + 1
                    
                    if (f .lt. gx) then
                    
                        gx = f
                        jfaa = 0
                        x_tf = y_tf
                        
                        x = inv_transformation(x_tf, lb, ub)
                        
                        do p=1, np
                        
                            where (mask_ac .eq. 1) 
                            
                                parameters_matrix(:,:,p) = x(p)
                            
                            end where
                        
                        end do
                        
                        do s=1, ns
                        
                            where (mask_ac .eq. 1) 
                            
                                states_matrix(:,:,s) = x(np+s)
                            
                            end where
                        
                        end do
                        
                        call matrix_to_parameters(parameters_matrix, parameters)
                        call matrix_to_states(states_matrix, states)
                        
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
                        
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                        
                        where (mask_ac .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    call matrix_to_states(states_matrix, states)
                    
                    states_imd = states
                            
                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    states =  states_imd
                    
                    exit
                
                end if
                
                !% ======================================================================================================= %!
                !%   Maximum Number of Iteration
                !% ======================================================================================================= %!
                
                if (iter .eq. setup%maxiter * nops) then
                
                    write(*,'(4x,a)') "STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT"
            
                    do p=1, np
                        
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    do s=1, ns
                        
                        where (mask_ac .eq. 1)
                        
                            states_matrix(:,:,s) = x(np+s)
                        
                        end where
                        
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    call matrix_to_states(states_matrix, states)
                    
                    states_imd = states
                            
                    call forward(setup, mesh, input_data, parameters, parameters_bgd, &
                    & states, states_bgd, output, cost)
                    
                    states =  states_imd
                    
                    exit
                
                end if
                
            end do
            
        end subroutine optimize_sbs

        
        function transformation(x, lb, ub) result(x_tf)
        
            implicit none
            
            real(sp), dimension(np+ns), intent(in) :: x, lb, ub
            real(sp), dimension(np+ns) :: x_tf
            
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

        end function transformation
        
        
        function inv_transformation(x_tf, lb, ub) result(x)
        
            implicit none
            
            real(sp), dimension(np+ns), intent(in) :: x_tf, lb, ub
            real(sp), dimension(np+ns) :: x
            
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
            
        end function inv_transformation
        
        
        !% Calling setulb from optimize/lbfgsb.f
        !% Calling forward_b from forward/forward_b.f90
        subroutine optimize_lbfgsb(setup, mesh, input_data, parameters, states, output)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            
            integer :: n, m, iprint, nps
            integer, dimension(:), allocatable :: nbd, iwa
            integer, dimension(np+ns) :: optim_ps
            real(sp), dimension(np+ns) :: lb, ub
            real(sp), dimension(mesh%nrow, mesh%ncol, np+ns) :: &
            & ps_matrix, norm_ps_matrix, ps_b_matrix
            real(dp) :: factr, pgtol, f
            real(dp), dimension(:), allocatable :: x, l, u, g, wa
            character(lchar) :: task, csave
            logical :: lsave(4)
            integer :: isave(44)
            real(dp) :: dsave(29)
            
            type(ParametersDT) :: parameters_bgd, parameters_b
            type(StatesDT) :: states_bgd, states_imd, states_b
            type(OutputDT) :: output_b
            real(sp) :: cost, cost_b
            
            call optimize_message(setup, mesh, mesh%nac)
            
            iprint = -1
            
            nps = np + ns
            
            optim_ps(1:np) = setup%optim_parameters
            optim_ps(np+1:nps) = setup%optim_states
            
            n = mesh%nac * count(optim_ps .eq. 1)
            m = 10
            factr = 1.e7_dp
            pgtol = 1.e-12_dp
            
            allocate(nbd(n), x(n), l(n), u(n), g(n))
            allocate(iwa(3 * n))
            allocate(wa(2 * m * n + 5 * n + 11 * m * m + 8 * m))
            
            nbd = 2
            l = 0._dp
            u = 1._dp
            
            call parameters_to_matrix(parameters, ps_matrix(:,:,1:np))
            call states_to_matrix(states, ps_matrix(:,:,np+1:nps))
            
            lb(1:np) = setup%lb_parameters 
            lb(np+1:nps) = setup%lb_states 
            
            ub(1:np) = setup%ub_parameters 
            ub(np+1:nps) = setup%ub_states
            
            call normalize_matrix(ps_matrix, lb, ub, norm_ps_matrix)
            
            call optimize_matrix_to_vector(mesh, optim_ps, norm_ps_matrix, x)
            
            call ParametersDT_initialise(parameters_b, mesh)
            
            call StatesDT_initialise(states_b, mesh)
            
            call OutputDT_initialise(output_b, setup, mesh)
            
            parameters_bgd = parameters
            states_bgd = states
            
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
                            
                call optimize_vector_to_matrix(mesh, optim_ps, x, norm_ps_matrix)
                
                call unnormalize_matrix(norm_ps_matrix, lb, ub, ps_matrix)
                
                call matrix_to_parameters(ps_matrix(:,:,1:np), parameters)
                call matrix_to_states(ps_matrix(:,:,np+1:nps), states)
                
                if (task(1:2) .eq. 'FG') then
                
                    cost_b = 1._sp
                    cost = 0._sp
                    states_imd = states
                    
                    call forward_b(setup, mesh, input_data, parameters, &
                    & parameters_b, parameters_bgd, states, states_b, states_bgd, &
                    & output, output_b, cost, cost_b)
                    
                    states = states_imd
        
                    f = real(cost, kind(f))
                    
                    call parameters_to_matrix(parameters_b, ps_b_matrix(:,:,1:np))
                    call states_to_matrix(states_b, ps_b_matrix(:,:,np+1:nps))
                    
                    call optimize_matrix_to_vector(mesh, optim_ps, ps_b_matrix, g)
                    
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

        states_imd = states
        
        call forward(setup, mesh, input_data, parameters, &
        & parameters_bgd, states, states_bgd, output, cost)
            
        states = states_imd
            
        end subroutine optimize_lbfgsb
        

        subroutine optimize_matrix_to_vector(mesh, mask, matrix, vector)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(:), intent(in) :: mask
            real(sp), dimension(:,:,:), intent(in) :: matrix
            real(dp), dimension(:), intent(inout) :: vector
            
            integer :: i, k, col, row
            
            k = 0
            
            do i=1, size(mask)
                
                if (mask(i) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%global_active_cell(row, col) .eq. 1) then
                                
                                k = k + 1
                                vector(k) = real(matrix(row, col, i), kind(vector))
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if
            
            end do
        
        end subroutine optimize_matrix_to_vector
        
        
        subroutine optimize_vector_to_matrix(mesh, mask, vector, matrix)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, dimension(:), intent(in) :: mask
            real(dp), dimension(:), intent(in) :: vector
            real(sp), dimension(:,:,:), intent(inout) :: matrix
            
            integer :: i, k, col, row
            
            k = 0
            
            do i=1, size(mask)
                
                if (mask(i) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%global_active_cell(row, col) .eq. 1) then
                                
                                k = k + 1
                                
                                matrix(row, col, i) = real(vector(k), kind(matrix))
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if
            
            end do
        
        end subroutine optimize_vector_to_matrix
    
        
        subroutine normalize_matrix(matrix, lb, ub, norm_matrix)
            
            implicit none
            
            real(sp), dimension(:,:,:), intent(in) :: matrix
            real(sp), dimension(:), intent(in) :: lb, ub
            real(sp), dimension(:,:,:), intent(inout) :: norm_matrix
            
            integer :: i
            
            do i=1, size(matrix, 3)
            
                norm_matrix(:,:,i) = (matrix(:,:,i) - lb(i)) / (ub(i) - lb(i))
            
            end do
            
        end subroutine normalize_matrix
        
        
        subroutine unnormalize_matrix(norm_matrix, lb, ub, matrix)
            
            implicit none
            
            real(sp), dimension(:,:,:), intent(in) :: norm_matrix
            real(sp), dimension(:), intent(in) :: lb, ub
            real(sp), dimension(:,:,:), intent(inout) :: matrix
            
            integer :: i
            
            do i=1, size(matrix, 3)
            
                matrix(:,:,i) = norm_matrix(:,:,i) * (ub(i) - lb(i)) + lb(i)
            
            end do
            
        end subroutine unnormalize_matrix
        
        
        subroutine optimize_message(setup, mesh, nx)
            
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            integer, intent(in) :: nx
            
            integer :: i, flag
            character(lchar) :: msg, imd_char
            
            write(*,'(a)') "</> Optimize Model J"
            write(*,'(4x,4a)') "Algorithm: ", "'", trim(setup%algorithm), "'"
            write(*,'(4x,4a)') "Jobs function: ", "'", trim(setup%jobs_fun), "'"
            write(*,'(4x,a,i0)') "Nx: ", nx
        
            msg = ""
            flag = 1
            
            do i=1, np
                
                if (setup%optim_parameters(i) .gt. 0) then
                    
                    msg(flag:flag + len_trim(name_parameters(i))) = trim(name_parameters(i))
                    
                    flag = flag + len_trim(name_parameters(i)) + 1
                    
                end if
            
            end do
            
            write(*,'(4x,a,i0,3a)') "Np: ", count(setup%optim_parameters .eq. 1), " [ ", trim(msg), " ] "
            
            msg = ""
            flag = 1
            
            do i=1, ns
                
                if (setup%optim_states(i) .gt. 0) then
                    
                    msg(flag:flag + len_trim(name_states(i))) = trim(name_states(i))
                    
                    flag = flag + len_trim(name_states(i)) + 1
                    
                end if
            
            end do
            
            write(*,'(4x,a,i0,3a)') "Ns: ", count(setup%optim_states .eq. 1), " [ ", trim(msg), " ] "
        
            msg = ""
            flag = 1
            
            do i=1, mesh%ng
                
                if (mesh%wgauge(i) .gt. 0) then
                    
                    msg(flag:flag + len_trim(mesh%code(i))) = trim(mesh%code(i))
                
                    flag = flag + len_trim(mesh%code(i)) + 1
                    
                end if
            
            end do
            
            write(*,'(4x,a,i0,3a)') "Ng: ", count(mesh%wgauge .gt. 0), " [ ", trim(msg), " ] "
            
            msg = ""
            flag = 1
            
            do i=1, mesh%ng
                
                if (mesh%wgauge(i) .gt. 0) then
                
                    write(imd_char,'(f0.6)') mesh%wgauge(i)
                    
                    msg(flag:flag + len_trim(imd_char)) = trim(imd_char)
                
                    flag = flag + len_trim(imd_char) + 1
                    
                end if
            
            end do
            
            write(*,'(4x,a,i0,3a)') "wg: ", count(mesh%wgauge .gt. 0), " [ ", trim(msg), " ] "
            write(*,*) ""
            
        end subroutine optimize_message
        
end module mw_optimize
