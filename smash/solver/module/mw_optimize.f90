!%    This module `mw_optimize` encapsulates all SMASH optimize (type, subroutines, functions)
module mw_optimize
    
    use mwd_common !% only: sp, dp, lchar, np, ns
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT, parameters_to_matrix, matrix_to_parameters
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    implicit none
    
    public :: optimize_sbs, optimize_lbfgsb
    
    private :: transformation, inv_transformation
    
    contains
        
        !% Calling forward from forward/forward.f90
        subroutine optimize_sbs(setup, mesh, input_data, parameters, states, output, cost)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            type(StatesDT) :: init_states
            integer, dimension(mesh%nrow, mesh%ncol) :: mask_ac
            integer, dimension(2) :: loc_ac
            integer :: iter, nfg, ia, iaa, iam, jf, jfa, jfaa, j, p, pp
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_matrix
            real(sp), dimension(np) :: x, x_tf, y, y_tf, z_tf, sdx, lb_tf, ub_tf
            real(sp) :: gx, ga, clg, ddx, dxn, nop, f
            
            where (mesh%global_active_cell .eq. 1)
            
                mask_ac = 1
            
            else where
            
                mask_ac = 0
            
            end where
        
            loc_ac = maxloc(mask_ac)
            
            init_states = states
            
            call forward(setup, mesh, input_data, parameters, states, output, cost)
            
            call parameters_to_matrix(parameters, parameters_matrix)
            
            x = parameters_matrix(loc_ac(1), loc_ac(2), :)
            
            x_tf = transformation(setup, x)
            lb_tf = transformation(setup, setup%lb_parameters)
            ub_tf = transformation(setup, setup%ub_parameters)

            gx = cost
            ga = gx
            clg = 0.7_sp ** (1._sp / real(np))
            z_tf = x_tf
            sdx = 0._sp
            ddx = 0.64_sp
            dxn = ddx
            ia = 0
            iaa = 0
            iam = 0
            jfaa = 0
            nfg = 0
            
            nop = count(setup%optim_parameters .eq. 1)
            
            do iter=1, int(setup%maxiter * nop + 1)
                
                if (dxn .gt. ddx) dxn = ddx
                if (ddx .gt. 2._sp) ddx = dxn
                
                do p=1, np
                
                    if (setup%optim_parameters(p) .eq. 1) then
                    
                        y_tf = x_tf
                        
                        do 7 j=1, 2
                        
                            jf = 2 * j - 3
                            if (p .eq. iaa .and. jf .eq. -jfaa) goto 7
                            if (x_tf(p) .le. lb_tf(p) .and. jf .lt. 0) goto 7
                            if (x_tf(p) .ge. ub_tf(p) .and. jf .gt. 0) goto 7
                            
                            y_tf(p) = x_tf(p) + jf * ddx
                            if (y_tf(p) .lt. lb_tf(p)) y_tf(p) = lb_tf(p)
                            if (y_tf(p) .gt. ub_tf(p)) y_tf(p) = ub_tf(p)
                            
                            y = inv_transformation(setup, y_tf)
                            
                            do pp=1, np
                            
                                where (mask_ac .eq. 1)
                                
                                    parameters_matrix(:,:,pp) = y(pp)
                                
                                end where
                                
                            end do
                            
                            call matrix_to_parameters(parameters_matrix, parameters)
                            
                            states = init_states
                            
                            call forward(setup, mesh, input_data, parameters, states, output, cost)
                            
                            f = cost
                            nfg = nfg + 1
                            
                            if (f .lt. gx) then
                            
                                z_tf = y_tf
                                gx = f
                                ia = p
                                jfa = jf
                            
                            end if
                            
                        7 continue

                    end if
                
                end do
                
                iaa = ia
                jfaa = jfa
                
                if (ia .ne. 0) then
                
                    x_tf = z_tf
                    x = inv_transformation(setup, x_tf)
                    
                    do p=1, np
                            
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = x(p)
                        
                        end where
                        
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    
                    sdx = clg * sdx
                    sdx(ia) = (1._sp - clg) * real(jfa) * ddx + clg * sdx(ia)
                    
                    iam = iam + 1
                    
                    if (iam .gt. 2 * nop) then
                        
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
                
                if (iter .gt. 4 * nop) then
                
                    do p=1, np
                    
                        if (setup%optim_parameters(p) .eq. 1) then
                        
                            y_tf(p) = x_tf(p) + sdx(p)
                            if (y_tf(p) .lt. lb_tf(p)) y_tf(p) = lb_tf(p)
                            if (y_tf(p) .gt. ub_tf(p)) y_tf(p) = ub_tf(p)
                        
                        end if
                    
                    end do
                    
                    y = inv_transformation(setup, y_tf)
                    
                    do p=1, np
                    
                        where (mask_ac .eq. 1)
                        
                            parameters_matrix(:,:,p) = y(p)
                        
                        end where
                    
                    end do
                    
                    call matrix_to_parameters(parameters_matrix, parameters)
                    
                    states = init_states
                    
                    call forward(setup, mesh, input_data, parameters, states, output, cost)
                    
                    f = cost
                    
                    nfg = nfg + 1
                    
                    if (f .lt. gx) then
                    
                        gx = f
                        jfaa = 0
                        x_tf = y_tf
                        
                        x = inv_transformation(setup, x_tf)
                        
                        do p=1, np
                        
                            where (mask_ac .eq. 1) 
                            
                                parameters_matrix(:,:,p) = x(p)
                            
                            end where
                        
                        end do
                        
                        call matrix_to_parameters(parameters_matrix, parameters)
                        
                        if (gx .lt. ga - 2) then
                        
                            ga = gx
                            
                        end if
                    
                    end if
                
                end if
                
                ia = 0
                
            end do

        end subroutine optimize_sbs
        
        
        function transformation(setup, x) result(x_tf)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            real(sp), dimension(np), intent(in) :: x
            real(sp), dimension(np) :: x_tf
            
            integer :: p
            
            do p=1, np
            
                if (setup%lb_parameters(p) .lt. 0._sp) then
                    
                    x_tf(p) = sinh(x(p))
                    
                else if (setup%lb_parameters(p) .ge. 0._sp .and. setup%ub_parameters(p) .le. 1._sp) then
                    
                    x_tf(p) = log(x(p) / (1._sp - x(p)))
                    
                else
                
                    x_tf(p) = log(x(p))
                
                end if
 
            end do

        end function transformation
        
        
        function inv_transformation(setup, x_tf) result(x)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            real(sp), dimension(np), intent(in) :: x_tf
            real(sp), dimension(np) :: x
            
            integer :: p
            
            do p=1, np
            
                if (setup%lb_parameters(p) .lt. 0._sp) then
                    
                    x(p) = asinh(x_tf(p))
                    
                else if (setup%lb_parameters(p) .ge. 0._sp .and. setup%ub_parameters(p) .le. 1._sp) then
                    
                    x(p) = exp(x_tf(p)) / (1._sp + exp(x_tf(p)))
                    
                else
                
                    x(p) = exp(x_tf(p))
                
                end if
 
            end do
            
        end function inv_transformation
        
        !% Calling setulb from optimize/lbfgsb.f
        !% Calling forward_b from forward/forward_b.f90
        subroutine optimize_lbfgsb(setup, mesh, input_data, parameters, states, output, cost)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            integer :: n, m, iprint, iwriteX
            integer, dimension(:), allocatable :: nbd, iwa
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: &
            & parameters_matrix, norm_parameters_matrix, parameters_b_matrix
            real(dp) :: factr, pgtol, f
            real(dp), dimension(:), allocatable :: x, l, u, g, wa
            character(lchar) :: task, csave
            logical :: lsave(4)
            integer :: isave(44)
            real(dp) :: dsave(29)
            
            type(ParametersDT) :: parameters_b
            type(StatesDT) :: states_b, init_states
            type(OutputDT) :: output_b
            real(sp) :: cost_b
            
            iprint = 99
            
            n = mesh%nac * count(setup%optim_parameters .eq. 1)
            m = 10
            factr=1.e7_dp
            pgtol=1.e-12_dp
            
            allocate(nbd(n), x(n), l(n), u(n), g(n))
            allocate (iwa(3 * n))
            allocate (wa(2 * m * n + 5 * n + 11 * m * m + 8 * m))
            
            nbd = 2
            l = 0._dp
            u = 1._dp

            call parameters_to_matrix(parameters, parameters_matrix)
            
            call normalize_matrix(setup, mesh, parameters_matrix, norm_parameters_matrix)
            
            call matrix_to_vector(setup, mesh, norm_parameters_matrix, x)
            
            call ParametersDT_initialise(parameters_b, setup, mesh)
            
            call StatesDT_initialise(states_b, setup, mesh)
            
            call OutputDT_initialise(output_b, setup, mesh)
            
            init_states = states
            
            task = 'START'
            iwriteX=0
            do while((task(1:2) .eq. 'FG' .or. task .eq. 'NEW_X' .or. &
                    & task .eq. 'START'))
                    
                call setulb(n      ,&    ! dimension of the problem
                            m      ,&    ! number of corrections of limited memory (approx. Hessian) 
                            x      ,&    ! control
                            l       ,&   ! lower bound on control
                            u       ,&   ! upper bound on control
                            nbd     ,&   ! type of bounds
                            f       ,&   ! value of the (cost) function at xx
                            g       ,&   ! value of the (cost) gradient at xx
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
                                
                call vector_to_matrix(setup, mesh, x, norm_parameters_matrix)
                
                call unnormalize_matrix(setup, mesh, norm_parameters_matrix, parameters_matrix)
                
                call matrix_to_parameters(parameters_matrix, parameters)
                
                if (task(1:2) .eq. 'FG') then
                
                    cost_b = 1._sp
                    cost = 0._sp
                    states = init_states
                    
                    call forward_b(setup, mesh, input_data, parameters, &
                    & parameters_b, states, states_b, output, output_b, cost, cost_b)
        
                    f = real(cost, kind(f))
                    
                    call parameters_to_matrix(parameters_b, parameters_b_matrix)
                    
                    call matrix_to_vector(setup, mesh, parameters_b_matrix, g)
 
                end if
                
                if (isave(30) .ge. setup%maxiter) then
                    
                    task='STOP: TOTAL NO. of iteration EXCEEDS LIMIT'
                    
                end if
                    
            end do

        states = init_states
        
        call forward(setup, mesh, input_data, parameters, states, output, cost)

        end subroutine optimize_lbfgsb
        
        
        subroutine matrix_to_vector(setup, mesh, matrix, vector)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(in) :: matrix
            real(dp), dimension(:), intent(inout) :: vector

            integer :: p, k, col, row
            
            k = 0
            
            do p=1, np
                
                if (setup%optim_parameters(p) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%global_active_cell(row, col) .eq. 1) then
                                
                                k = k + 1
                                vector(k) = real(matrix(row, col, p), kind(vector))
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if
            
            end do
        
        end subroutine matrix_to_vector
        
        
        subroutine vector_to_matrix(setup, mesh, vector, matrix)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            real(dp), dimension(:), intent(in) :: vector
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(inout) :: matrix
            
            integer :: p, k, col, row
            
            k = 0
            
            do p=1, np
                
                if (setup%optim_parameters(p) .gt. 0) then
                
                    do col=1, mesh%ncol
                    
                        do row=1, mesh%nrow
                        
                            if (mesh%global_active_cell(row, col) .eq. 1) then
                                
                                k = k + 1
                                
                                matrix(row, col, p) = real(vector(k), kind(matrix))
                                
                            end if
                            
                        end do
                        
                    end do
                
                end if
            
            end do
        
        end subroutine vector_to_matrix
        
        
        subroutine normalize_matrix(setup, mesh, matrix, norm_matrix)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(in) :: matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(inout) :: norm_matrix
        
            integer :: p
            
            do p=1, np
                
                norm_matrix(:,:,p) = (matrix(:,:,p) - setup%lb_parameters(p)) / &
                & (setup%ub_parameters(p) - setup%lb_parameters(p))
            
            end do
            
        end subroutine normalize_matrix
        
        
        subroutine unnormalize_matrix(setup, mesh, norm_matrix, matrix)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(in) :: norm_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, np), intent(inout) :: matrix
        
            integer :: p
            
            do p=1, np
                
                matrix(:,:,p) = norm_matrix(:,:,p) * &
                & (setup%ub_parameters(p) - setup%lb_parameters(p)) + setup%lb_parameters(p)
            
            end do
            
        end subroutine unnormalize_matrix
        
end module mw_optimize
