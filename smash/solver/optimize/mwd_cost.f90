!%    This module `mwd_cost` encapsulates all SMASH cost (type, subroutines, functions)
!%    This module is wrapped and differentiated.
!%
!%      contains
!%
!%      [1]  compute_jobs
!%      [2]  compute_jreg
!%      [3]  compute_cost
!%      [4]  nse
!%      [5]  kge_components
!%      [6]  kge
!%      [7]  se
!%      [8]  rmse
!%      [9]  logarithmic
!%      [10] reg_prior

module mwd_cost
    
    use md_constant !% only: sp, dp, lchar, GNP, GNS
    use mwd_setup  !% only: SetupDT
    use mwd_mesh   !%only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT, Hyper_ParametersDT
    use mwd_states !% only: StatesDT, Hyper_StatesDT
    use mwd_output !% only: OutputDT
    use mwd_parameters_manipulation !% only: get_parameters
    use mwd_states_manipulation !%only: get_states

    implicit none
    
    public :: compute_jobs, compute_jreg, compute_cost
    
    contains
        
        !% Way to improve: try do one single for loop to compute all cost function
        !% ATM, each cost function are computed separately with n for loop
        subroutine compute_jobs(setup, mesh, input_data, output, jobs)
        
            !% Notes
            !% -----
            !%
            !% Jobs computation subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, OutputDT,
            !% it returns the result of Jobs computation
            !%
            !% Jobs = f(Q*,Q)
            !%
            !% See Also
            !% --------
            !% nse
            !% kge
            !% se
            !% rmse
            !% logarithmic
            
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(OutputDT), intent(inout) :: output
            real(sp), intent(out) :: jobs
            
            real(sp), dimension(setup%ntime_step - setup%optimize%optimize_start_step + 1) :: po, qo, qs
            real(sp), dimension(mesh%ng) :: gauge_jobs
            real(sp) :: imd, j_imd
            integer :: g, row, col, j
            
            jobs = 0._sp
            
            do g=1, mesh%ng
            
                gauge_jobs = 0._sp
            
                if (setup%optimize%wgauge(g) .gt. 0._sp) then

                    po = input_data%mean_prcp(g, setup%optimize%optimize_start_step:setup%ntime_step)
                
                    qs = output%qsim(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                    & * setup%dt / mesh%area(g) * 1e3_sp
                    
                    row = mesh%gauge_pos(g, 1)
                    col = mesh%gauge_pos(g, 2)
                    
                    qo = input_data%qobs(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                    & * setup%dt / (real(mesh%flwacc(row, col)) * mesh%dx * mesh%dx) &
                    & * 1e3_sp
                    
                    do j=1, setup%optimize%njf

                        if (any(qo .ge. 0._sp)) then
                        
                            select case(setup%optimize%jobs_fun(j))
                            
                            case("nse")
                    
                                j_imd = nse(qo, qs)
                                
                            case("kge")
                            
                                j_imd = kge(qo, qs)
                                
                            case("kge2")
                            
                                imd = kge(qo, qs)
                                j_imd = imd * imd
                                
                            case("se")
                            
                                j_imd = se(qo, qs)
                                
                            case("rmse")
                            
                                j_imd = rmse(qo, qs)
                                
                            case("logarithmic")
                            
                                j_imd = logarithmic(qo, qs)

                            case("Crc", "Cfp2", "Cfp10", "Cfp50", "Cfp90", "Erc", "Elt", "Epf") ! CASE OF SIGNATURES

                                j_imd = signature(po, qo, qs, & 
                                & setup%optimize%mask_event(g, setup%optimize%optimize_start_step:setup%ntime_step), &
                                & setup%optimize%jobs_fun(j)) 

                            end select
            
                        end if

                        gauge_jobs(g) = gauge_jobs(g) + setup%optimize%wjobs_fun(j) * j_imd 

                    end do
                    
                end if
                
                jobs = jobs + setup%optimize%wgauge(g) * gauge_jobs(g)
                
            end do

        end subroutine compute_jobs
        
        !% WIP
        subroutine compute_jreg(setup, mesh, parameters, parameters_bgd, states, states_bgd, jreg)
        
            !% Notes
            !% -----
            !%
            !% Jreg computation subroutine
            !% Given SetupDT, MeshDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd,
            !% it returns the result of Jreg computation
            !%
            !% Jreg = f(theta_bgd,theta)
            !%
            !% See Also
            !% --------
            !% reg_prior
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(ParametersDT), intent(in) :: parameters, parameters_bgd
            type(StatesDT), intent(in) :: states, states_bgd
            real(sp), intent(inout) :: jreg
            
            real(sp) :: parameters_jreg, states_jreg
            real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: parameters_matrix, parameters_bgd_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, GNS) :: states_matrix, states_bgd_matrix
            
            call get_parameters(mesh, parameters, parameters_matrix)
            call get_parameters(mesh, parameters_bgd, parameters_bgd_matrix)
            
            call get_states(mesh, states, states_matrix)
            call get_states(mesh, states_bgd, states_bgd_matrix)
            
            jreg = 0._sp
            parameters_jreg = 0._sp
            states_jreg = 0._sp
            
            select case(setup%optimize%jreg_fun)
            
            !% Normalize prior between parameters and states
            case("prior")
            
                parameters_jreg = reg_prior(mesh, GNP, parameters_matrix, parameters_bgd_matrix)
                states_jreg = reg_prior(mesh, GNS, states_matrix, states_bgd_matrix)
                
            end select
            
            jreg = parameters_jreg + states_jreg

        end subroutine compute_jreg


        subroutine compute_cost(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)
        
            !% Notes
            !% -----
            !%
            !% cost computation subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd, OutputDT
            !% it returns the result of cost computation
            !%
            !% cost = Jobs + wJreg * Jreg
            !%
            !% See Also
            !% --------
            !% compute_jobs
            !% compute_jreg
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters, parameters_bgd
            type(StatesDT), intent(in) :: states, states_bgd
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            real(sp) :: jobs, jreg
            
            call compute_jobs(setup, mesh, input_data, output, jobs)
            
            !% Only compute in case wjreg > 0
            if (setup%optimize%wjreg .gt. 0._sp) then
            
                call compute_jreg(setup, mesh, parameters, parameters_bgd, states, states_bgd, jreg)
                
            else
            
                jreg = 0._sp
                
            end if
            
            cost = jobs + setup%optimize%wjreg * jreg
            output%cost = cost
            
        end subroutine compute_cost
        
        
        !% TODO comment and refactorize
        subroutine hyper_compute_cost(setup, mesh, input_data, &
        & hyper_parameters, hyper_parameters_bgd, hyper_states, &
        & hyper_states_bgd, output, cost)
        
            !% Notes
            !% -----
            !%
            !% cost computation subroutine
            !%
            !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd, OutputDT
            !% it returns the result of cost computation
            !%
            !% cost = Jobs + wJreg * Jreg
            !%
            !% See Also
            !% --------
            !% compute_jobs
            !% compute_jreg
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_bgd
            type(Hyper_StatesDT), intent(in) :: hyper_states, hyper_states_bgd
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            real(sp) :: jobs, jreg
            
            call compute_jobs(setup, mesh, input_data, output, jobs)
            
            jreg = 0._sp
            
            cost = jobs + setup%optimize%wjreg * jreg
            output%cost = cost
        
        end subroutine hyper_compute_cost
        

        function nse(x, y) result(res)
            
            !% Notes
            !% -----
            !%
            !% NSE computation function
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of NSE computation
            !% num = sum(x**2) - 2 * sum(x*y) + sum(y**2)
            !% den = sum(x**2) - n * mean(x) ** 2
            !% NSE = num / den
    
            implicit none
            
            real(sp), dimension(:), intent(in) :: x, y
            real(sp) :: res
            
            real(sp) :: sum_x, sum_xx, sum_yy, sum_xy, mean_x, num, den
            integer :: i, n
            
            !% Metric computation
            n = 0
            sum_x = 0.
            sum_xx = 0.
            sum_yy = 0.
            sum_xy = 0.
            
            do i=1, size(x)
            
                if (x(i) .ge. 0.) then
                    
                    n = n + 1
                    sum_x = sum_x + x(i)
                    sum_xx = sum_xx + (x(i) * x(i))
                    sum_yy = sum_yy + (y(i) * y(i))
                    sum_xy = sum_xy + (x(i) * y(i))
                
                end if
            
            end do
            
            mean_x = sum_x / n
        
            !% NSE numerator / denominator
            num = sum_xx - 2 * sum_xy + sum_yy
            den = sum_xx - n * mean_x * mean_x
            
            !% NSE criterion
            res = num / den

        end function nse


        subroutine kge_components(x, y, r, a, b)
        
            !% Notes
            !% -----
            !%
            !% KGE components computation subroutine
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns KGE components r, a, b
            !% r = cov(x,y) / std(y) / std(x)
            !% a = mean(y) / mean(x)
            !% b = std(y) / std(x)
        
            implicit none
            
            real, dimension(:), intent(in) :: x, y
            real, intent(inout) :: r, a, b
            
            real :: sum_x, sum_y, sum_xx, sum_yy, sum_xy, mean_x, mean_y, &
            & var_x, var_y, cov
            integer :: n, i
            
            ! Metric computation
            n = 0
            sum_x = 0.
            sum_y = 0.
            sum_xx = 0.
            sum_yy = 0.
            sum_xy = 0.
            
            do i=1, size(x)
            
                if (x(i) .ge. 0.) then
                    
                    n = n + 1
                    sum_x = sum_x + x(i)
                    sum_y = sum_y + y(i)
                    sum_xx = sum_xx + (x(i) * x(i))
                    sum_yy = sum_yy + (y(i) * y(i))
                    sum_xy = sum_xy + (x(i) * y(i))
                
                end if
                
            end do
            
            mean_x = sum_x / n
            mean_y = sum_y / n
            var_x = (sum_xx / n) - (mean_x * mean_x)
            var_y = (sum_yy / n) - (mean_y * mean_y)
            cov = (sum_xy / n) - (mean_x * mean_y)
            
            ! KGE components (r, alpha, beta)
            r = (cov / sqrt(var_x)) / sqrt(var_y)
            a = sqrt(var_y) / sqrt(var_x)
            b = mean_y / mean_x
    
        end subroutine kge_components


        function kge(x, y) result(res)
        
            !% Notes
            !% -----
            !%
            !% KGE computation function
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of KGE computation
            !% KGE = sqrt((1 - r) ** 2 + (1 - a) ** 2 + (1 - b) ** 2)
            !%
            !% See Also
            !% --------
            !% kge_components
        
            implicit none
            
            real, dimension(:), intent(in) :: x, y
            real :: res
            
            real :: r, a, b
            
            call kge_components(x, y, r, a, b)
            
            ! KGE criterion
            res = sqrt(&
            & (r - 1) * (r - 1) + (b - 1) * (b - 1) + (a - 1) * (a - 1) &
            & )
        
        end function kge


        function se(x, y) result(res)
        
            !% Notes
            !% -----
            !%
            !% Square Error (SE) computation function
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of SE computation
            !% SE = sum((x - y) ** 2)
            
            implicit none
            
            real, dimension(:), intent(in) :: x, y
            real :: res
            
            integer :: i
            
            res = 0.
            
            do i=1, size(x)
                
                if (x(i) .ge. 0.) then
                
                    res = res + (x(i) - y(i)) * (x(i) - y(i))
                
                end if
            
            end do
        
        end function se


        function rmse(x, y) result(res)
        
            !% Notes
            !% -----
            !%
            !% Root Mean Square Error (RMSE) computation function
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of SE computation
            !% RMSE = sqrt(SE / n)
            !%
            !% See Also
            !% --------
            !% se
        
            implicit none
            
            real, dimension(:), intent(in) :: x, y
            real :: res
            
            integer :: i, n
            
            n = 0
            
            do i=1, size(x)
                
                if (x(i) .ge. 0.) then
                
                    n = n + 1
                    
                end if
            
            end do
            
            res = sqrt(se(x,y) / n)
            
        end function rmse



        function logarithmic(x, y) result(res)
            
            !% Notes
            !% -----
            !%
            !% Logarithmic (LGRM) computation function
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of LGRM computation
            !% LGRM = sum(x * log(y/x) ** 2)
            
            implicit none
            
            real, dimension(:), intent(in) :: x, y
            real :: res
            
            integer :: i
            
            res = 0.
            
            do i=1, size(x)
            
                if (x(i) .gt. 0. .and. y(i) .gt. 0.) then
                
                    res = res + x(i) * log(y(i) / x(i)) * log(y(i) / x(i))
                
                end if
            
            
            end do

        end function logarithmic


        subroutine insertion_sort(arr, n)

            !% Notes
            !% -----
            !%
            !% insertion sort more efficient than bubble sort
            !% less efficient than quicksort but differentiable 

            implicit none
            
            integer, intent(in) :: n
            real, dimension(n), intent(inout) :: arr

            integer :: i, j
            real :: tmp_sort

            do i = 1, n-1

                tmp_sort = arr(i)

                j = i - 1

                do while (j >= 0 .and. arr(j) > tmp_sort)

                    arr(j+1) = arr(j)

                    j = j - 1

                end do

                arr(j+1) = tmp_sort

            end do
            
        end subroutine insertion_sort


        function quantile(dat, p) result(res)

            !% Notes
            !% -----
            !%
            !% quantile function using linear interpolation
            !% similar to numpy.quantile

            implicit none

            real, intent(in) :: p
            real, dimension(:), intent(in) :: dat
            real, dimension(size(dat)) :: sorted_dat
            integer :: n
            real :: res, q1, q2, frac

            res = 0.
            
            n = size(dat)

            sorted_dat = dat

            call insertion_sort(sorted_dat, n)

            frac = (n - 1) * p + 1

            if (frac <= 1) then

                res = sorted_dat(1)

            else if (frac >= n) then

                res = sorted_dat(n)

            else
                q1 = sorted_dat(int(frac))

                q2 = sorted_dat(int(frac) + 1)

                res = q1 + (q2 - q1) * (frac - int(frac)) ! linear interpolation

            end if
            
        end function quantile


        subroutine quantile_signature(qo, qs, p, num, den)

            !% Notes
            !% -----
            !%
            !% compute quantiles of observed (qo) and simulated signature (qs)

            implicit none

            real, dimension(:), intent(in) :: qo, qs
            real, intent(in) :: p

            real, intent(inout) :: num, den

            real, dimension(size(qo)) :: pos_qo, pos_qs

            integer :: i, j, n

            n = size(qo)

            pos_qo = 0.
            pos_qs = 0.

            j = 0

            do i = 1, n

                if (qo(i) .ge. 0. .and. qs(i) .ge. 0.) then

                    j = j + 1

                    pos_qo(j) = qo(i)
                    pos_qs(j) = qs(i)

                end if

            end do

            num = quantile(pos_qs(1:j), p)

            den = quantile(pos_qo(1:j), p)

        end subroutine quantile_signature


        function signature(po, qo, qs, mask_event, stype) result(res)

            !% Notes
            !% -----
            !%
            !% Signatures-based cost computation (SBC)
            !%
            !% Given two single precision array (x, y) of dim(1) and size(n),
            !% it returns the result of SBC computation
            !% SBC_i = (s_i(y)/s_i(x) - 1) ** 2
            !% where i is a signature i and s_i is its associated signature computation function 

            implicit none

            real, dimension(:), intent(in) :: po, qo, qs
            integer, dimension(:), intent(in) :: mask_event
            character(len=*), intent(in) :: stype
            
            real :: res
            
            logical, dimension(size(mask_event)) :: lgc_mask_event
            integer :: n_event, i, j, start_event, ntime_step_event
            real :: sum_qo, sum_qs, sum_po, &
            & max_qo, max_qs, max_po, num, den
            integer :: imax_qo, imax_qs, imax_po
            
            res = 0.
            
            n_event = 0
            
            if (stype(:1) .eq. "E") then
                
                ! Reverse loop on mask_event to find number of event (array sorted filled with 0)
                do i=size(mask_event), 1, -1
                    
                    if (mask_event(i) .gt. 0) then
                        
                        n_event = mask_event(i)
                        exit
                        
                    end if
                
                end do
                
                do i=1, n_event
                
                    lgc_mask_event = (mask_event .eq. i)
                    
                    do j=1, size(mask_event)
                    
                        if (lgc_mask_event(j)) then
                            
                            start_event = j
                            exit
                            
                        end if
                    
                    end do
                
                    ntime_step_event = count(lgc_mask_event)
                    
                    sum_qo = 0.
                    sum_qs = 0.
                    sum_po = 0.
                    
                    max_qo = 0.
                    max_qs = 0.
                    max_po = 0.
                    
                    imax_qo = 0
                    imax_qs = 0
                    imax_po = 0
                    
                    do j=start_event, start_event + ntime_step_event - 1
                    
                        if (qo(j) .ge. 0. .and. po(j) .ge. 0.) then
                    
                            sum_qo = sum_qo + qo(j)
                            sum_qs = sum_qs + qs(j)
                            sum_po = sum_po + po(j)
                            
                            if (qo(j) .gt. max_qo) then
                            
                                max_qo = qo(j)
                                imax_qo = j
                            
                            end if
                            
                            if (qs(j) .gt. max_qs) then
                            
                                max_qs = qs(j)
                                imax_qs = j
                            
                            end if
                            
                            if (po(j) .gt. max_po) then
                            
                                max_po = po(j)
                                imax_po = j
                            
                            end if
                            
                        end if
                        
                    end do
                    
                    select case(stype)
                    
                    case("Epf")
                    
                        num = max_qs
                        den = max_qo
                        
                    case("Elt")
                    
                        num = imax_qs - imax_po
                        den = imax_qo - imax_po
                        
                    case("Erc")
                    
                        if (sum_po .gt. 0.) then
                        
                            num = sum_qs / sum_po
                            den = sum_qo / sum_po
                        
                        end if
                    
                    end select
                    
                    if (den .gt. 0.) then
                    
                        res = res + (num / den - 1.) * (num / den - 1.)
                    
                    end if
                    
                end do
                
                if (n_event .gt. 0) then
                
                    res = res / n_event
                
                end if
                
            else
                
                select case(stype)
                
                case("Crc")

                    sum_qo = 0.
                    sum_qs = 0.
                    sum_po = 0.
                    
                    do i=1, size(qo)
                        
                        if (qo(i) .ge. 0. .and. po(i) .ge. 0.) then
                            
                            sum_qo = sum_qo + qo(i)
                            sum_qs = sum_qs + qs(i)
                            sum_po = sum_po + po(i)
                        
                        end if
                        
                    end do
                    
                    if (sum_po .gt. 0.) then
                        
                        num = sum_qs / sum_po
                        den = sum_qo / sum_po
                    
                    end if

                case("Cfp2")

                    call quantile_signature(qo, qs, 0.02, num, den)

                case("Cfp10")

                    call quantile_signature(qo, qs, 0.1, num, den)

                case("Cfp50")

                    call quantile_signature(qo, qs, 0.5, num, den)

                case("Cfp90")

                    call quantile_signature(qo, qs, 0.9, num, den)
                
                end select
                
                if (den .gt. 0.) then
                
                    res = (num / den - 1.) * (num / den - 1.)
                
                end if
            
            end if

        end function signature
            

        !% TODO refactorize
        function reg_prior(mesh, size_mat3, matrix, matrix_bgd) result(res)
        
            !% Notes
            !% -----
            !%
            !% Prior regularization (PR) computation function
            !%
            !% Given two matrix of dim(3) and size(mesh%nrow, mesh%ncol, size_mat3), 
            !% it returns the result of PR computation. (Square Error between matrix)
            !% 
            !% PR = sum((mat1 - mat2) ** 2)
            
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            integer, intent(in) :: size_mat3
            real(sp), dimension(mesh%nrow, mesh%ncol, size_mat3), intent(in) :: matrix, matrix_bgd
            real(sp) :: res
            
            res = sum((matrix - matrix_bgd) * (matrix - matrix_bgd))
        
        end function reg_prior

end module mwd_cost
