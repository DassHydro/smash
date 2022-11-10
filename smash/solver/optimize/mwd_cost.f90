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
!%      [9]  logarithmique
!%      [10] reg_prior

module mwd_cost
    
    use md_constant !% only: sp, dp, lchar, np, ns
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
            !% logarithmique
            
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(OutputDT), intent(inout) :: output
            real(sp), intent(out) :: jobs
            
            real(sp), dimension(setup%ntime_step - setup%optimize%optimize_start_step + 1) :: qo, qs
            real(sp), dimension(mesh%ng) :: gauge_jobs
            real(sp) :: imd
            integer :: g, row, col
            
            jobs = 0._sp
            gauge_jobs = 0._sp
            
            do g=1, mesh%ng
            
                qs = output%qsim(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                & * setup%dt / mesh%area(g) * 1e3_sp
                
                row = mesh%gauge_pos(g, 1)
                col = mesh%gauge_pos(g, 2)
                
                qo = input_data%qobs(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                & * setup%dt / (real(mesh%drained_area(row, col)) * mesh%dx * mesh%dx) &
                & * 1e3_sp
                
                if (any(qo .ge. 0._sp)) then
                
                    select case(setup%optimize%jobs_fun)
                    
                    case("nse")
            
                        gauge_jobs(g) = nse(qo, qs)
                        
                    case("kge")
                    
                        gauge_jobs(g) = kge(qo, qs)
                        
                    case("kge2")
                    
                        imd = kge(qo, qs)
                        gauge_jobs(g) = imd * imd
                        
                    case("se")
                    
                        gauge_jobs(g) = se(qo, qs)
                        
                    case("rmse")
                    
                        gauge_jobs(g) = rmse(qo, qs)
                        
                    case("logarithmique")
                    
                        gauge_jobs(g) = logarithmique(qo, qs)

                    end select
    
                end if
                
            end do
            
            do g=1, mesh%ng
                
                jobs = jobs + setup%optimize%wgauge(g) * gauge_jobs(g)
            
            end do

        end subroutine compute_jobs
        
        !% WIP
        subroutine compute_jreg(setup, mesh, parameters, parameters_bgd, states, states_bgd, jreg)
        
            !% Notes
            !% -----
            !%
            !% Jreg computation subroutine
            !%
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
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_matrix, parameters_bgd_matrix
            real(sp), dimension(mesh%nrow, mesh%ncol, ns) :: states_matrix, states_bgd_matrix
            
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
            
                parameters_jreg = reg_prior(mesh, np, parameters_matrix, parameters_bgd_matrix)
                states_jreg = reg_prior(mesh, ns, states_matrix, states_bgd_matrix)
                
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
        
        
        function logarithmique(x, y) result(res)
            
            !% Notes
            !% -----
            !%
            !% Logarithmique (LGRM) computation function
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

        end function logarithmique


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
