!%    This module `mw_cost` encapsulates all SMASH cost (type, subroutines, functions)
module mwd_cost
    
    use mwd_common  !% only: sp, dp, lchar, np, ns
    use mwd_setup  !% only: SetupDT
    use mwd_mesh   !%only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_output !% only: OutputDT

    implicit none
    
    public :: compute_jobs
    
    contains
    
        subroutine compute_jobs(setup, mesh, input_data, output, jobs)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(OutputDT), intent(inout) :: output
            real(sp), intent(out) :: jobs
            
            real(sp), dimension(setup%ntime_step - setup%optim_start_step + 1) :: qo, qs
            real(sp), dimension(mesh%ng) :: jobs_gauge
            real(sp) :: imd
            integer :: g, row, col
            
            jobs = 0._sp
            jobs_gauge = 0._sp
            
            do g=1, mesh%ng
            
                qs = output%qsim(g, setup%optim_start_step:setup%ntime_step) &
                & * setup%dt / mesh%area(g) * 1e3_sp
                
                row = mesh%gauge_pos(1, g)
                col = mesh%gauge_pos(2, g)
                
                qo = input_data%qobs(g, setup%optim_start_step:setup%ntime_step) &
                & * setup%dt / (real(mesh%drained_area(row, col)) * mesh%dx * mesh%dx) &
                & * 1e3_sp
                
                if (any(qo .ge. 0._sp)) then
                
                    select case(setup%jobs_fun)
                    
                    case("nse")
            
                        jobs_gauge(g) = nse(qo, qs)
                        
                    case("kge")
                    
                        jobs_gauge(g) = kge(qo, qs)
                        
                    case("kge2")
                    
                        imd = kge(qo, qs)
                        jobs_gauge(g) = imd * imd
                        
                    case("se")
                    
                        jobs_gauge(g) = se(qo, qs)
                        
                    case("rmse")
                    
                        jobs_gauge(g) = rmse(qo, qs)
                        
                    case("logarithmique")
                    
                        jobs_gauge(g) = logarithmique(qo, qs)

                    end select
    
                end if
                
            end do
            
            do g=1, mesh%ng
                
                jobs = jobs + mesh%wgauge(g) * jobs_gauge(g)
            
            end do

        end subroutine compute_jobs
        
        
!%        subroutine compute_jreg(setup, 


!%        end subroutine compute_jreg
        
        function nse(x, y) result(res)
    
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

        
!%        subroutine reg_evolution(setup, domain, nbz, param_reg, param, &
!%    & optim, alpha, omega, reg)
        
!%        use module_smash_setup
!%        use module_smash_mesh
!%        use common_data
        
!%        implicit none
        
!%        type(model_setup),intent(in) :: setup
!%        type(mesh),intent(in) :: domain
!%        integer :: nbz
!%        real, dimension(domain%nbx,domain%nby,nbz) :: param_reg, param
!%        integer, dimension(nbz) :: optim
!%        real, dimension(nbz) :: alpha, omega
!%        real :: reg

!%        real :: deviation,deviation_total ! Euclidian Norm
!%        real :: penalty,penalty_total ! penalty term
!%        integer :: ix,iy,p
!%        real :: dy
!%        integer :: ixmin,ixmax,iymin,iymax !index des dérivés
        
!%        !initialisation
!%        reg = 0.
        
!%        dy = setup%dx
!%        deviation_total = 0.0
!%        do p=1, nbz
            
!%            if (optim(p) .gt. 0) then
!%                deviation = 0.
                
!%                do ix=1, domain%nbx
                    
!%                    do iy=1, domain%nby
!%                        deviation = deviation+&
!%                        &(1./((alpha(p)**2.) * &
!%                        & (omega(p)*1000./setup%dx)**0.5)) * &
!%                        & ((param(ix,iy,p)-param_reg(ix,iy,p))**2.)
!%                    end do
                    
!%                end do
!%                deviation_total = deviation_total + deviation
                
!%            end if
            
!%        end do
        
!%        ! penality term
!%        penalty_total = 0.0
!%        do p=1, nbz ! loop on all parameters
            
!%            if (optim(p) .gt. 0) then
!%                penalty = 0.
            
!%                do ix=1, domain%nbx
            
!%                    do iy=1, domain%nby

!%                        ixmin = ix - 1
!%                        ixmax = ix + 1
!%                        iymin = iy - 1
!%                        iymax = iy + 1

!%                        !condition limite !
!%                        if (ix .eq. 1) ixmin = ix
!%                        if (ix .eq. domain%nbx) ixmax = ix
!%                        if (iy .eq. 1) iymin = iy
!%                        if (iy .eq. domain%nby) iymax = iy

!%                        !derivée seconde
!%                        penalty=penalty+1./((omega(p)*1000./setup%dx)**0.5)*&
!%                        &( &
!%                        &((omega(p)*1000./setup%dx)*(&
!%                        &(param(ixmax,iy,p)-param(ix,iy,p))&
!%                        &-(param(ix,iy,p)-param(ixmin,iy,p))&
!%                        &))**2. + &
!%                        &((omega(p)*1000./dy)*(&
!%                        &(param(ix,iymax,p)-param(ix,iy,p))&
!%                        &-(param(ix,iy,p)-param(ix,iymin,p))&
!%                        &))**2.&
!%                        &)
                        
!%                    end do
                    
!%                end do
!%                penalty_total = penalty_total + penalty
            
!%            end if
        
!%        end do
        
!%        reg = deviation_total + penalty_total

!%    end subroutine reg_evolution


end module mwd_cost
