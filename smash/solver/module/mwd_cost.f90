!%    This module `mw_cost` encapsulates all SMASH cost (type, subroutines, functions)
module mwd_cost
    
    use mwd_common  !% only: sp, dp, lchar, np, ns
    use mwd_setup  !% only: SetupDT
    use mwd_mesh   !%only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_output !% only: OutputDT

    implicit none
    
    contains
    
        subroutine compute_jobs(setup, mesh, input_data, output, jobs)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(OutputDT), intent(inout) :: output
            real(sp), intent(out) :: jobs
            
            real(sp), dimension(setup%ntime_step - setup%optim_start_step + 1) :: qo, qs
!            real(sp), dimension(mesh%ng) :: jobs_gauge
            integer :: g, row, col
            
            jobs = 0._sp
            
            do g=1, mesh%ng
            
                if (mesh%optim_gauge(g) .eq. 1) then
            
                    qs = output%qsim(g, setup%optim_start_step:setup%ntime_step) &
                    & * setup%dt / mesh%area(g) * 1e3_sp
                    
                    row = mesh%gauge_pos(1, g)
                    col = mesh%gauge_pos(2, g)
                    
                    qo = input_data%qobs(g, setup%optim_start_step:setup%ntime_step) &
                    & * setup%dt / (real(mesh%drained_area(row, col)) * mesh%dx * mesh%dx) &
                    & * 1e3_sp
                    
                    if (any(qo .ge. 0._sp)) then
                
                        jobs = jobs + nse(qo, qs)
        
                    end if
                
                end if
                
            end do
        
        
        end subroutine compute_jobs
        
        
!~         subroutine compute_jreg
        
        
!~         end subroutine compute_jreg
        
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

        
end module mwd_cost
