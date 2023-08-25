!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - cls_compute_jobs
!%      - cls_compute_cost
!%      - compute_cost
!%
!%      Function
!%      --------
!%
!%      - get_range_event

module mwd_cost

    use md_constant !% only: sp
    use md_stats !% only: quantile1d_r
    use mwd_metrics !% only: nse, nnse, kge, mae, mape, mse, rmse, lgrm
    use mwd_signatures !% only: baseflow_separation
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT

    implicit none

contains

    function get_range_event(mask_event, i_event) result(res)

        implicit none

        integer, dimension(:), intent(in) :: mask_event
        integer, intent(in) :: i_event

        integer, dimension(2) :: res

        integer :: i

        res(1) = 0
        res(2) = 0

        do i = 1, size(mask_event)

            if (mask_event(i) .eq. i_event) then
                res(1) = i
                exit
            end if

        end do

        do i = size(mask_event), 1, -1

            if (mask_event(i) .eq. i_event) then
                res(2) = i
                exit
            end if

        end do

    end function get_range_event

    ! TODO FC: Can gain memory by allocate baseflow and fastflow arrays
    ! but need to handle allocation state
    subroutine cls_compute_jobs(setup, mesh, input_data, output, options, returns, jobs)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OutputDT), intent(in) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        real(sp), intent(inout) :: jobs

        integer :: i, j, k, n_computed_event
        real(sp), dimension(setup%ntime_step - options%cost%end_warmup + 1) :: qo, qs, mprcp
        integer, dimension(setup%ntime_step - options%cost%end_warmup + 1) :: mask_event
        real(sp), dimension(mesh%ng, options%cost%njoc) :: jobs_cmpt_values
        integer, dimension(2) :: range_event
        real(sp), dimension(mesh%ng) :: jobs_gauge
        real(sp) :: jobs_tmp

        jobs_cmpt_values = 0._sp

        do i = 1, mesh%ng

            ! Cycle if wgauge is equal to 0
            if (abs(options%cost%wgauge(i)) .le. 0._sp) cycle

            qo = input_data%obs_response%q(i, options%cost%end_warmup:setup%ntime_step)
            qs = output%sim_response%q(i, options%cost%end_warmup:setup%ntime_step)
            where (qo .lt. 0._sp) qs = -99._sp

            ! Convert mean_prcp from mm/dt to m3/s
            mprcp = input_data%atmos_data%mean_prcp(i, options%cost%end_warmup:setup%ntime_step)* &
            & mesh%area_dln(i)*1.e-3_sp/setup%dt

            mask_event = options%cost%mask_event(i, options%cost%end_warmup:setup%ntime_step)

            do j = 1, options%cost%njoc

                select case (options%cost%jobs_cmpt(j))

                    ! Efficiency Metrics
                case ("nse")
                    jobs_cmpt_values(i, j) = 1._sp - nse(qo, qs)
                case ("nnse")
                    jobs_cmpt_values(i, j) = 1._sp - nnse(qo, qs)
                case ("kge")
                    jobs_cmpt_values(i, j) = 1._sp - kge(qo, qs)
                case ("mae")
                    jobs_cmpt_values(i, j) = mae(qo, qs)
                case ("mape")
                    jobs_cmpt_values(i, j) = mape(qo, qs)
                case ("mse")
                    jobs_cmpt_values(i, j) = mse(qo, qs)
                case ("rmse")
                    jobs_cmpt_values(i, j) = rmse(qo, qs)
                case ("lgrm")
                    jobs_cmpt_values(i, j) = lgrm(qo, qs)

                    ! Continuous Signatures
                case ("Crc")
                    jobs_tmp = rc(mprcp, qo)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (rc(mprcp, qs)/jobs_tmp - 1._sp)**2
                case ("Crchf")
                    jobs_tmp = rchf(mprcp, qo)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (rchf(mprcp, qs)/jobs_tmp - 1._sp)**2
                case ("Crclf")
                    jobs_tmp = rclf(mprcp, qo)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (rclf(mprcp, qs)/jobs_tmp - 1._sp)**2
                case ("Crch2r")
                    jobs_tmp = rch2r(mprcp, qo)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (rch2r(mprcp, qs)/jobs_tmp - 1._sp)**2
                case ("Cfp2")
                    jobs_tmp = cfp(qo, 0.02)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.02)/jobs_tmp - 1._sp)**2
                case ("Cfp10")
                    jobs_tmp = cfp(qo, 0.1)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.1)/jobs_tmp - 1._sp)**2
                case ("Cfp50")
                    jobs_tmp = cfp(qo, 0.5)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.5)/jobs_tmp - 1._sp)**2
                case ("Cfp90")
                    jobs_tmp = cfp(qo, 0.9)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.9)/jobs_tmp - 1._sp)**2

                    ! Event Signatures
                case ("Erc")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = rc(mprcp(range_event(1):range_event(2)), qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (rc(mprcp(range_event(1):range_event(2)), qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event

                case ("Erchf")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = rchf(mprcp(range_event(1):range_event(2)), qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (rchf(mprcp(range_event(1):range_event(2)), qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Erclf")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = rclf(mprcp(range_event(1):range_event(2)), qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (rclf(mprcp(range_event(1):range_event(2)), qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Erch2r")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = rch2r(mprcp(range_event(1):range_event(2)), qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (rch2r(mprcp(range_event(1):range_event(2)), qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Eff")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = eff(qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (eff(qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Ebf")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = ebf(qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (ebf(qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Elt")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = elt(mprcp(range_event(1):range_event(2)), qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (elt(mprcp(range_event(1):range_event(2)), qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event
                case ("Epf")
                    n_computed_event = 0
                    do k = 1, options%cost%n_event(i)
                        range_event = get_range_event(mask_event, k)
                        if (range_event(1) .lt. 1) cycle
                        jobs_tmp = epf(qo(range_event(1):range_event(2)))
                        if (jobs_tmp .gt. 0._sp) then
                            jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j) + &
                            & (epf(qs(range_event(1):range_event(2))) &
                            & /jobs_tmp - 1._sp)**2
                            n_computed_event = n_computed_event + 1
                        end if
                    end do
                    jobs_cmpt_values(i, j) = jobs_cmpt_values(i, j)/n_computed_event

                    ! Should be unreachable.
                case default

                end select

            end do

        end do

        ! TODO TH: handle with alias (median, low/upp quartiles) for jobs_cmpt

        if (any(options%cost%wgauge(:) .lt. 0._sp)) then
            jobs_gauge = 0._sp

            k = 0

            do i = 1, mesh%ng
                jobs_tmp = 0._sp

                do j = 1, options%cost%njoc

                    jobs_tmp = jobs_tmp + options%cost%wjobs_cmpt(j)*jobs_cmpt_values(i, j)

                end do

                if (jobs_tmp .le. 0._sp) cycle

                k = k + 1
                jobs_gauge(k) = jobs_tmp

            end do

            jobs = quantile1d_r(jobs_gauge(1:k), abs(options%cost%wgauge(1)))

        else
            jobs = 0._sp

            do i = 1, mesh%ng

                do j = 1, options%cost%njoc

                    jobs = jobs + options%cost%wgauge(i)*options%cost%wjobs_cmpt(j)*jobs_cmpt_values(i, j)

                end do

            end do

        end if

    end subroutine cls_compute_jobs

    subroutine cls_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        real(sp) :: jobs, jreg

        jobs = 0._sp
        jreg = 0._sp

        call cls_compute_jobs(setup, mesh, input_data, output, options, returns, jobs)

        ! TODO FC: Not Implemented yet
        ! call cls_compute_jreg(setup, mesh, parameters, options, returns, jreg)

        output%cost = jobs + options%cost%wjreg*jreg

    end subroutine cls_compute_cost

    subroutine compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        select case (options%cost%variant)

        case ("cls")

            call cls_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

            ! TODO BR: Not Implemented Yet
        case ("bys")

        end select

    end subroutine compute_cost

end module mwd_cost
