!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - discharge_transformation
!%      - bayesian_compute_cost
!%      - classical_compute_jobs
!%      - classical_compute_cost
!%      - compute_cost
!%
!%      Function
!%      --------
!%
!%      - get_range_event

module mwd_cost

    use mwd_bayesian_tools !% only: compute_logPost, PriorType
    use md_constant !% only: sp, dp
    use md_stats !% only: quantile1d_r
    use mwd_metrics !% only: nse, nnse, kge, mae, mape, mse, rmse, lgrm
    use mwd_signatures !% only: rc, rchf, rclf, rch2r, cfp, ebf, elt, eff
    use md_regularization !% only: prior_regularization, smoothing_regularization
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

        res = 0

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

    subroutine discharge_tranformation(tfm, qo, qs)

        implicit none

        character(lchar), intent(in) :: tfm
        real(sp), dimension(:), intent(inout) :: qo, qs

        real(sp) :: mean_qo, e
        logical, dimension(size(qo)) :: mask

        mask = (qo .ge. 0._sp)
        mean_qo = sum(qo, mask=mask)/count(mask)
        e = 1e-2_sp*mean_qo

        select case (tfm)

        case ("sqrt")

            where (qo .ge. 0._sp) qo = sqrt(qo + e)
            where (qs .ge. 0._sp) qs = sqrt(qs + e)

        case ("inv")

            where (qo .ge. 0._sp) qo = 1._sp/(qo + e)
            where (qs .ge. 0._sp) qs = 1._sp/(qs + e)

            !% Should be reach by "keep" only. Do nothing
        case default

        end select

    end subroutine discharge_tranformation

    subroutine bayesian_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        integer :: i, j, n
        real(dp) :: log_lkh, log_prior, log_h, log_post
        logical :: feas, isnull
        character(lchar) :: mu_funk, sigma_funk
        real(dp), dimension(setup%ntime_step, options%cost%nog) :: obs, uobs, sim
        real(dp), dimension(sum(parameters%control%nbk(1:2))) :: theta
        real(dp), dimension(setup%nsep_mu, options%cost%nog) :: mu_gamma
        real(dp), dimension(setup%nsep_sigma, options%cost%nog) :: sigma_gamma
        ! Derived Type from md_BayesianTools
        type(PriorType), dimension(sum(parameters%control%nbk(1:2))) :: theta_prior
        type(PriorType), dimension(0, 0) :: mu_gamma_prior, sigma_gamma_prior, dummy_prior2d

        j = 0

        do i = 1, mesh%ng

            if (options%cost%gauge(i) .eq. 0) cycle

            j = j + 1
            obs(:, j) = input_data%response_data%q(i, options%cost%end_warmup:setup%ntime_step)
            uobs(:, j) = input_data%u_response_data%q_stdev(i, options%cost%end_warmup:setup%ntime_step)
            sim(:, j) = output%response%q(i, options%cost%end_warmup:setup%ntime_step)
            mu_gamma(:, j) = parameters%serr_mu_parameters%values(i, :)
            sigma_gamma(:, j) = parameters%serr_sigma_parameters%values(i, :)

        end do

        ! TODO: For the moment only priors for theta are handled.
        ! Priors for mu_gamma and sigma_gamma are hard-coded to non-informative dummy_prior2d
        obs = real(obs, dp)
        uobs = real(uobs, dp)
        sim = real(sim, dp)

        n = sum(parameters%control%nbk(1:2))
        if (n .gt. 0) then
            theta = parameters%control%x(1:n)
            theta_prior = options%cost%control_prior(1:n)
        end if

        mu_funk = setup%serr_mu_mapping
        mu_gamma = real(mu_gamma, dp)
        mu_gamma_prior = dummy_prior2d

        sigma_funk = setup%serr_sigma_mapping
        sigma_gamma = real(sigma_gamma, dp)
        sigma_gamma_prior = dummy_prior2d

        call compute_logPost(obs, uobs, sim, theta, theta_prior, mu_funk, mu_gamma, mu_gamma_prior, &
        & sigma_funk, sigma_gamma, sigma_gamma_prior, log_post, log_prior, log_lkh, log_h, feas, isnull)

        ! TODO: Should be count(obs .ge. 0._sp .and. uobs .ge. 0._sp)
        output%cost = -1._sp*real(log_post, sp)/size(obs)

        !$AD start-exclude
        if (returns%cost_flag) returns%cost = output%cost
        if (returns%log_lkh_flag) returns%log_lkh = real(log_lkh, sp)
        if (returns%log_prior_flag) returns%log_prior = real(log_prior, sp)
        if (returns%log_h_flag) returns%log_h = real(log_h, sp)
        !$AD end-exclude

    end subroutine bayesian_compute_cost

    subroutine classical_compute_jobs(setup, mesh, input_data, output, options, jobs)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OutputDT), intent(in) :: output
        type(OptionsDT), intent(in) :: options
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

            ! Cycle if gauge is equal to 0
            if (options%cost%gauge(i) .eq. 0) cycle

            qo = input_data%response_data%q(i, options%cost%end_warmup:setup%ntime_step)
            qs = output%response%q(i, options%cost%end_warmup:setup%ntime_step)
            where (qo .lt. 0._sp) qs = -99._sp

            ! Convert mean_prcp from mm/dt to m3/s
            mprcp = input_data%atmos_data%mean_prcp(i, options%cost%end_warmup:setup%ntime_step)* &
            & mesh%area_dln(i)*1.e-3_sp/setup%dt

            mask_event = options%cost%mask_event(i, options%cost%end_warmup:setup%ntime_step)

            do j = 1, options%cost%njoc

                call discharge_tranformation(options%cost%jobs_cmpt_tfm(j), qo, qs)

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
                    jobs_tmp = cfp(qo, 0.02_sp)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.02_sp)/jobs_tmp - 1._sp)**2
                case ("Cfp10")
                    jobs_tmp = cfp(qo, 0.1_sp)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.1_sp)/jobs_tmp - 1._sp)**2
                case ("Cfp50")
                    jobs_tmp = cfp(qo, 0.5_sp)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.5_sp)/jobs_tmp - 1._sp)**2
                case ("Cfp90")
                    jobs_tmp = cfp(qo, 0.9_sp)
                    if (jobs_tmp .gt. 0._sp) jobs_cmpt_values(i, j) = (cfp(qs, 0.9_sp)/jobs_tmp - 1._sp)**2

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
                if (options%cost%gauge(i) .eq. 0) cycle

                jobs_tmp = 0._sp

                do j = 1, options%cost%njoc

                    jobs_tmp = jobs_tmp + options%cost%wjobs_cmpt(j)*jobs_cmpt_values(i, j)

                end do

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

    end subroutine classical_compute_jobs

    subroutine classical_compute_jreg(setup, mesh, input_data, parameters, options, jreg)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OptionsDT), intent(in) :: options
        real(sp), intent(inout) :: jreg

        integer :: i
        real(sp), dimension(options%cost%njrc) :: jreg_cmpt_values

        ! Case of forward run
        if (.not. allocated(parameters%control%x)) return

        jreg_cmpt_values = 0._sp

        do i = 1, options%cost%njrc

            select case (options%cost%jreg_cmpt(i))

                ! Can be applied to any control
            case ("prior")

                jreg_cmpt_values(i) = prior_regularization(parameters)

                ! Should be only used with distributed mapping. Applied on rr_parameters and rr_initial_states
            case ("smoothing")

                jreg_cmpt_values(i) = smoothing_regularization(setup, mesh, input_data, parameters, options, .false.)

                ! Should be only used with distributed mapping. Applied on rr_parameters and rr_initial_states
            case ("hard-smoothing")

                jreg_cmpt_values(i) = smoothing_regularization(setup, mesh, input_data, parameters, options, .true.)

            end select

        end do

        jreg = sum(options%cost%wjreg_cmpt*jreg_cmpt_values)

    end subroutine classical_compute_jreg

    subroutine classical_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

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

        call classical_compute_jobs(setup, mesh, input_data, output, options, jobs)

        call classical_compute_jreg(setup, mesh, input_data, parameters, options, jreg)

        output%cost = jobs + options%cost%wjreg*jreg

        !$AD start-exclude
        if (returns%cost_flag) returns%cost = output%cost
        if (returns%jobs_flag) returns%jobs = jobs
        if (returns%jreg_flag) returns%jreg = jreg
        !$AD end-exclude

    end subroutine classical_compute_cost

    subroutine compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        if (options%cost%bayesian) then

            call bayesian_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        else

            call classical_compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        end if

    end subroutine compute_cost

end module mwd_cost
