!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr_interception
!%      - gr_production
!%      - gr_exchange
!%      - gr_threshold_exchange
!%      - gr_transfer
!%      - gr_production_transfer_ode
!%      - gr_production_transfer_ode_mlp
!%      - gr4_time_step
!%      - gr4_mlp_time_step
!%      - gr4_ri_time_step
!%      - gr4_ode_time_step
!%      - gr4_ode_mlp_time_step
!%      - gr5_time_step
!%      - gr5_mlp_time_step
!%      - gr5_ri_time_step
!%      - gr6_time_step
!%      - gr6_mlp_time_step
!%      - grc_time_step
!%      - grc_mlp_time_step
!%      - grd_time_step
!%      - grd_mlp_time_step
!%      - loieau_time_step
!%      - loieau_mlp_time_step

module md_gr_operator

    use md_constant !% only : sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use mwd_atmos_manipulation !% get_ac_atmos_data_time_step
    use md_algebra !% only: solve_linear_system_2vars
    use md_neural_network !% only: forward_mlp

    implicit none

contains

    subroutine gr_interception(prcp, pet, ci, hi, pn, en)

        implicit none

        real(sp), intent(in) :: prcp, pet, ci
        real(sp), intent(inout) :: hi
        real(sp), intent(out) :: pn, en

        real(sp) :: ei

        ei = min(pet, prcp + hi*ci)

        pn = max(0._sp, prcp - ci*(1._sp - hi) - ei)

        en = pet - ei

        hi = hi + (prcp - ei - pn)/ci

    end subroutine gr_interception

    subroutine gr_production(fq_ps, fq_es, pn, en, cp, beta, hp, pr, perc, ps, es)

        implicit none

        real(sp), intent(in) :: fq_ps, fq_es, pn, en, cp, beta
        real(sp), intent(inout) :: hp
        real(sp), intent(out) :: pr, perc, ps, es

        real(sp) :: inv_cp, hp_imd

        inv_cp = 1._sp/cp
        pr = 0._sp

        ps = cp*(1._sp - hp*hp)*tanh(pn*inv_cp)/ &
        & (1._sp + hp*tanh(pn*inv_cp))
        ps = min(pn, (1._sp + fq_ps)*ps)  ! Range of correction coef: (0, 2)

        es = (hp*cp)*(2._sp - hp)*tanh(en*inv_cp)/ &
        & (1._sp + (1._sp - hp)*tanh(en*inv_cp))
        es = min(en, (1._sp + fq_es)*es)  ! Range of correction coef: (0, 2)

        hp_imd = hp + (ps - es)*inv_cp

        if (pn .gt. 0) then

            pr = pn - ps

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc*inv_cp

    end subroutine gr_production

    subroutine gr_ri_production(pn, en, cp, beta, alpha1, hp, pr, perc, ps, es, dt)

        implicit none

        real(sp), intent(in) :: pn, en, cp, beta, alpha1
        real(sp), intent(in) :: dt
        real(sp), intent(inout) :: hp
        real(sp), intent(out) :: pr, perc, ps, es

        real(sp) :: inv_cp, hp_imd
        real(sp) :: lambda, gam, inv_lambda

        inv_cp = 1._sp/cp
        pr = 0._sp
        gam = 1._sp - exp(-pn*alpha1)
        lambda = sqrt(1._sp - gam)
        inv_lambda = 1._sp/lambda

        ps = cp*inv_lambda*tanh(lambda*pn*inv_cp)*(1._sp - (lambda*hp)**2) &
        & /(1._sp + lambda*hp*tanh(lambda*pn*inv_cp)) - gam*dt

        es = (hp*cp)*(2._sp - hp)*tanh(en*inv_cp)/ &
        & (1._sp + (1._sp - hp)*tanh(en*inv_cp))

        hp_imd = hp + (ps - es)*inv_cp

        if (pn .gt. 0) then

            pr = pn - (hp_imd - hp)*cp

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc*inv_cp

    end subroutine gr_ri_production

    subroutine gr_exchange(fq_l, kexc, ht, l)

        implicit none

        real(sp), intent(in) :: fq_l, kexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = (1._sp + fq_l)*kexc*ht**3.5_sp  ! Range of correction coef: (0, 2)

    end subroutine gr_exchange

    subroutine gr_threshold_exchange(fq_l, kexc, aexc, ht, l)

        implicit none

        real(sp), intent(in) :: fq_l, kexc, aexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = (1._sp + fq_l)*kexc*(ht - aexc)  ! Range of correction coef: (0, 2)

    end subroutine gr_threshold_exchange

    subroutine gr_transfer(n, prcp, pr, ct, ht, q)

        implicit none

        real(sp), intent(in) :: n, prcp, pr, ct
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: q

        real(sp) :: pr_imd, ht_imd, nm1, d1pnm1

        nm1 = n - 1._sp
        d1pnm1 = 1._sp/nm1

        if (prcp .lt. 0._sp) then

            pr_imd = ((ht*ct)**(-nm1) - ct**(-nm1))**(-d1pnm1) - (ht*ct)

        else

            pr_imd = pr

        end if

        ht_imd = max(1.e-6_sp, ht + pr_imd/ct)

        ht = (((ht_imd*ct)**(-nm1) + ct**(-nm1))**(-d1pnm1))/ct

        q = (ht_imd - ht)*ct

    end subroutine gr_transfer

    subroutine gr_exponential_transfer(pre, be, he, qe)

        implicit none

        real(sp), intent(in) :: pre, be
        real(sp), intent(inout) :: he
        real(sp), intent(out) :: qe
        real(sp) :: he_star, ar

        he_star = he + pre
        ar = he_star/be
        if (ar .lt. -7._sp) then
            qe = be*exp(ar)
        else if (ar .gt. 7._sp) then
            qe = he_star + be/exp(ar)
        else
            qe = be*log(exp(ar) + 1._sp)
        end if

        he = he_star - qe

    end subroutine gr_exponential_transfer

    subroutine gr_production_transfer_ode(pn, en, cp, ct, kexc, hp, ht, q, l)
        !% Solve state-space ODE system with implicit Euler

        implicit none

        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, q
        real(sp), intent(out) :: l

        real(sp), dimension(2, 2) :: jacob
        real(sp), dimension(2) :: dh, delta_h
        real(sp) :: inv_cp, hp0, ht0, dt, fhp, fht, tmp_j
        logical :: converged
        integer :: j
        integer :: maxiter = 10
        ! integer :: n_subtimesteps = 2

        inv_cp = 1._sp/cp

        ! dt = 1._sp/real(n_subtimesteps, sp)
        dt = 1._sp

        ! do i = 1, n_subtimesteps

        hp0 = hp
        ht0 = ht

        converged = .false.
        j = 0

        do while ((.not. converged) .and. (j .lt. maxiter))

            fhp = (1._sp - hp**2)*pn - hp*(2._sp - hp)*en
            dh(1) = hp - hp0 - dt*fhp*inv_cp

            fht = ct*ht**5 - 0.9_sp*pn*hp**2 - kexc*ht**3.5_sp
            dh(2) = ht - ht0 + dt*fht/ct  ! fht here is -fht

            jacob(1, 1) = 1._sp + dt*2._sp*(hp*(pn - en) + en)*inv_cp

            jacob(1, 2) = 0._sp

            jacob(2, 1) = dt*1.8_sp*pn*hp/ct

            tmp_j = 5._sp*ht**4 - 3.5_sp*kexc*(ht**2.5_sp)/ct
            jacob(2, 2) = 1._sp + dt*tmp_j

            call solve_linear_system_2vars(jacob, delta_h, dh)

            hp = hp + delta_h(1)
            if (hp .le. 0._sp) hp = 1.e-6_sp
            if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp

            ht = ht + delta_h(2)
            if (ht .le. 0._sp) ht = 1.e-6_sp
            if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

            converged = (sqrt((delta_h(1)/hp)**2 + (delta_h(2)/ht)**2) .lt. 1.e-6_sp)
            j = j + 1

        end do

        ! end do

        l = kexc*ht**3.5_sp

        q = ct*ht**5 + 0.1_sp*pn*hp**2 + l

    end subroutine gr_production_transfer_ode

    subroutine gr_production_transfer_ode_mlp(fq, pn, en, cp, ct, kexc, hp, ht, q, l)
        !% Solve state-space ODE system with explicit Euler and MLP

        implicit none

        real(sp), dimension(5), intent(in) :: fq  ! fixed NN output size
        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, q
        real(sp), intent(out) :: l

        real(sp) :: inv_cp, dt, fhp, fht
        ! integer :: i
        ! integer :: n_subtimesteps = 4

        inv_cp = 1._sp/cp

        ! dt = 1._sp/real(n_subtimesteps, sp)
        dt = 1._sp

        !do i = 1, n_subtimesteps

        fhp = (1._sp + fq(1))*pn*(1._sp - hp**2) &
        & - (1._sp + fq(2))*en*hp*(2._sp - hp)  ! Range of correction pn, en: (0, 2)

        hp = hp + dt*fhp*inv_cp

        if (hp .le. 0._sp) hp = 1.e-6_sp
        if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp

        fht = (0.9_sp*(1._sp - fq(3)**2))*(1._sp + fq(1))*pn*hp**2 &
        & + (1._sp + fq(4))*kexc*ht**3.5_sp &
        & - (1._sp + fq(5))*ct*ht**5  ! Range of correction c0.9: (1, 0); kexc, ct: (0, 2)

        ht = ht + dt*fht/ct

        if (ht .le. 0._sp) ht = 1.e-6_sp
        if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

        !end do

        l = (1._sp + fq(4))*kexc*ht**3.5_sp  ! Range of correction kexc: (0, 2)

        q = (1._sp + fq(5))*ct*ht**5 &  ! Range of correction ct: (0, 2)
        & + (0.1_sp + 0.9_sp*fq(3)**2) &  ! Range of correction c0.1: (1, 10)
        & *(1._sp + fq(1))*pn*hp**2 + l  ! Range of correction pn: (0, 2)

    end subroutine gr_production_transfer_ode_mlp

    subroutine gr4_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, &
        !$OMP& ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(0._sp, 0._sp, pn, en, ac_cp(k), beta, ac_hp(k), pr, perc, ps, es)

                    call gr_exchange(0._sp, ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr4_time_step

    subroutine gr4_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pr, perc, ps, es, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_cp, beta, ac_ct, ac_kexc, ac_hp, ac_ht, ac_qt, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_cp(k), &
                    & beta, ac_hp(k), pr, perc, ps, es)

                    call gr_exchange(output_layer(4, k), ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = (0.9_sp*(1._sp - output_layer(3, k)**2))*(pr + perc) + l  ! Range of correction c0.9: (1, 0)
                prd = (0.1_sp + 0.9_sp*output_layer(3, k)**2)*(pr + perc)  ! Range of correction c0.1: (1, 10)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine gr4_mlp_time_step

    subroutine gr4_ri_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_alpha1, ac_alpha2, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(in) :: ac_alpha1, ac_alpha2
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, split

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, beta, &
        !$OMP&  ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, split)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_ri_production(pn, en, ac_cp(k), beta, ac_alpha1(k), ac_hp(k), pr, perc, ps, es, setup%dt)

                    call gr_exchange(0._sp, ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if
                split = 0.9_sp*tanh(ac_alpha2(k)*pn)**2 + 0.1_sp

                prr = (1._sp - split)*(pr + perc) + l
                prd = split*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr4_ri_time_step

    subroutine gr4_ode_time_step(setup, mesh, input_data, options, returns, time_step, &
    & ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: l

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Production and transfer without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                call gr_production_transfer_ode(pn(k), en(k), ac_cp(k), ac_ct(k), &
                & ac_kexc(k), ac_hp(k), ac_ht(k), ac_qt(k), l)

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), l, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do

    end subroutine gr4_ode_time_step

    subroutine gr4_ode_mlp_time_step(setup, mesh, input_data, options, returns, time_step, &
    & weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, &
    & ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: l

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_cp, ac_ct, ac_kexc, &
        !$OMP& ac_hp, ac_ht, ac_qt, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, l)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                call gr_production_transfer_ode_mlp(output_layer(:, k), pn(k), en(k), &
                & ac_cp(k), ac_ct(k), ac_kexc(k), ac_hp(k), ac_ht(k), ac_qt(k), l)

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), l, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine gr4_ode_mlp_time_step

    subroutine gr5_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, ac_aexc, &
        !$OMP& ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(0._sp, 0._sp, pn, en, ac_cp(k), beta, ac_hp(k), pr, perc, ps, es)

                    call gr_threshold_exchange(0._sp, ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr5_time_step

    subroutine gr5_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc, &
    & ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pr, perc, ps, es, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_cp, beta, ac_ct, ac_kexc, ac_aexc, ac_hp, ac_ht, ac_qt, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_cp(k), &
                    & beta, ac_hp(k), pr, perc, ps, es)

                    call gr_threshold_exchange(output_layer(4, k), ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = (0.9_sp*(1._sp - output_layer(3, k)**2))*(pr + perc) + l  ! Range of correction c0.9: (1, 0)
                prd = (0.1_sp + 0.9_sp*output_layer(3, k)**2)*(pr + perc)  ! Range of correction c0.1: (1, 10)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine gr5_mlp_time_step

    subroutine gr5_ri_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_alpha1, ac_alpha2, ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(in) :: ac_alpha1, ac_alpha2
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, split

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, beta, &
        !$OMP& ac_alpha1, ac_alpha2, ac_ct, ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, split)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_ri_production(pn, en, ac_cp(k), beta, ac_alpha1(k), ac_hp(k), pr, perc, ps, es, setup%dt)

                    call gr_threshold_exchange(0._sp, ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                split = 0.9_sp*tanh(ac_alpha2(k)*pn)**2 + 0.1_sp

                prr = (1._sp - split)*(pr + perc) + l
                prd = split*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr5_ri_time_step

    subroutine gr6_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_be, ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_he, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_be, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht, ac_he
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pn, en, pr, perc, ps, es, l, prr, pre, prd, qr, qd, qe

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_be, ac_kexc, &
        !$OMP& ac_aexc, ac_hi, ac_hp, ac_ht, ac_he, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, pre, prd, qr, qd, qe)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(0._sp, 0._sp, pn, en, ac_cp(k), beta, ac_hp(k), pr, perc, ps, es)

                    call gr_threshold_exchange(0._sp, ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.6_sp*0.9_sp*(pr + perc) + l
                pre = 0.4_sp*0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                call gr_exponential_transfer(pre, ac_be(k), ac_he(k), qe)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd + qe

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, pre, qr, qd, qe, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr6_time_step

    subroutine gr6_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ci, ac_cp, ac_ct, ac_be, ac_kexc, ac_aexc, &
    & ac_hi, ac_hp, ac_ht, ac_he, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_be, ac_kexc, ac_aexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht, ac_he
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pr, perc, ps, es, l, prr, pre, prd, qr, qd, qe

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), ac_he(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_cp, beta, ac_ct, ac_be, ac_kexc, ac_aexc, ac_hp, ac_ht, ac_he, ac_qt, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, l, prr, pre, prd, qr, qd, qe)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_cp(k), &
                    & beta, ac_hp(k), pr, perc, ps, es)

                    call gr_threshold_exchange(output_layer(5, k), ac_kexc(k), ac_aexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = (0.6_sp - 0.4_sp*output_layer(4, k))* &  ! Range of correction c0.6: (5/3, 1/3)
                & (0.9_sp*(1._sp - output_layer(3, k)**2))* &  ! Range of correction c0.9: (1, 0)
                & (pr + perc) + l

                pre = (0.4_sp*(1._sp + output_layer(4, k)))* &  ! Range of correction c0.4: (0, 2)
                & (0.9_sp*(1._sp - output_layer(3, k)**2))* &  ! Range of correction c0.9: (1, 0)
                & (pr + perc) + l

                prd = (0.1_sp + 0.9_sp*output_layer(3, k)**2)*(pr + perc)  ! Range of correction c0.1: (0, 10)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)
                call gr_exponential_transfer(pre, ac_be(k), ac_he(k), qe)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + qd + qe

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), pr, perc, ps, es, l, prr, prd, pre, qr, qd, qe, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine gr6_mlp_time_step

    subroutine grc_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_cl, ac_kexc, ac_hi, ac_hp, ac_ht, ac_hl, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_cl, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht, ac_hl
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: pn, en, pr, perc, ps, es, l, prr, prl, prd, qr, ql, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ci, ac_cp, ac_ct, ac_cl, ac_kexc, &
        !$OMP& ac_hi, ac_hp, ac_ht, ac_hl, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, pn, en, pr, perc, ps, es, l, prr, prl, prd, qr, ql, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(0._sp, 0._sp, pn, en, ac_cp(k), 1000._sp, ac_hp(k), pr, perc, ps, es)

                    call gr_exchange(0._sp, ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.6_sp*0.9_sp*(pr + perc) + l
                prl = 0.4_sp*0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)
                call gr_transfer(5._sp, ac_prcp(k), prl, ac_cl(k), ac_hl(k), ql)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + ql + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn, en, pr, perc, ps, es, l, prr, prd, prl, qr, qd, ql, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine grc_time_step

    subroutine grc_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ci, ac_cp, ac_ct, ac_cl, ac_kexc, &
    & ac_hi, ac_hp, ac_ht, ac_hl, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ci, ac_cp, ac_ct, ac_cl, ac_kexc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hi, ac_hp, ac_ht, ac_hl
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: pr, perc, ps, es, l, prr, prl, prd, qr, ql, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ac_ci, ac_hi, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), ac_hi(k), pn(k), en(k))

                else

                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), ac_hl(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_cp, ac_ct, ac_cl, ac_kexc, ac_hp, ac_ht, ac_hl, ac_qt, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, l, prr, prl, prd, qr, ql, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_cp(k), &
                    & 1000._sp, ac_hp(k), pr, perc, ps, es)

                    call gr_exchange(output_layer(5, k), ac_kexc(k), ac_ht(k), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = (0.6_sp - 0.4_sp*output_layer(4, k))* &  ! Range of correction c0.6: (5/3, 1/3)
                & (0.9_sp*(1._sp - output_layer(3, k)**2))* &  ! Range of correction c0.9: (1, 0)
                & (pr + perc) + l

                prl = (0.4_sp*(1._sp + output_layer(4, k)))* &  ! Range of correction c0.4: (0, 2)
                & (0.9_sp*(1._sp - output_layer(3, k)**2))* &  ! Range of correction c0.9: (1, 0)
                & (pr + perc) + l

                prd = (0.1_sp + 0.9_sp*output_layer(3, k)**2)*(pr + perc)  ! Range of correction c0.1: (0, 10)

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)
                call gr_transfer(5._sp, ac_prcp(k), prl, ac_cl(k), ac_hl(k), ql)

                qd = max(0._sp, prd + l)

                ac_qt(k) = qr + ql + qd

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/pn(k), en(k), pr, perc, ps, es, l, prr, prd, prl, qr, qd, ql, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine grc_mlp_time_step

    subroutine grd_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_cp, ac_ct, ac_hp, &
    & ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_cp, ac_ct
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: ei, pn, en, pr, perc, ps, es, prr, qr

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_cp, ac_ct, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, ei, pn, en, pr, perc, ps, es, prr, qr)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(0._sp, 0._sp, pn, en, ac_cp(k), 1000._sp, ac_hp(k), pr, perc, ps, es)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = pr + perc

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                ac_qt(k) = qr

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/ei, pn, en, pr, perc, ps, es, prr, qr, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine grd_time_step

    subroutine grd_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_cp, ac_ct, ac_hp, ac_ht, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_cp, ac_ct
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hp, ac_ht
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, ei, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: pr, perc, ps, es, prr, qr

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ei, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei(k) = min(ac_pet(k), ac_prcp(k))
                    pn(k) = max(0._sp, ac_prcp(k) - ei(k))
                    en(k) = ac_pet(k) - ei(k)

                else

                    ei(k) = 0._sp
                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_hp(k), ac_ht(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_cp, ac_ct, ac_hp, ac_ht, ac_qt, ei, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, prr, qr)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_cp(k), &
                    & 1000._sp, ac_hp(k), pr, perc, ps, es)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = pr + perc

                call gr_transfer(5._sp, ac_prcp(k), prr, ac_ct(k), ac_ht(k), qr)

                ac_qt(k) = qr

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/ei(k), pn(k), en(k), pr, perc, ps, es, prr, qr, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine grd_mlp_time_step

    subroutine loieau_time_step(setup, mesh, input_data, options, returns, time_step, ac_mlt, ac_ca, ac_cc, ac_kb, &
    & ac_ha, ac_hc, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ca, ac_cc, ac_kb
        real(sp), dimension(mesh%nac), intent(inout) :: ac_ha, ac_hc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, ei, pn, en, pr, perc, ps, es, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, ac_prcp, ac_pet, ac_ca, beta, ac_cc, ac_kb, ac_ha, ac_hc, ac_qt) &
        !$OMP& private(row, col, k, time_step_returns, ei, pn, en, pr, perc, ps, es, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(0._sp, 0._sp, pn, en, ac_ca(k), beta, ac_ha(k), pr, perc, ps, es)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = 0.9_sp*(pr + perc)
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(4._sp, ac_prcp(k), prr, ac_cc(k), ac_hc(k), qr)

                qd = max(0._sp, prd)

                ac_qt(k) = ac_kb(k)*(qr + qd)

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/ei, pn, en, pr, perc, ps, es, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine loieau_time_step

    subroutine loieau_mlp_time_step(setup, mesh, input_data, options, returns, time_step, weight_1, bias_1, &
    & weight_2, bias_2, weight_3, bias_3, ac_mlt, ac_ca, ac_cc, ac_kb, ac_ha, ac_hc, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(setup%neurons(2), setup%neurons(1)), intent(in) :: weight_1
        real(sp), dimension(setup%neurons(2)), intent(in) :: bias_1
        real(sp), dimension(setup%neurons(3), setup%neurons(2)), intent(in) :: weight_2
        real(sp), dimension(setup%neurons(3)), intent(in) :: bias_2
        real(sp), dimension(setup%neurons(4), setup%neurons(3)), intent(in) :: weight_3
        real(sp), dimension(setup%neurons(4)), intent(in) :: bias_3
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_ca, ac_cc, ac_kb
        real(sp), dimension(mesh%nac), intent(inout) :: ac_ha, ac_hc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(setup%neurons(1)) :: input_layer
        real(sp), dimension(setup%neurons(setup%n_layers + 1), mesh%nac) :: output_layer
        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet, ei, pn, en
        integer :: row, col, k, time_step_returns
        real(sp) :: beta, pr, perc, ps, es, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp

        ! Interception with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(mesh, ac_prcp, ac_pet, ei, pn, en) &
        !$OMP& private(row, col, k)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei(k) = min(ac_pet(k), ac_prcp(k))
                    pn(k) = max(0._sp, ac_prcp(k) - ei(k))
                    en(k) = ac_pet(k) - ei(k)

                else

                    ei(k) = 0._sp
                    pn(k) = 0._sp
                    en(k) = 0._sp

                end if

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

        ! Forward MLP without OPENMP
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    input_layer(:) = (/ac_ha(k), ac_hc(k), pn(k), en(k)/)
                    call forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
                    & input_layer, output_layer(:, k))

                else
                    output_layer(:, k) = 0._sp

                end if

            end do
        end do

        ! Production and transfer with OPENMP
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, returns, output_layer, ac_prcp, ac_pet, &
        !$OMP& ac_ca, beta, ac_cc, ac_kb, ac_ha, ac_hc, ac_qt, ei, pn, en) &
        !$OMP& private(row, col, k, time_step_returns, pr, perc, ps, es, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_production(output_layer(1, k), output_layer(2, k), pn(k), en(k), ac_ca(k), &
                    & beta, ac_ha(k), pr, perc, ps, es)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = (0.9_sp*(1._sp - output_layer(3, k)**2))*(pr + perc)  ! Range of correction c0.9: (1, 0)
                prd = (0.1_sp + 0.9_sp*output_layer(3, k)**2)*(pr + perc)  ! Range of correction c0.1: (0, 10)

                call gr_transfer(4._sp, ac_prcp(k), prr, ac_cc(k), ac_hc(k), qr)

                qd = max(0._sp, prd)

                ac_qt(k) = ac_kb(k)*(qr + qd)

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

                !$AD start-exclude
                !internal fluxes
                if (returns%internal_fluxes_flag) then
                    if (allocated(returns%mask_time_step)) then
                        if (returns%mask_time_step(time_step)) then
                            time_step_returns = returns%time_step_to_returns_time_step(time_step)
                            ! the fluxes of the snow module are the first ones inside internal fluxes
                            ! due to the building of the modules so n_snow_fluxes
                            ! moves the index of the array
                            returns%internal_fluxes( &
                                row, &
                                col, &
                                time_step_returns, &
                                setup%n_snow_fluxes + 1:setup%n_snow_fluxes + setup%n_hydro_fluxes &
                                ) = (/ei(k), pn(k), en(k), pr, perc, ps, es, prr, prd, qr, qd, ac_qt(k)/)
                        end if
                    end if
                end if
                !$AD end-exclude
            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif

    end subroutine loieau_mlp_time_step

end module md_gr_operator
