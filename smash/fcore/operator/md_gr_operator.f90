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
!%      - gr4_time_step
!%      - gr5_time_step
!%      - grd_time_step
!%      - loieau_time_step

module md_gr_operator

    use md_constant !% only : sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnDT
    use mwd_atmos_manipulation !% get_ac_atmos_data_time_step

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

    subroutine gr_production(pn, en, cp, beta, hp, pr, perc)

        implicit none

        real(sp), intent(in) :: pn, en, cp, beta
        real(sp), intent(inout) :: hp
        real(sp), intent(out) :: pr, perc

        real(sp) :: inv_cp, ps, es, hp_imd

        inv_cp = 1._sp/cp
        pr = 0._sp

        ps = cp*(1._sp - hp*hp)*tanh(pn*inv_cp)/ &
        & (1._sp + hp*tanh(pn*inv_cp))

        es = (hp*cp)*(2._sp - hp)*tanh(en*inv_cp)/ &
        & (1._sp + (1._sp - hp)*tanh(en*inv_cp))

        hp_imd = hp + (ps - es)*inv_cp

        if (pn .gt. 0) then

            pr = pn - (hp_imd - hp)*cp

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc*inv_cp

    end subroutine gr_production

    subroutine gr_exchange(kexc, ht, l)

        implicit none

        real(sp), intent(in) :: kexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = kexc*ht**3.5_sp

    end subroutine gr_exchange

    subroutine gr_threshold_exchange(kexc, aexc, ht, l)

        implicit none

        real(sp), intent(in) :: kexc, aexc
        real(sp), intent(inout) :: ht
        real(sp), intent(out) :: l

        l = kexc*(ht - aexc)

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

    subroutine gr4_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_hi, ac_hp, ac_ht, ac_qt, returns)

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
        integer :: row, col, k
        real(sp) :: beta, pn, en, pr, perc, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, ac_hi, ac_hp, ac_ht, &
        !$OMP& ac_qt) &
        !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(pn, en, ac_cp(k), beta, ac_hp(k), pr, perc)

                    call gr_exchange(ac_kexc(k), ac_ht(k), l)

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
                if (returns%stats_flag) then
                    returns%stats%internal_fluxes(row, col, 1) = pn
                    returns%stats%internal_fluxes(row, col, 2) = en
                    returns%stats%internal_fluxes(row, col, 3) = pr
                    returns%stats%internal_fluxes(row, col, 4) = perc
                    returns%stats%internal_fluxes(row, col, 5) = l
                    returns%stats%internal_fluxes(row, col, 6) = prr
                    returns%stats%internal_fluxes(row, col, 7) = prd
                    returns%stats%internal_fluxes(row, col, 8) = qr
                    returns%stats%internal_fluxes(row, col, 9) = qd
                end if
                if (returns%pn_flag) returns%pn(row, col, time_step) = pn
                if (returns%en_flag) returns%en(row, col, time_step) = en
                if (returns%pr_flag) returns%pr(row, col, time_step) = pr
                if (returns%perc_flag) returns%perc(row, col, time_step) = perc
                if (returns%lexc_flag) returns%lexc(row, col, time_step) = l
                if (returns%prr_flag) returns%prr(row, col, time_step) = prr
                if (returns%prd_flag) returns%prd(row, col, time_step) = prd
                if (returns%qr_flag) returns%qr(row, col, time_step) = qr
                if (returns%qd_flag) returns%qd(row, col, time_step) = qd
                !$AD end-exclude

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr4_time_step

    subroutine gr5_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ci, ac_cp, ac_ct, &
    & ac_kexc, ac_aexc, ac_hi, ac_hp, ac_ht, ac_qt, returns)

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
        integer :: row, col, k
        real(sp) :: beta, pn, en, pr, perc, l, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ci, ac_cp, beta, ac_ct, ac_kexc, ac_aexc, ac_hi, &
        !$OMP& ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, pn, en, pr, perc, l, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    call gr_interception(ac_prcp(k), ac_pet(k), ac_ci(k), &
                    & ac_hi(k), pn, en)

                    call gr_production(pn, en, ac_cp(k), beta, ac_hp(k), pr, perc)

                    call gr_threshold_exchange(ac_kexc(k), ac_aexc(k), ac_ht(k), l)

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
                if (returns%stats_flag) then
                    returns%stats%internal_fluxes(row, col, 1) = pn
                    returns%stats%internal_fluxes(row, col, 2) = en
                    returns%stats%internal_fluxes(row, col, 3) = pr
                    returns%stats%internal_fluxes(row, col, 4) = perc
                    returns%stats%internal_fluxes(row, col, 5) = l
                    returns%stats%internal_fluxes(row, col, 6) = prr
                    returns%stats%internal_fluxes(row, col, 7) = prd
                    returns%stats%internal_fluxes(row, col, 8) = qr
                    returns%stats%internal_fluxes(row, col, 9) = qd
                end if
                if (returns%pn_flag) returns%pn(row, col, time_step) = pn
                if (returns%en_flag) returns%en(row, col, time_step) = en
                if (returns%pr_flag) returns%pr(row, col, time_step) = pr
                if (returns%perc_flag) returns%perc(row, col, time_step) = perc
                if (returns%lexc_flag) returns%lexc(row, col, time_step) = l
                if (returns%prr_flag) returns%prr(row, col, time_step) = prr
                if (returns%prd_flag) returns%prd(row, col, time_step) = prd
                if (returns%qr_flag) returns%qr(row, col, time_step) = qr
                if (returns%qd_flag) returns%qd(row, col, time_step) = qd
                !$AD end-exclude

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine gr5_time_step

    subroutine grd_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_cp, ac_ct, ac_hp, &
    & ac_ht, ac_qt, returns)

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
        integer :: row, col, k
        real(sp) :: ei, pn, en, pr, perc, prr, qr

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_cp, ac_ct, ac_hp, ac_ht, ac_qt) &
        !$OMP& private(row, col, k, ei, pn, en, pr, perc, prr, qr)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(pn, en, ac_cp(k), 1000._sp, ac_hp(k), pr, perc)

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
                if (returns%stats_flag) then
                    returns%stats%internal_fluxes(row, col, 1) = ei
                    returns%stats%internal_fluxes(row, col, 2) = pn
                    returns%stats%internal_fluxes(row, col, 3) = en
                    returns%stats%internal_fluxes(row, col, 4) = pr
                    returns%stats%internal_fluxes(row, col, 5) = perc
                    returns%stats%internal_fluxes(row, col, 6) = prr
                    returns%stats%internal_fluxes(row, col, 7) = qr
                end if
                if (returns%ei_flag) returns%ei(row, col, time_step) = ei
                if (returns%pn_flag) returns%pn(row, col, time_step) = pn
                if (returns%en_flag) returns%en(row, col, time_step) = en
                if (returns%pr_flag) returns%pr(row, col, time_step) = pr
                if (returns%perc_flag) returns%perc(row, col, time_step) = perc
                if (returns%prr_flag) returns%prr(row, col, time_step) = prr
                if (returns%qr_flag) returns%qr(row, col, time_step) = qr
                !$AD end-exclude

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine grd_time_step

    subroutine loieau_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_ca, ac_cc, ac_kb, &
    & ac_ha, ac_hc, ac_qt, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in):: ac_mlt
        real(sp), dimension(mesh%nac), intent(in):: ac_ca, ac_cc, ac_kb
        real(sp), dimension(mesh%nac), intent(inout):: ac_ha, ac_hc
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: beta, ei, pn, en, pr, perc, prr, prd, qr, qd

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt

        ! Beta percolation parameter is time step dependent
        beta = (9._sp/4._sp)*(86400._sp/setup%dt)**0.25_sp
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_ca, beta, ac_cc, ac_kb, ac_ha, ac_hc, ac_qt) &
        !$OMP& private(row, col, k, ei, pn, en, pr, perc, prr, prd, qr, qd)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ei = min(ac_pet(k), ac_prcp(k))

                    pn = max(0._sp, ac_prcp(k) - ei)

                    en = ac_pet(k) - ei

                    call gr_production(pn, en, ac_ca(k), beta, ac_ha(k), pr, perc)

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
                if (returns%stats_flag) then
                    returns%stats%internal_fluxes(row, col, 1) = ei
                    returns%stats%internal_fluxes(row, col, 2) = pn
                    returns%stats%internal_fluxes(row, col, 3) = en
                    returns%stats%internal_fluxes(row, col, 4) = pr
                    returns%stats%internal_fluxes(row, col, 5) = perc
                    returns%stats%internal_fluxes(row, col, 6) = prr
                    returns%stats%internal_fluxes(row, col, 7) = prd
                    returns%stats%internal_fluxes(row, col, 8) = qr
                    returns%stats%internal_fluxes(row, col, 9) = qd
                end if
                if (returns%ei_flag) returns%ei(row, col, time_step) = ei
                if (returns%pn_flag) returns%pn(row, col, time_step) = pn
                if (returns%en_flag) returns%en(row, col, time_step) = en
                if (returns%pr_flag) returns%pr(row, col, time_step) = pr
                if (returns%perc_flag) returns%perc(row, col, time_step) = perc
                if (returns%prr_flag) returns%prr(row, col, time_step) = prr
                if (returns%prd_flag) returns%prd(row, col, time_step) = prd
                if (returns%qr_flag) returns%qr(row, col, time_step) = qr
                if (returns%qd_flag) returns%qd(row, col, time_step) = qd
                !$AD end-exclude

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine loieau_time_step

end module md_gr_operator
