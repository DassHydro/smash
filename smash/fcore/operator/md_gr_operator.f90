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
!%      - gr4_timestep
!%      - gr5_timestep
!%      - gr6_timestep
!%      - grd_timestep
!%      - loieau_timestep

module md_gr_operator

    use md_constant !% only : sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_options !% only: OptionsDT

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

    subroutine exponential_transfer(he, pre, ce, qre)

        implicit none

        real(sp), intent(in) :: pre, ce
        real(sp), intent(inout) :: he
        real(sp), intent(out) :: qre
        real(sp) :: he_star, AR
        
        he_star = he + pre
        AR = he_star / ce
        if (AR .lt. -7._sp) then
            qre = ce * exp(AR)
        elseif (AR .gt. 7._sp) then
            qre = he_star + ce / exp(AR)
        else
            qre = ce * log(exp(AR) + 1._sp)
        end if
        he = he_star - qre

    end subroutine exponential_transfer
    
    subroutine gr4_timestep(setup, mesh, options, prcp, pet, ci, cp, ct, kexc, hi, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: ci, cp, ct, kexc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: hi, hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd
        
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, prcp, pet, ci, cp, ct, kexc, hi, hp, ht, qt) &
        !$OMP& private(row, col, pn, en, pr, perc, l, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, en)

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    call gr_exchange(kexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = qr + qd

                ! Transform from mm/dt to m3/s
                qt(row, col) = qt(row, col)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr4_timestep

    subroutine gr5_timestep(setup, mesh, options, prcp, pet, ci, cp, ct, kexc, aexc, hi, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: ci, cp, ct, kexc, aexc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: hi, hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, prcp, pet, ci, cp, ct, kexc, aexc, hi, hp, ht, qt) &
        !$OMP& private(row, col, pn, en, pr, perc, l, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, en)

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    call gr_threshold_exchange(kexc(row, col), aexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = qr + qd

                ! Transform from mm/dt to m3/s
                qt(row, col) = qt(row, col)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine gr5_timestep

    subroutine gr6_timestep(setup, mesh, options, prcp, pet, ci, cp, ct, ce, kexc, aexc, hi, hp, ht, he, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: ci, cp, ct, ce, kexc, aexc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: hi, hp, ht, he
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt
        
        integer :: row, col
        real(sp) :: pn, en, pr, perc, l, prr, prd, pre, qr, qd, qre
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, prcp, pet, ci, cp, ct, ce, kexc, aexc, hi, hp, ht, he, qt) &
        !$OMP& private(row, col, pn, en, pr, perc, l, prr, prd, pre, qr, qd, qre)
        
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, en)

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    call gr_threshold_exchange(kexc(row, col), aexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if
                
                prr = 0.6_sp * 0.9_sp * (pr + perc) + l
                pre = 0.4_sp * 0.9_sp * (pr + perc) + l
                prd = 0.1_sp * (pr + perc) 
                
                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)
                
                call exponential_transfer(he(row, col), pre, ce(row, col), qre)

                qd = max(0._sp, prd + l)
                
                qt(row, col) = qr + qd + qre
    
                ! Transform from mm/dt to m3/s
                qt(row, col) = qt(row, col)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do
       
    end subroutine gr6_timestep

    subroutine grd_timestep(setup, mesh, options, prcp, pet, cp, ct, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: cp, ct
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: ei, pn, en, pr, perc, prr, qr

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, prcp, pet, cp, ct, hp, ht, qt) &
        !$OMP& private(row, col, ei, pn, en, pr, perc, prr, qr)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then

                    ei = min(pet(row, col), prcp(row, col))

                    pn = max(0._sp, prcp(row, col) - ei)

                    en = pet(row, col) - ei

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = pr + perc

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qt(row, col) = qr

                ! Transform from mm/dt to m3/s
                qt(row, col) = qt(row, col)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine grd_timestep

    subroutine loieau_timestep(setup, mesh, options, prcp, pet, ca, cc, kb, ha, hc, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: ca, cc, kb
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: ha, hc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: ei, pn, en, pr, perc, prr, prd, qr, qd

        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, prcp, pet, ca, cc, kb, ha, hc, qt) &
        !$OMP& private(row, col, ei, pn, en, pr, perc, prr, prd, qr, qd)
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then

                    ei = min(pet(row, col), prcp(row, col))

                    pn = max(0._sp, prcp(row, col) - ei)

                    en = pet(row, col) - ei

                    call gr_production(pn, en, ca(row, col), 9._sp/4._sp, ha(row, col), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if

                prr = 0.9_sp*(pr + perc)
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(4._sp, prcp(row, col), prr, cc(row, col), hc(row, col), qr)

                qd = max(0._sp, prd)

                qt(row, col) = kb(row, col)*(qr + qd)

                ! Transform from mm/dt to m3/s
                qt(row, col) = qt(row, col)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
        !$OMP end parallel do

    end subroutine loieau_timestep

end module md_gr_operator
