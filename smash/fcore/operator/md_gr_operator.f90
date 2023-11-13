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
!%      - grd_timestep
!%      - loieau_timestep

module md_gr_operator

    use md_constant !% only : sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

contains

    !% TODO comment
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

    subroutine gr4_timestep(setup, mesh, prcp, pet, ci, cp, ct, kexc, hi, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: ci, cp, ct, kexc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: hi, hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd

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

            end do
        end do

        ! Transform from mm/dt to m3/s
        qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

    end subroutine gr4_timestep

    subroutine gr5_timestep(setup, mesh, prcp, pet, ci, cp, ct, kexc, aexc, hi, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: ci, cp, ct, kexc, aexc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: hi, hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: pn, en, pr, perc, l, prr, prd, qr, qd

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

            end do
        end do

        ! Transform from mm/dt to m3/s
        qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

    end subroutine gr5_timestep

    subroutine grd_timestep(setup, mesh, prcp, pet, cp, ct, hp, ht, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: cp, ct
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: hp, ht
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: ei, pn, en, pr, perc, l, prr, qr

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

            end do
        end do

        ! Transform from mm/dt to m3/s
        qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

    end subroutine grd_timestep

    subroutine loieau_timestep(setup, mesh, prcp, pet, ca, cc, kb, ha, hc, qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in):: ca, cc, kb
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout):: ha, hc
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: qt

        integer :: row, col
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd

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

            end do
        end do

        ! Transform from mm/dt to m3/s
        qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

    end subroutine loieau_timestep

end module md_gr_operator
