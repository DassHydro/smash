module md_vic3l_operator

    use md_constant !% only: sp

    implicit none

contains

    subroutine vic3l_canopy_interception(prcp, pet, ccl, hcl, pn, en)

        implicit none

        real(sp), intent(in) :: prcp, pet, ccl
        real(sp), intent(inout) :: hcl
        real(sp), intent(out) :: pn, en

        real(sp) :: ec

        ec = min(pet*(hcl**(2._sp/3._sp)), prcp + hcl*ccl)

        pn = max(0._sp, prcp - ccl*(1._sp - hcl) - ec)

        en = pet - ec

        hcl = min(0.999999_sp, hcl + (prcp - ec - pn)/ccl)

    end subroutine vic3l_canopy_interception

    ! https://github.com/UW-Hydro/VIC/blob/master/vic/vic_run/src/arno_evap.c
    subroutine vic3l_upper_soil_layer_evaporation(en, b, cusl, husl)

        implicit none

        integer, parameter :: npe = 5

        real(sp), intent(in) :: en, b, cusl
        real(sp), intent(inout) :: husl

        integer :: i
        real(sp) :: iflm, ifl0, ratio, as, pe_value, beta, es

        iflm = (1._sp + b)*cusl

        ifl0 = iflm*(1._sp - (1._sp - husl)**(1._sp/(1._sp + b)))

        if (ifl0 .ge. iflm) then
            es = en

        else
            ratio = (1._sp - ifl0/iflm)
            as = 1._sp - ratio**b

            pe_value = 1._sp
            do i = 1, npe
                pe_value = pe_value + b*ratio**i/(b + i)
            end do

            beta = as + (1._sp - as)*(1._sp - ratio)*pe_value
            es = en*beta

        end if

        es = min(es, husl*cusl)
        husl = husl - es/cusl

    end subroutine vic3l_upper_soil_layer_evaporation

    subroutine vic3l_infiltration(pn, b, cusl, cmsl, husl, hmsl, qr)

        implicit none

        real(sp), intent(in) :: pn, b, cusl, cmsl
        real(sp), intent(inout) :: husl, hmsl
        real(sp), intent(out) :: qr

        real(sp) :: cumsl, wumsl, humsl, iflm, ifl0, ifl, ifl_usl, ifl_msl

        cumsl = cusl + cmsl
        wumsl = husl*cusl + hmsl*cmsl
        humsl = wumsl/cumsl

        iflm = (1._sp + b)*cumsl
        ifl0 = iflm*(1._sp - (1._sp - humsl)**(1._sp/(1._sp + b)))

        if (ifl0 + pn .gt. iflm) then
            ifl = cumsl - wumsl
        else
            ifl = cumsl - wumsl - cumsl*(1._sp - (ifl0 + pn)/iflm)**(b + 1._sp)

        end if

        ifl = min(pn, ifl)

        ifl_usl = min((1._sp - husl)*cusl, ifl)
        ifl_msl = min((1._sp - hmsl)*cmsl, (ifl - ifl_usl))

        husl = husl + ifl_usl/cusl
        hmsl = hmsl + ifl_msl/cmsl

        qr = pn - (ifl_usl - ifl_msl)

    end subroutine vic3l_infiltration

    subroutine vic3l_drainage_2l(cu, cl, ks, pbc, hu, hl)

        implicit none

        real(sp), intent(in) :: cu, cl, ks, pbc
        real(sp), intent(inout) :: hu, hl

        real(sp) :: hpbc, d, wu, wl

        ! Gradient issue when pbc is too high (i.e. ]0, 1[ ^ 30)
        hpbc = max(1e-6_sp, hu**pbc)
        d = ks*hpbc
        wu = hu*cu
        wl = hl*cl

        d = min(d, min(wu, cl - wl))

        hu = hu - d/cu
        hl = hl + d/cl

    end subroutine vic3l_drainage_2l

    subroutine vic3l_drainage(cusl, cmsl, cbsl, ks, pbc, husl, hmsl, hbsl)

        implicit none

        real(sp), intent(in) :: cusl, cmsl, cbsl, ks, pbc
        real(sp), intent(inout) :: husl, hmsl, hbsl

        call vic3l_drainage_2l(cusl, cmsl, ks, pbc, husl, hmsl)

        call vic3l_drainage_2l(cmsl, cbsl, ks, pbc, hmsl, hbsl)

    end subroutine vic3l_drainage

    subroutine vic3l_baseflow(cbsl, ds, dsm, ws, hbsl, qb)

        implicit none

        real(sp), intent(in) :: cbsl, ds, dsm, ws
        real(sp), intent(inout) :: hbsl
        real(sp), intent(out) :: qb

        if (hbsl .gt. ws) then
            qb = (dsm*ds/ws)*hbsl + dsm*(1._sp - ds/ws)*((hbsl - ws)/(1._sp - ws))**2._sp

        else
            qb = (dsm*ds/ws)*hbsl

        end if

        qb = min(hbsl*cbsl, qb)

        hbsl = hbsl - qb/cbsl

    end subroutine vic3l_baseflow

end module md_vic3l_operator