module md_vic3l_operator

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_options !% only: OptionsDT
    use mwd_atmos_manipulation !% get_ac_atmos_data_time_step

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

        hcl = hcl + (prcp - ec - pn)/ccl

        ! % Avoid under and overflow
        hcl = min(0.999999_sp, hcl)
        hcl = max(1e-6_sp, hcl)

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
                pe_value = pe_value + b*(ratio**i)/(b + i)
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

        ! % Avoid under and overflow
        husl = min(0.999999_sp, husl)
        husl = max(1e-6_sp, husl)
        hmsl = min(0.999999_sp, hmsl)
        hmsl = max(1e-6_sp, hmsl)

        qr = pn - (ifl_usl + ifl_msl)

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

    subroutine vic3l_time_step(setup, mesh, input_data, options, time_step, ac_mlt, ac_b, ac_cusl, ac_cmsl, &
    & ac_cbsl, ac_ks, ac_pbc, ac_dsm, ac_ds, ac_ws, ac_hcl, ac_husl, ac_hmsl, ac_hbsl, ac_qt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OptionsDT), intent(in) :: options
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac), intent(in) :: ac_mlt
        real(sp), dimension(mesh%nac), intent(in) :: ac_b, ac_cusl, ac_cmsl, ac_cbsl, ac_ks, &
        & ac_pbc, ac_ds, ac_dsm, ac_ws
        real(sp), dimension(mesh%nac), intent(inout) :: ac_hcl, ac_husl, ac_hmsl, ac_hbsl
        real(sp), dimension(mesh%nac), intent(inout) :: ac_qt

        real(sp), dimension(mesh%nac) :: ac_prcp, ac_pet
        integer :: row, col, k
        real(sp) :: pn, en, qr, qb

        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "prcp", ac_prcp)
        call get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, "pet", ac_pet)

        ac_prcp = ac_prcp + ac_mlt
#ifdef _OPENMP
        !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
        !$OMP& shared(setup, mesh, ac_prcp, ac_pet, ac_b, ac_cusl, ac_cmsl, ac_cbsl, ac_ks, ac_pbc, ac_ds, &
        !$OMP& ac_dsm, ac_ws, ac_hcl, ac_husl, ac_hmsl, ac_hbsl, ac_qt) &
        !$OMP& private(row, col, k, pn, en, qr, qb)
#endif
        do col = 1, mesh%ncol
            do row = 1, mesh%nrow

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle

                k = mesh%rowcol_to_ind_ac(row, col)

                if (ac_prcp(k) .ge. 0._sp .and. ac_pet(k) .ge. 0._sp) then

                    ! Canopy maximum capacity is (0.2 * LAI). Here we fix maximum capacity to 1 mm
                    call vic3l_canopy_interception(ac_prcp(k), ac_pet(k), 1._sp, ac_hcl(k), pn, en)

                    call vic3l_upper_soil_layer_evaporation(en, ac_b(k), ac_cusl(k), ac_husl(k))

                    call vic3l_infiltration(pn, ac_b(k), ac_cusl(k), ac_cmsl(k), ac_husl(k), &
                    & ac_hmsl(k), qr)

                    call vic3l_drainage(ac_cusl(k), ac_cmsl(k), ac_cbsl(k), ac_ks(k), ac_pbc(k), &
                    & ac_husl(k), ac_hmsl(k), ac_hbsl(k))

                else

                    qr = 0._sp

                end if

                call vic3l_baseflow(ac_cbsl(k), ac_ds(k), ac_dsm(k), ac_ws(k), ac_hbsl(k), qb)

                ac_qt(k) = qr + qb

                ! Transform from mm/dt to m3/s
                ac_qt(k) = ac_qt(k)*1e-3_sp*mesh%dx(row, col)*mesh%dy(row, col)/setup%dt

            end do
        end do
#ifdef _OPENMP
        !$OMP end parallel do
#endif
    end subroutine vic3l_time_step

end module md_vic3l_operator
