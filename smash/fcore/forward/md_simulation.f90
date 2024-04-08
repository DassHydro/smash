!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - swap_discharge_buffer
!%      - get_atmos_data_timstep
!%      - get_extended_atmos_data_timestep
!%      - store_timestep
!%      - simulation

module md_simulation

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use mwd_sparse_matrix_manipulation !% only: sparse_matrix_to_matrix
    use md_snow_operator !% only: ssn_timestep
    use md_gr_operator !% only: gr4_timestep, gr5_timestep, grd_timestep, loieau_timestep
    use md_vic3l_operator !% only: vic3l_timestep
    use md_routing_operator !% only: lag0_timestep, lr_timestep, kw_timestep
    use mwd_parameters_manipulation !% only: get_rr_parameters, get_rr_states, set_rr_states
    use md_stats !% only: quicksort
    implicit none

contains

    subroutine swap_discharge_buffer(qt, q)

        implicit none

        real(sp), dimension(:, :, :), intent(inout) :: qt, q

        integer :: i, z

        z = size(q, 3)

        do i = 1, z - 1
            qt(:, :, i) = qt(:, :, i + 1)
            q(:, :, i) = q(:, :, i + 1)
        end do

        qt(:, :, z) = 0._sp
        q(:, :, z) = 0._sp

    end subroutine swap_discharge_buffer

    subroutine get_atmos_data_timestep(setup, mesh, input_data, t, prcp, pet)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, intent(in) :: t
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: prcp, pet

        if (setup%sparse_storage) then

            call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
            call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

        else

            prcp = input_data%atmos_data%prcp(:, :, t)
            pet = input_data%atmos_data%pet(:, :, t)

        end if

    end subroutine get_atmos_data_timestep

    subroutine get_extended_atmos_data_timestep(setup, mesh, input_data, t, snow, temp)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, intent(in) :: t
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: snow, temp

        if (setup%sparse_storage) then

            call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_snow(t), snow)
            call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_temp(t), temp)

        else

            snow = input_data%atmos_data%snow(:, :, t)
            temp = input_data%atmos_data%temp(:, :, t)

        end if

    end subroutine get_extended_atmos_data_timestep

    subroutine store_timestep(mesh, output, returns, t, iret, qt, q)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(OutputDT), intent(inout) :: output
        type(ReturnsDT), intent(inout) :: returns
        integer, intent(in) :: t
        integer, intent(inout) :: iret
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: qt
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: q

        integer :: i

        do i = 1, mesh%ng

            output%response%q(i, t) = q(mesh%gauge_pos(i, 1), mesh%gauge_pos(i, 2))
            
        end do

        !$AD start-exclude
        if (allocated(returns%mask_time_step)) then
            if (returns%mask_time_step(t)) then
                iret = iret + 1
                if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                if (returns%qt_flag) returns%qt(:, :, iret) = qt
            end if
        end if
        !$AD end-exclude

    end subroutine store_timestep
    
    subroutine compute_fluxes_stats(mesh, t, idx, returns)
        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: t, idx
        type(ReturnsDT), intent(inout) :: returns
        real(sp), dimension(mesh%nrow, mesh%ncol) :: fx
        real(sp), dimension(:), allocatable :: fx_flat
        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        integer :: j, npos_val
        real(sp) :: m

        fx = returns%stats%internal_fluxes(:, :, idx)
        !$AD start-exclude
        do j = 1, mesh%ng
            
            if (returns%stats%fluxes_keys(idx) .eq. 'kexc') then
                mask = mesh%mask_gauge(:, :, j)
            else
                mask = (fx .ge. 0._sp .and. mesh%mask_gauge(:, :, j))
            end if
            
            npos_val = count(mask) 
            m = sum(fx, mask = mask) / npos_val 
            returns%stats%fluxes_values(j, t, 1, idx) = m
            returns%stats%fluxes_values(j, t, 2, idx) = sum((fx - m) * (fx - m), mask = mask) / npos_val 
            returns%stats%fluxes_values(j, t, 3, idx) = minval(fx, mask = mask)
            returns%stats%fluxes_values(j, t, 4, idx) = maxval(fx, mask = mask)
                        
            if (.not. allocated(fx_flat)) allocate (fx_flat(npos_val)) 
            fx_flat = pack(fx, mask .eqv. .True.)

            call quicksort(fx_flat)

            if (mod(npos_val, 2) .ne. 0) then
                returns%stats%fluxes_values(j, t, 5, idx) = fx_flat(npos_val / 2 + 1)
            else
                returns%stats%fluxes_values(j, t, 5, idx) = (fx_flat(npos_val / 2) + fx_flat(npos_val / 2 + 1)) / 2
            end if
            
        end do
        !$AD end-exclude
    end subroutine
    
    subroutine compute_states_stats(mesh, output, t, idx, returns)
        implicit none

        type(MeshDT), intent(in) :: mesh
        type(OutputDT), intent(in) :: output
        integer, intent(in) :: t, idx
        type(ReturnsDT), intent(inout) :: returns
        real(sp), dimension(mesh%nrow, mesh%ncol) :: h
        real(sp), dimension(:), allocatable :: h_flat
        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        integer :: j, npos_val
        real(sp) :: m
        
        h = output%rr_final_states%values(:, :, idx)
        !$AD start-exclude
        do j = 1, mesh%ng
            
            mask = (h .ge. 0._sp .and. mesh%mask_gauge(:, :, j))
            
            npos_val = count(mask) 
            m = sum(h, mask = mask) / npos_val 
            returns%stats%rr_states_values(j, t, 1, idx) = m
            returns%stats%rr_states_values(j, t, 2, idx) = sum((h - m) * (h - m), mask = mask) / npos_val 
            returns%stats%rr_states_values(j, t, 3, idx) = minval(h, mask = mask)
            returns%stats%rr_states_values(j, t, 4, idx) = maxval(h, mask = mask)
            
            if (.not. allocated(h_flat)) allocate (h_flat(npos_val)) 
            h_flat = pack(h, mask .eqv. .True.)

            call quicksort(h_flat)

            if (mod(npos_val, 2) .ne. 0) then
                returns%stats%rr_states_values(j, t, 5, idx) = h_flat(npos_val / 2 + 1)
            else
                returns%stats%rr_states_values(j, t, 5, idx) = (h_flat(npos_val / 2) + h_flat(npos_val / 2 + 1)) / 2
            end if
            
        end do
        !$AD end-exclude
    end subroutine
    
    subroutine simulation(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns
        
        integer :: t, iret, zq, i
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet
        real(sp), dimension(:, :, :), allocatable :: q, qt
        real(sp), dimension(:, :), allocatable :: mlt
        real(sp), dimension(:, :), allocatable :: snow, temp
        real(sp), dimension(:, :), allocatable :: kmlt, ci, cp, ct, kexc, aexc, ca, cc, kb, &
        & b, cusl, cmsl, cbsl, ks, pbc, ds, dsm, ws, llr, akw, bkw
        real(sp), dimension(:, :), allocatable :: hs, hi, hp, ht, ha, hc, hcl, husl, hmsl, hbsl, hlr
        real(sp), dimension(mesh%ng, setup%ntime_step) :: mean, var, minimum, maximum, med

        ! Snow module initialisation
        select case (setup%snow_module)

            ! 'zero' snow module
        case ("zero")
            ! Nothing to do

            ! 'ssn' snow module
        case ("ssn")

            ! Snow related atmospheric data ; snow and temp
            allocate (snow(mesh%nrow, mesh%ncol), temp(mesh%nrow, mesh%ncol))

            ! Melt grid
            allocate (mlt(mesh%nrow, mesh%ncol))

            ! Snow module rr parameters
            allocate (kmlt(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "kmlt", kmlt)

            ! Snow module rr states
            allocate (hs(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hs", hs)

        end select

        ! Hydrological module initialisation
        select case (setup%hydrological_module)

            ! 'gr4' module
        case ("gr4")

            ! Hydrological module rr parameters
            allocate (ci(mesh%nrow, mesh%ncol), cp(mesh%nrow, mesh%ncol), ct(mesh%nrow, mesh%ncol), kexc(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "ci", ci)
            call get_rr_parameters(parameters%rr_parameters, "cp", cp)
            call get_rr_parameters(parameters%rr_parameters, "ct", ct)
            call get_rr_parameters(parameters%rr_parameters, "kexc", kexc)

            ! Hydrological module rr states
            allocate (hi(mesh%nrow, mesh%ncol), hp(mesh%nrow, mesh%ncol), ht(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hi", hi)
            call get_rr_states(parameters%rr_initial_states, "hp", hp)
            call get_rr_states(parameters%rr_initial_states, "ht", ht)

            ! 'gr5' module
        case ("gr5")

            ! Hydrological module rr parameters
            allocate (ci(mesh%nrow, mesh%ncol), cp(mesh%nrow, mesh%ncol), ct(mesh%nrow, mesh%ncol), kexc(mesh%nrow, mesh%ncol), &
                      aexc(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "ci", ci)
            call get_rr_parameters(parameters%rr_parameters, "cp", cp)
            call get_rr_parameters(parameters%rr_parameters, "ct", ct)
            call get_rr_parameters(parameters%rr_parameters, "kexc", kexc)
            call get_rr_parameters(parameters%rr_parameters, "aexc", aexc)

            ! Hydrological module rr states
            allocate (hi(mesh%nrow, mesh%ncol), hp(mesh%nrow, mesh%ncol), ht(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hi", hi)
            call get_rr_states(parameters%rr_initial_states, "hp", hp)
            call get_rr_states(parameters%rr_initial_states, "ht", ht)

            ! 'grd' module
        case ("grd")

            ! Hydrological module rr parameters
            allocate (cp(mesh%nrow, mesh%ncol), ct(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "cp", cp)
            call get_rr_parameters(parameters%rr_parameters, "ct", ct)

            ! Hydrological module rr states
            allocate (hp(mesh%nrow, mesh%ncol), ht(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hp", hp)
            call get_rr_states(parameters%rr_initial_states, "ht", ht)

            ! 'loieau' module
        case ("loieau")

            ! Hydrological module rr parameters
            allocate (ca(mesh%nrow, mesh%ncol), cc(mesh%nrow, mesh%ncol), kb(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "ca", ca)
            call get_rr_parameters(parameters%rr_parameters, "cc", cc)
            call get_rr_parameters(parameters%rr_parameters, "kb", kb)

            ! Hydrological module rr states
            allocate (ha(mesh%nrow, mesh%ncol), hc(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "ha", ha)
            call get_rr_states(parameters%rr_initial_states, "hc", hc)

            ! 'vic3l' module
        case ("vic3l")

            ! Hydrological module rr parameters
            allocate (b(mesh%nrow, mesh%ncol), cusl(mesh%nrow, mesh%ncol), cmsl(mesh%nrow, mesh%ncol), cbsl(mesh%nrow, mesh%ncol), &
            ks(mesh%nrow, mesh%ncol), pbc(mesh%nrow, mesh%ncol), ds(mesh%nrow, mesh%ncol), dsm(mesh%nrow, mesh%ncol), &
            & ws(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "b", b)
            call get_rr_parameters(parameters%rr_parameters, "cusl", cusl)
            call get_rr_parameters(parameters%rr_parameters, "cmsl", cmsl)
            call get_rr_parameters(parameters%rr_parameters, "cbsl", cbsl)
            call get_rr_parameters(parameters%rr_parameters, "ks", ks)
            call get_rr_parameters(parameters%rr_parameters, "pbc", pbc)
            call get_rr_parameters(parameters%rr_parameters, "ds", ds)
            call get_rr_parameters(parameters%rr_parameters, "dsm", dsm)
            call get_rr_parameters(parameters%rr_parameters, "ws", ws)

            ! Hydrological module rr states
            allocate (hcl(mesh%nrow, mesh%ncol), husl(mesh%nrow, mesh%ncol), hmsl(mesh%nrow, mesh%ncol), hbsl(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hcl", hcl)
            call get_rr_states(parameters%rr_initial_states, "husl", husl)
            call get_rr_states(parameters%rr_initial_states, "hmsl", hmsl)
            call get_rr_states(parameters%rr_initial_states, "hbsl", hbsl)

        end select

        ! Routing module initialisation
        select case (setup%routing_module)

            ! 'lag0' module
        case ("lag0")

            ! Discharge buffer depth
            zq = 1

            ! 'lr' module
        case ("lr")

            ! Discharge buffer depth
            zq = 1

            ! Routing module rr parameters
            allocate (llr(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "llr", llr)

            ! Routing module rr states
            allocate (hlr(mesh%nrow, mesh%ncol))
            call get_rr_states(parameters%rr_initial_states, "hlr", hlr)

            ! 'kw' module
        case ("kw")

            ! Discharge buffer depth
            zq = 2

            ! Routing module rr parameters
            allocate (akw(mesh%nrow, mesh%ncol), bkw(mesh%nrow, mesh%ncol))
            call get_rr_parameters(parameters%rr_parameters, "akw", akw)
            call get_rr_parameters(parameters%rr_parameters, "bkw", bkw)

        end select

        ! Discharge grids
        allocate (qt(mesh%nrow, mesh%ncol, zq), q(mesh%nrow, mesh%ncol, zq))

        iret = 0
        qt = 0._sp
        q = 0._sp
 
        ! Start time loop
        do t = 1, setup%ntime_step

            ! Swap discharge buffer
            call swap_discharge_buffer(qt, q)

            ! Get atmospheric data ; prcp and pet
            call get_atmos_data_timestep(setup, mesh, input_data, t, prcp, pet)

            ! Snow module
            select case (setup%snow_module)

            case ("zero")

                ! Nothing to do

            case ("ssn")

                ! Get extended atmospheric data ; snow and temp
                call get_extended_atmos_data_timestep(setup, mesh, input_data, t, snow, temp)
                call ssn_timestep(setup, mesh, options, snow, temp, kmlt, hs, mlt)

                prcp = prcp + mlt

            end select

            ! Hydrological module
            select case (setup%hydrological_module)

                ! 'gr4' module
            case ("gr4")
                call gr4_timestep(setup, mesh, options, prcp, pet, ci, cp, ct, kexc, hi, hp, ht, qt(:, :, zq), returns)

                call set_rr_states(output%rr_final_states, "hi", hi)
                call set_rr_states(output%rr_final_states, "hp", hp)
                call set_rr_states(output%rr_final_states, "ht", ht)

                ! 'gr5' module
            case ("gr5")

                call gr5_timestep(setup, mesh, options, prcp, pet, ci, cp, ct, kexc, aexc, hi, hp, ht, qt(:, :, zq), returns)

                call set_rr_states(output%rr_final_states, "hi", hi)
                call set_rr_states(output%rr_final_states, "hp", hp)
                call set_rr_states(output%rr_final_states, "ht", ht)
                
                ! 'grd' module
            case ("grd")

                call grd_timestep(setup, mesh, options, prcp, pet, cp, ct, hp, ht, qt(:, :, zq), returns)

                call set_rr_states(output%rr_final_states, "hp", hp)
                call set_rr_states(output%rr_final_states, "ht", ht)
                
                ! 'loieau' module
            case ("loieau")

                call loieau_timestep(setup, mesh, options, prcp, pet, ca, cc, kb, ha, hc, qt(:, :, zq), returns)

                call set_rr_states(output%rr_final_states, "ha", ha)
                call set_rr_states(output%rr_final_states, "hc", hc)
                
                ! 'vic3l' module
            case ("vic3l")

                call vic3l_timestep(setup, mesh, options, prcp, pet, b, cusl, cmsl, cbsl, ks, pbc, ds, dsm, ws, &
                & hcl, husl, hmsl, hbsl, qt(:, :, zq), returns)

                call set_rr_states(output%rr_final_states, "hcl", hcl)
                call set_rr_states(output%rr_final_states, "husl", husl)
                call set_rr_states(output%rr_final_states, "hmsl", hmsl)
                call set_rr_states(output%rr_final_states, "hbsl", hbsl)
                
            end select
            

            ! Routing module
            select case (setup%routing_module)

                ! 'lag0' module
            case ("lag0")

                call lag0_timestep(setup, mesh, options, qt, q)
                
                ! 'lr' module
            case ("lr")

                call lr_timestep(setup, mesh, options, qt, llr, hlr, q)
                
                call set_rr_states(output%rr_final_states, "hlr", hlr)
                
                ! 'kw' module
            case ("kw")

                call kw_timestep(setup, mesh, options, qt, akw, bkw, q)

            end select
        
            ! Store variables
            if (returns%stats_flag) then
                do i = 1, setup%nfx
                    call compute_fluxes_stats(mesh, t, i, returns)
                end do
                
                do i = 1, setup%nrrs
                    call compute_states_stats(mesh, output, t, i, returns)
                end do
            end if
            
            call store_timestep(mesh, output, returns, t, iret, qt(:, :, zq), q(:, :, zq))

        end do

    end subroutine simulation

end module md_simulation
