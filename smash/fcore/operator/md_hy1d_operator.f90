!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - hy1d_non_inertial_time_step

module md_hy1d_operator

    use md_constant !% only : sp, gravity
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_cross_section !% only: Cross_SectionsDT
    use mwd_segment !% only: SegmentDT

    implicit none
    !character(len=256) :: mass_balance_file

contains

    subroutine hy1d_non_inertial_get_dt(setup, mesh, hy1d_h, dt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%ncs), intent(in) :: hy1d_h
        real(sp), intent(out) :: dt

        integer :: ics
        real(sp) :: dt_min, dx, h_cs, alpha

        ! Initialize dt with hydrological time step
        dt_min = setup%dt 
        alpha = mesh%alpha

        ! Loop over all cross-sections
        do ics = 1, mesh%ncs

            ! Skip outlet --> no downstream constraint
            if (mesh%cross_sections(ics)%is_outlet) cycle

            dx = mesh%cross_sections(ics)%dx
            h_cs = hy1d_h(ics)

            if (h_cs <= 0.01_sp) cycle

            ! CFL condition
            dt_min = min(dt_min, alpha * dx / sqrt(gravity * h_cs))

        end do

        dt = dt_min

    end subroutine hy1d_non_inertial_get_dt

    subroutine hy1d_non_inertial_momemtum(mesh, dt, hy1d_h, hy1d_q, time_step, hy1d_step) ! added time_step and hy1d_step

        implicit none

        type(MeshDT), intent(in) :: mesh
        real(sp), intent(in) :: dt
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h, hy1d_q
        integer, intent(in) :: time_step, hy1d_step

        integer :: ics, ids_cs
        real(sp) :: h_i, h_ip1, b_i, b_ip1, dx, manning_i
        real(sp) :: s_ip1d2, h_ip1d2
        real(sp) :: w_ip1d2, a_ip1d2, p_ip1d2, r_ip1d2
        real(sp) :: u_ip1d2, froude

        ! Loop over all cross-sections
        do ics = 1, mesh%ncs

            ids_cs = mesh%cross_sections(ics)%ids_cs

            dx = mesh%cross_sections(ics)%dx
            b_i = mesh%cross_sections(ics)%bathy
            manning_i = mesh%cross_sections(ics)%manning(1)
            h_i = hy1d_h(ics)

            if (mesh%cross_sections(ics)%is_outlet) then ! outlet boundary condition
                b_ip1 = mesh%cross_sections(ics)%bathy_bc
                h_ip1 = 0._sp
            else
                b_ip1 = mesh%cross_sections(ids_cs)%bathy
                h_ip1 = hy1d_h(ids_cs)
            end if

            s_ip1d2 = ((h_i + b_i) - (h_ip1 + b_ip1)) / dx

            h_ip1d2 = max(h_i + b_i, h_ip1 + b_ip1) - max(b_i, b_ip1)
            
            w_ip1d2 = min(mesh%cross_sections(ics)%level_widths(1), &
                                mesh%cross_sections(ids_cs)%level_widths(1))

            a_ip1d2 = h_ip1d2 * w_ip1d2
            p_ip1d2 = w_ip1d2 + 2._sp * h_ip1d2
            r_ip1d2 = a_ip1d2 / p_ip1d2

            if (h_ip1d2 <= 0._sp) then ! if (h_ip1d2 .le. 0._sp)
                hy1d_q(ics) = 0._sp
            else
                hy1d_q(ics) = (hy1d_q(ics) - gravity*dt*a_ip1d2*(-s_ip1d2)) / &
                            (1._sp + gravity*dt * &
                            ((manning_i**2 * abs(hy1d_q(ics))) / &
                            (a_ip1d2 * r_ip1d2**(4._sp/3._sp))))
            end if

            if (h_ip1d2 > 0._sp .and. a_ip1d2 > 0._sp) then
                u_ip1d2 = hy1d_q(ics) / a_ip1d2
                froude = abs(u_ip1d2) / sqrt(gravity * h_ip1d2)
                if (froude > 1._sp) then
                    hy1d_q(ics) = sign(1._sp, hy1d_q(ics)) * &
                                a_ip1d2 * sqrt(gravity * h_ip1d2)
                end if
            end if
        end do

    end subroutine hy1d_non_inertial_momemtum

    subroutine hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q, time_step, hy1d_step)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT),  intent(in) :: mesh
        real(sp), intent(in) :: dt
        real(sp), dimension(mesh%nac, setup%nqz), intent(in) :: ac_qtz, ac_qz
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h, hy1d_q
        integer, intent(in) :: time_step, hy1d_step ! added time_step and hy1d_step


        integer :: ics, i, k, ids_cs, nus_cs_i, nup_i, nlat_i
        real(sp) :: a_t, a_tp1
        real(sp) :: q_im1d2, q_ip1d2, q_lat
        real(sp) :: dx, width_i
        !real(sp) :: q_im1d2_mb ! hydrological upstream inflows for mass balance calculation

        ! Varibales for mass balance calculation (TEMPORARY)
        !real(sp) :: Vin_step, Vout_step, Vt_step, Vtp1_step, Et_step
        !Vin_step  = 0._sp
        !Vout_step = 0._sp
        !Vt_step   = 0._sp
        !Vtp1_step = 0._sp
        
        ! Loop over all cross-sections 
        do ics = 1, mesh%ncs
            ids_cs = mesh%cross_sections(ics)%ids_cs
            dx = mesh%cross_sections(ics)%dx
            nus_cs_i = mesh%cross_sections(ics)%nus_cs
            nup_i = mesh%cross_sections(ics)%nup
            nlat_i = mesh%cross_sections(ics)%nlat
            width_i = mesh%cross_sections(ics)%level_widths(1)

            ! ------------------------------------------
            ! Upstream discharge Q_{i-1/2}
            ! ------------------------------------------
            q_im1d2 = 0._sp
            if (nus_cs_i == 0) then
                ! Headwater: inflow from hydrological coupling
                do i = 1, nup_i
                    k = mesh%rowcol_to_ind_ac(mesh%cross_sections(ics)%up_rowcols(i,1), &
                                               mesh%cross_sections(ics)%up_rowcols(i,2))
                    q_im1d2 = q_im1d2 + ac_qz(k, setup%nqz)
                end do
            else if (nus_cs_i > 1) then
                ! Confluence: sum upstream discharges
                do i = 1, nus_cs_i
                    q_im1d2 = q_im1d2 + hy1d_q(mesh%cross_sections(ics)%ius_cs(i))
                end do
            else
                q_im1d2 = hy1d_q(mesh%cross_sections(ics)%ius_cs(1))
            end if

            ! ------------------------------------------
            ! Lateral inflow
            ! ------------------------------------------
            q_lat = 0._sp

            ! Net rainfall applied only to cross-sections linked to actual river cells (skip if CS is intermediate, rowcol=-99, see preprocessing)
            if (mesh%cross_sections(ics)%rowcol(1) /= -99) then
                k = mesh%rowcol_to_ind_ac(mesh%cross_sections(ics)%rowcol(1), &
                                           mesh%cross_sections(ics)%rowcol(2))
                q_lat = ac_qtz(k, setup%nqz)
            end if

            do i = 1, nlat_i  ! lateral hydrological inflows
                k = mesh%rowcol_to_ind_ac(mesh%cross_sections(ics)%lat_rowcols(i,1), &
                                           mesh%cross_sections(ics)%lat_rowcols(i,2))
                q_lat = q_lat + ac_qz(k, setup%nqz)
            end do

            ! ------------------------------------------
            ! Downstream discharge Q_{i+1/2}
            ! ------------------------------------------
            q_ip1d2 = hy1d_q(ics)

            ! ------------------------------------------
            ! Update cross-section area
            ! ------------------------------------------
            a_t   = width_i * hy1d_h(ics)
            a_tp1 = a_t + dt * (q_lat + q_im1d2 - q_ip1d2) / dx

            if (a_tp1 < 0._sp) then
                a_tp1 = 0._sp
                hy1d_q(ics) = q_lat + q_im1d2 + a_t / (dt/dx)
            end if

            hy1d_h(ics) = a_tp1 / width_i

            ! ------------------------------------------
            ! Mass balance analysis
            ! ------------------------------------------
            !q_im1d2_mb = 0._sp
            !if (nus_cs_i == 0) then
                !do i = 1, nup_i
                    !k = mesh%rowcol_to_ind_ac(mesh%cross_sections(ics)%up_rowcols(i,1), &
                                            !mesh%cross_sections(ics)%up_rowcols(i,2))
                    !q_im1d2_mb = q_im1d2_mb + ac_qz(k, setup%nqz)
                !end do
            !end if

            !Vin_step  = Vin_step  + (q_lat + q_im1d2_mb) * dt
            !if (mesh%cross_sections(ics)%is_outlet) Vout_step = Vout_step + q_ip1d2 * dt

            !Vt_step   = Vt_step   + a_t   * dx
            !Vtp1_step = Vtp1_step + a_tp1 * dx

        end do ! ics

        ! --- Write aggregated global mass balance per step (Et only) ---
        !if (time_step == 1 .and. hy1d_step == 1) then
            !mass_balance_file = "/home/adminberkaoui/Documents/article_garonne_latest/" // &
                                !"HH_NEW_SOLVER_LATEST/HH_SIMULATIONS/hy1d_mb.csv"
            !mass_balance_file = "/home/adminberkaoui/Bureau/xs_topology_smash/hy1d_mb.csv"
            !open(unit=10, file=mass_balance_file, status="replace")
            !write(10,'(A)') "Et_step"
        !end if
        !if (Vtp1_step /= 0.0_sp) then
            !Et_step = ((Vin_step - Vout_step) - (Vtp1_step - Vt_step)) / Vtp1_step
        !else
            !Et_step = -999.0_sp   ! flag for undefined relative error
        !end if

        ! Write only Et
        !write(10,'(G0.16)') Et_step
        ! ===== END MASS BALANCE ANALYSIS =====

    end subroutine hy1d_non_inertial_mass

    subroutine hy1d_non_inertial_time_step(setup, mesh, time_step, ac_qtz, ac_qz, hy1d_h, hy1d_q)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: time_step
        real(sp), dimension(mesh%nac, setup%nqz), intent(in) :: ac_qtz, ac_qz
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h, hy1d_q

        real(sp) :: dt
        integer :: ntime_step, t, ceil

        !integer :: global_step, global_total, last_printed_percent, percent

        call hy1d_non_inertial_get_dt(setup, mesh, hy1d_h, dt) 

        ceil = setup%dt/dt
        if (ceil == int(ceil)) then
            ntime_step = int(ceil)
        else
            ntime_step = int(ceil) + 1
        end if
        
        !global_total = setup%ntime_step * ntime_step
        !last_printed_percent = -1

        do t = 1, ntime_step
            !global_step = (time_step - 1) * ntime_step + t
            call hy1d_non_inertial_momemtum(mesh, dt, hy1d_h, hy1d_q, time_step, t) ! Pass time_step and t
            call hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q,time_step,t) ! Pass t as hy1d_step counter and time_step as hydro_step counter
        end do

        ! ---- Print progress once per hydrological step ----
        !percent = int(100.0 * time_step / setup%ntime_step)
        !if (percent > last_printed_percent) then
        !    print *, "Global Progress: ", percent, "% completed"
        !    last_printed_percent = percent
        !end if
        
        ! Close file after all hydrological time steps are complete
        !if (time_step == setup%ntime_step) then
        !    !close(10)
        !end if

    end subroutine hy1d_non_inertial_time_step

end module md_hy1d_operator
