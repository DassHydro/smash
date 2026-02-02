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
    character(len=256) :: mass_balance_file

contains

    subroutine hy1d_get_downstream_segment(mesh, iseg, seg, ids_seg, ds_seg)

        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: iseg
        type(SegmentDT), intent(in) :: seg
        integer, intent(out) :: ids_seg
        type(SegmentDT), intent(out) :: ds_seg

        ! Same segment if there is no downstream segment
        if (seg%nds_seg .eq. 0) then
            ids_seg = iseg
            ! No bifurcation. Only one downstream segment
        else if (seg%nds_seg .eq. 1) then
            ids_seg = seg%ds_segment(1)
            ! Bifurcation. Should be unreachable
        else
        end if

        ds_seg = mesh%segments(ids_seg)

    end subroutine hy1d_get_downstream_segment

    subroutine hy1d_get_upstream_segments(mesh, iseg, seg, ius_seg, us_seg)

        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: iseg
        type(SegmentDT), intent(in) :: seg
        integer, allocatable, dimension(:), intent(out) :: ius_seg
        type(SegmentDT), allocatable, dimension(:), intent(out) :: us_seg

        integer :: i

        if (allocated(ius_seg)) deallocate (ius_seg)
        if (allocated(us_seg)) deallocate (us_seg)

        ! Same segment if there is no upstream segment
        if (seg%nus_seg .eq. 0) then
            allocate (ius_seg(1))
            allocate (us_seg(1))
            ius_seg(1) = iseg
        else
            allocate (ius_seg(seg%nus_seg))
            allocate (us_seg(seg%nus_seg))
            ius_seg = seg%us_segment
        end if

        do i = 1, size(ius_seg)
            us_seg(i) = mesh%segments(ius_seg(i))
        end do

    end subroutine hy1d_get_upstream_segments

    subroutine hy1d_get_downstream_cross_section(mesh, seg, ds_seg, ics, cs, ids_cs, ds_cs)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(SegmentDT), intent(in) :: seg, ds_seg
        integer, intent(in) :: ics
        type(Cross_SectionDT), intent(in) :: cs
        integer, intent(out) :: ids_cs
        type(Cross_SectionDT), intent(out) :: ds_cs

        ! Last cross section of the segment. The downstream cross section is the first cross section of the
        ! downstream segment
        if (ics .eq. seg%last_cross_section) then
            ids_cs = ds_seg%first_cross_section
            ! Otherwise, downstream cross section is the next cross section
        else
            ids_cs = ics + 1
        end if

        ds_cs = mesh%cross_sections(ids_cs)

    end subroutine hy1d_get_downstream_cross_section

    subroutine hy1d_get_upstream_cross_sections(mesh, seg, us_seg, ics, cs, ius_cs, us_cs)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(SegmentDT), intent(in) :: seg
        type(SegmentDT), allocatable, dimension(:), intent(in) :: us_seg
        integer, intent(in) :: ics
        type(Cross_SectionDT), intent(in) :: cs
        integer, allocatable, dimension(:), intent(out) :: ius_cs
        type(Cross_SectionDT), allocatable, dimension(:), intent(out) :: us_cs

        integer :: i

        if (allocated(ius_cs)) deallocate (ius_cs)
        if (allocated(us_cs)) deallocate (us_cs)

        ! First cross section of the segment. The upstream cross sections are the last cross sections of the
        ! upstream segments
        if (ics .eq. seg%first_cross_section) then
            allocate (ius_cs(size(us_seg)))
            allocate (us_cs(size(us_seg)))
            do i = 1, size(us_seg)
                ius_cs(i) = us_seg(i)%last_cross_section
            end do
        else
            allocate (ius_cs(1))
            allocate (us_cs(1))
            ius_cs(1) = ics - 1
        end if

        do i = 1, size(ius_cs)
            us_cs(i) = mesh%cross_sections(ius_cs(i))
        end do

    end subroutine hy1d_get_upstream_cross_sections

    subroutine hy1d_get_downstream_boundary_condition_xbh(cs, us_cs, x, b, h)

        implicit none

        type(Cross_SectionDT), intent(in) :: cs
        type(Cross_SectionDT), dimension(:), intent(in) :: us_cs
        real(sp), intent(out) :: x, b, h

        integer :: i
        real(sp) :: us_x, us_s

        us_x = 0._sp
        us_s = 0._sp
        do i = 1, size(us_cs)
            us_x = us_x + us_cs(i)%x - cs%x
            us_s = us_s + (us_cs(i)%bathy - cs%bathy)/(us_cs(i)%x - cs%x)
        end do
        us_x = us_x/size(us_cs)
        us_s = us_s/size(us_cs)
        x = cs%x - us_x
        b = cs%bathy + us_s*x
        h = 0._sp

    end subroutine hy1d_get_downstream_boundary_condition_xbh

    subroutine hy1d_non_inertial_get_dt(setup, mesh, hy1d_h, dt)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%ncs), intent(in) :: hy1d_h
        real(sp), intent(out) :: dt
        integer :: iseg, ids_seg, ics, ids_cs, ifcs, ilcs
        real(sp) :: dt_min, dx, h_cs, alpha
        type(SegmentDT) :: seg, ds_seg
        type(Cross_SectionDT) :: cs, ds_cs

        dt_min = setup%dt ! initialize dt_min with hydrological dt
        alpha = mesh%alpha
        do iseg = 1, mesh%nseg
            seg = mesh%segments(iseg)
            if (seg%nds_seg .eq. 0) cycle
            call hy1d_get_downstream_segment(mesh, iseg, seg, ids_seg, ds_seg)
            ifcs = seg%first_cross_section
            ilcs = seg%last_cross_section
            do ics = ifcs, ilcs
                cs = mesh%cross_sections(ics)
                call hy1d_get_downstream_cross_section(mesh, seg, ds_seg, ics, cs, ids_cs, ds_cs)
                dx = cs%x - ds_cs%x
                h_cs = hy1d_h(ics)
                if (h_cs <= 0.01_sp) cycle
                dt_min = min(dt_min, alpha * dx / sqrt(gravity * h_cs))
            end do
        end do

        dt = dt_min

    end subroutine hy1d_non_inertial_get_dt

    subroutine hy1d_non_inertial_momemtum(mesh, dt, hy1d_h, hy1d_q, time_step, hy1d_step) ! added time_step and hy1d_step

        implicit none

        type(MeshDT), intent(in) :: mesh
        real(sp), intent(in) :: dt
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h, hy1d_q
        integer, intent(in) :: time_step, hy1d_step

        integer :: i, iseg, ids_seg, ics, ids_cs, ifcs, ilcs
        logical :: is_last_ds_seg, is_first_us_seg, is_ds_bc
        real(sp) :: h_i, h_ip1, b_i, b_ip1, x_i, x_ip1, dx
        real(sp) :: s_ip1d2, h_ip1d2, w_ip1d2, a_ip1d2, p_ip1d2, r_ip1d2
        real(sp) :: u_ip1d2, froude
        type(SegmentDT) :: seg, ds_seg
        type(Cross_SectionDT) :: cs, ds_cs
        integer, allocatable, dimension(:) :: ius_seg, ius_cs
        type(SegmentDT), allocatable, dimension(:) :: us_seg
        type(Cross_SectionDT), allocatable, dimension(:) :: us_cs

        do iseg = 1, mesh%nseg

            seg = mesh%segments(iseg)
            call hy1d_get_downstream_segment(mesh, iseg, seg, ids_seg, ds_seg)
            call hy1d_get_upstream_segments(mesh, iseg, seg, ius_seg, us_seg)
            is_last_ds_seg = iseg .eq. ids_seg
            is_first_us_seg = iseg .eq. ius_seg(1)
            ifcs = seg%first_cross_section
            ilcs = seg%last_cross_section

            do ics = ifcs, ilcs

                cs = mesh%cross_sections(ics)
                call hy1d_get_downstream_cross_section(mesh, seg, ds_seg, ics, cs, ids_cs, ds_cs)
                call hy1d_get_upstream_cross_sections(mesh, seg, us_seg, ics, cs, ius_cs, us_cs)
                is_ds_bc = ics .eq. ilcs .and. is_last_ds_seg

                x_i = cs%x
                x_ip1 = ds_cs%x
                b_i = cs%bathy
                b_ip1 = ds_cs%bathy
                h_i = hy1d_h(ics)
                h_ip1 = hy1d_h(ids_cs)

                if (is_ds_bc) call hy1d_get_downstream_boundary_condition_xbh(cs, us_cs, x_ip1, b_ip1, h_ip1)

                dx = x_i - x_ip1
                s_ip1d2 = ((h_i + b_i) - (h_ip1 + b_ip1))/dx

                h_ip1d2 = max(h_i + b_i, h_ip1 + b_ip1) - max(b_i, b_ip1)

                w_ip1d2 = min(cs%level_widths(1), ds_cs%level_widths(1))
                a_ip1d2 = h_ip1d2*w_ip1d2
                p_ip1d2 = w_ip1d2 + 2._sp*h_ip1d2
                r_ip1d2 = a_ip1d2/p_ip1d2

                if (h_ip1d2 .le. 0._sp) then
                    hy1d_q(ics) = 0._sp
                else
                    hy1d_q(ics) = (hy1d_q(ics) - gravity*dt*a_ip1d2*-s_ip1d2)/ &
                                  (1._sp + gravity*dt*((cs%manning(1)**2*abs(hy1d_q(ics)))/(a_ip1d2*r_ip1d2**(4._sp/3._sp))))
                end if

                if (h_ip1d2 > 0._sp .and. a_ip1d2 > 0._sp) then
                    u_ip1d2 = hy1d_q(ics) / a_ip1d2
                    froude = abs(u_ip1d2) / sqrt(gravity * h_ip1d2)
                    if (froude > 1._sp) then
                        hy1d_q(ics) = sign(1._sp, hy1d_q(ics)) * a_ip1d2 * sqrt(gravity * h_ip1d2)
                    end if
                end if
            end do
        end do

    end subroutine hy1d_non_inertial_momemtum

    subroutine hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q, time_step, hy1d_step)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), intent(in) :: dt
        real(sp), dimension(mesh%nac, setup%nqz), intent(in) :: ac_qtz, ac_qz
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_q
        integer, intent(in) :: time_step, hy1d_step   ! Added


        integer :: i, k, iseg, ids_seg, ics, ids_cs, ifcs, ilcs
        logical :: is_last_ds_seg, is_first_us_seg, is_ds_bc
        real(sp) :: a_t, a_tp1, q_ip1d2, q_im1d2, q_lat, b_ip1, h_ip1, x_i, x_ip1, dx
        type(SegmentDT) :: seg, ds_seg
        type(Cross_SectionDT) :: cs, ds_cs
        integer, allocatable, dimension(:) :: ius_seg, ius_cs
        type(SegmentDT), allocatable, dimension(:) :: us_seg
        type(Cross_SectionDT), allocatable, dimension(:) :: us_cs

        ! Varibales for mass balance calculation
        real(sp) :: q_im1d2_mb, Vin_step, Vout_step, Vt_step, Vtp1_step, Et_step

        Vin_step  = 0.0_sp
        Vout_step = 0.0_sp
        Vt_step   = 0.0_sp
        Vtp1_step = 0.0_sp
        Et_step   = 0.0_sp    

        do iseg = 1, mesh%nseg
            seg = mesh%segments(iseg)
            call hy1d_get_downstream_segment(mesh, iseg, seg, ids_seg, ds_seg)
            call hy1d_get_upstream_segments(mesh, iseg, seg, ius_seg, us_seg)
            is_last_ds_seg = iseg .eq. ids_seg
            is_first_us_seg = iseg .eq. ius_seg(1)
            ifcs = seg%first_cross_section
            ilcs = seg%last_cross_section
            do ics = ifcs, ilcs
                cs = mesh%cross_sections(ics)
                call hy1d_get_downstream_cross_section(mesh, seg, ds_seg, ics, cs, ids_cs, ds_cs)
                call hy1d_get_upstream_cross_sections(mesh, seg, us_seg, ics, cs, ius_cs, us_cs)
                is_ds_bc = ics .eq. ilcs .and. is_last_ds_seg

                a_t = cs%level_widths(1)*hy1d_h(ics)
                q_ip1d2 = hy1d_q(ics)
                q_im1d2_mb = 0._sp

                ! Upstream boundary condition
                if (ics .eq. ifcs .and. is_first_us_seg) then
                    q_im1d2 = 0._sp
                    do i = 1, cs%nup
                        k = mesh%rowcol_to_ind_ac(cs%up_rowcols(i, 1), cs%up_rowcols(i, 2))
                        q_im1d2 = q_im1d2 + ac_qz(k, setup%nqz)
                    end do
                else if (ics .eq. ifcs .and. .not. is_first_us_seg) then
                    q_im1d2 = 0._sp
                    do i = 1, size(us_cs)  
                        q_im1d2 = q_im1d2 + hy1d_q(ius_cs(i))
                    end do
                else
                    q_im1d2 = hy1d_q(ius_cs(1))
                end if

                q_lat = 0._sp
                if (cs%rowcol(1) /= -99) then ! no hydrological couplings....
                    k = mesh%rowcol_to_ind_ac(cs%rowcol(1), cs%rowcol(2))
                    q_lat = ac_qtz(k, setup%nqz)
                end if
                do i = 1, cs%nlat
                    k = mesh%rowcol_to_ind_ac(cs%lat_rowcols(i, 1), cs%lat_rowcols(i, 2))
                    q_lat = q_lat + ac_qz(k, setup%nqz)
                end do

                x_i = cs%x
                b_ip1 = ds_cs%bathy
                h_ip1 = hy1d_h(ids_cs)
                if (is_ds_bc) then
                    call hy1d_get_downstream_boundary_condition_xbh(cs, us_cs, x_ip1, b_ip1, h_ip1)
                else
                    x_ip1 = ds_cs%x
                endif
                dx = x_i - x_ip1 

                a_tp1 = a_t + dt*(q_lat + q_im1d2 - q_ip1d2)/dx
                if (a_tp1 .lt. 0._sp) then 
                    a_tp1 = 0._sp
                    hy1d_q(ics) = (q_lat + q_im1d2) + a_t/(dt/dx)
                end if
                hy1d_h(ics) = a_tp1/cs%level_widths(1)

                ! ===== MASS BALANCE ANALYSIS =====
                if (ics .eq. ifcs .and. is_first_us_seg) then
                    q_im1d2_mb = 0._sp
                    do i = 1, cs%nup
                        k = mesh%rowcol_to_ind_ac(cs%up_rowcols(i, 1), cs%up_rowcols(i, 2))
                        q_im1d2_mb = q_im1d2_mb + ac_qz(k, setup%nqz)
                    end do
                end if

                Vin_step = Vin_step + (q_lat + q_im1d2_mb) * dt

                if (is_ds_bc) then
                    Vout_step = Vout_step + hy1d_q(ics) * dt
                end if

                Vt_step   = Vt_step   + a_t   * dx               
                Vtp1_step = Vtp1_step + a_tp1 * dx
                
            end do ! ics
        end do ! iseg

        ! --- Write aggregated global mass balance per step (Et only) ---
        if (time_step == 1 .and. hy1d_step == 1) then
            !mass_balance_file = "/home/adminberkaoui/Documents/article_garonne_latest/HH_LATEST_ARTICLE/HH_SIMULATIONS/hy1d_mb.csv"
            mass_balance_file = "/home/adminberkaoui/Bureau/branch_migration_check/hy1d_mb.csv"
            open(unit=10, file=mass_balance_file, status="replace")
            write(10,'(A)') "Et_step"
        end if
        if (Vtp1_step /= 0.0_sp) then
            Et_step = ((Vin_step - Vout_step) - (Vtp1_step - Vt_step)) / Vtp1_step
        else
            Et_step = -999.0_sp   ! flag for undefined relative error
        end if

        ! Write only Et
        write(10,'(G0.16)') Et_step
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

        integer :: global_step, global_total, last_printed_percent, percent

        call hy1d_non_inertial_get_dt(setup, mesh, hy1d_h, dt) 

        ceil = setup%dt/dt
        if (ceil == int(ceil)) then
            ntime_step = int(ceil)
        else
            ntime_step = int(ceil) + 1
        end if
        
        global_total = setup%ntime_step * ntime_step
        last_printed_percent = -1

        do t = 1, ntime_step
            global_step = (time_step - 1) * ntime_step + t
            call hy1d_non_inertial_momemtum(mesh, dt, hy1d_h, hy1d_q, time_step, t) ! Pass time_step and t
            call hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q,time_step,t) ! Pass t as hy1d_step counter and time_step as hydro_step counter
        end do

        ! ---- Print progress once per hydrological step ----
        percent = int(100.0 * time_step / setup%ntime_step)
        if (percent > last_printed_percent) then
            print *, "Global Progress: ", percent, "% completed"
            last_printed_percent = percent
        end if
        
        ! Close file after all hydrological time steps are complete
        if (time_step == setup%ntime_step) then
            close(10)
        end if

    end subroutine hy1d_non_inertial_time_step

end module md_hy1d_operator
