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
    character(len=256) :: wse_slope_file

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

        !TODO check BC   ;  mgb : y2=y1-0.0005*dxart*1000

        ! h_ip1 = h_i

    end subroutine hy1d_get_downstream_boundary_condition_xbh

    subroutine hy1d_compute_ar(cs, h, a, r)

        implicit none

        type(Cross_SectionDT), intent(in) :: cs
        real(sp), intent(in) :: h
        real(sp), intent(out) :: a, r

        real(sp) :: w

        ! ! Rectangular cross section
        ! w = min(sum(cs%level_widths), sum(ds_cs%level_widths))
        !     a_ip1d2 = h_ip1d2*w_ip1d2
        !     p_ip1d2 = w_ip1d2 + 2._sp*h_ip1d2
        !     r_ip1d2 = a_ip1d2/p_ip1d2

    end subroutine hy1d_compute_ar

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
        alpha = 0.9_sp ! parameter to control the CFL condition

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
        !dt = max(dt_min, 60._sp)

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
        real(sp) :: wse_i  ! New variable for water surface elevation
        real(sp) :: u_ip1d2, froude ! New variables for flow limitation
        type(SegmentDT) :: seg, ds_seg
        type(Cross_SectionDT) :: cs, ds_cs
        integer, allocatable, dimension(:) :: ius_seg, ius_cs
        type(SegmentDT), allocatable, dimension(:) :: us_seg
        type(Cross_SectionDT), allocatable, dimension(:) :: us_cs
        
        ! Open the WSE and slope file at the first hydrological and hydraulic time step
        !if (time_step == 1 .and. hy1d_step == 1) then
            !wse_slope_file = "/home/adminberkaoui/Documents/article_runs/cance/hy1d_sim/wse_wss.csv"
            !open(unit=11, file=wse_slope_file, status="replace")
            !write(11, '(A)') "hydro_step,hy1d_step,cs_idx,wse,s_ip1d2"
        !endif

        ! Initialize water depth at the first hydrological time step
        !if (time_step == 1 .and. hy1d_step == 1) then
            !hy1d_h(:) = 1.0_sp
        !end if

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
                !dx = 300._sp ! FOR DEBUGGING ~ quantile 1%
                !dx = 1000._sp ! FOR DEBUGGING
                s_ip1d2 = ((h_i + b_i) - (h_ip1 + b_ip1))/dx
                wse_i = h_i + b_i ! Calculate water surface elevation
                ! Write to the WSE and slope file
                !write(11, '(I5,",",I5,",",I5,",",F30.16,",",F30.16)') &
                    !time_step, hy1d_step, ics, wse_i, s_ip1d2

                h_ip1d2 = max(h_i + b_i, h_ip1 + b_ip1) - max(b_i, b_ip1)

                ! call hy1d_compute_ar(cs, h_ip1d2, a_ip1d2, r_ip1d2)

                w_ip1d2 = min(cs%level_widths(1), ds_cs%level_widths(1))
                !w_ip1d2 = 100._sp
                a_ip1d2 = h_ip1d2*w_ip1d2
                p_ip1d2 = w_ip1d2 + 2._sp*h_ip1d2
                r_ip1d2 = a_ip1d2/p_ip1d2

                if (h_ip1d2 .le. 0._sp) then
                    hy1d_q(ics) = 0._sp
                else
                    hy1d_q(ics) = (hy1d_q(ics) - gravity*dt*a_ip1d2*-s_ip1d2)/ &
                                  (1._sp + gravity*dt*((cs%manning(1)**2*abs(hy1d_q(ics)))/(a_ip1d2*r_ip1d2**(4._sp/3._sp))))
                end if

                ! if froude number > 1.0, limit flow
                ! TODO FROUDE RECTANGULAIRE...
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

    subroutine hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q, time_step, hy1d_step) ! added internal_step as argument

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), intent(in) :: dt
        real(sp), dimension(mesh%nac, setup%nqz), intent(in) :: ac_qtz, ac_qz
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_h
        real(sp), dimension(mesh%ncs), intent(inout) :: hy1d_q
        integer, intent(in) :: time_step, hy1d_step  ! Added

        integer :: i, k, iseg, ids_seg, ics, ids_cs, ifcs, ilcs
        logical :: is_last_ds_seg, is_first_us_seg, is_ds_bc
        real(sp) :: a_t, a_tp1, q_ip1d2, q_im1d2, q_lat, b_ip1, h_ip1, x_i, x_ip1, dx
        type(SegmentDT) :: seg, ds_seg
        type(Cross_SectionDT) :: cs, ds_cs
        integer, allocatable, dimension(:) :: ius_seg, ius_cs
        type(SegmentDT), allocatable, dimension(:) :: us_seg
        type(Cross_SectionDT), allocatable, dimension(:) :: us_cs
        

        ! New variables for hydrological connectivity debugging
        integer :: cs_cell_index ! Index of the cross-section's own cell
        integer, dimension(:), allocatable :: upstream_cell_indices ! Hydrological upstream cell indices
        integer, dimension(:), allocatable :: lateral_cell_indices ! Hydrological lateral cell indices
        integer :: n_up_cells, n_lat_cells
        character(len=100) :: up_inds_str, lat_inds_str

        ! Open the mass balance file at the first hydrological and hydraulic time step
        !if (time_step == 1 .and. hy1d_step == 1) then
            !mass_balance_file = "/home/adminberkaoui/Documents/article_runs/garonne/debug_carthage_hy1d/debug.csv"
            !open(unit=10, file=mass_balance_file, status="replace")
            !write(10, '(A)') "hydro_step,hy1d_step,dt,cs_idx," // &
                !"up_cell_idxs,netrain,q_im1d2,q_lat," // &
                !"a_t,a_tp1,h,q,dx,w"
        !endif       


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
                !a_t = 100._sp*hy1d_h(ics)
                q_ip1d2 = hy1d_q(ics)

                ! Initialize hydrological indices
                cs_cell_index = -99 ! Default value for missing cell index
                n_up_cells = 0
                n_lat_cells = 0

                if (allocated(upstream_cell_indices)) deallocate(upstream_cell_indices)
                if (allocated(lateral_cell_indices)) deallocate(lateral_cell_indices)
                allocate(upstream_cell_indices(max(1, cs%nup)))
                allocate(lateral_cell_indices(max(1, cs%nlat)))
                upstream_cell_indices = -99
                lateral_cell_indices = -99


                ! Upstream boundary condition
                if (ics .eq. ifcs .and. is_first_us_seg) then
                    q_im1d2 = 0._sp
                    n_up_cells = cs%nup
                    do i = 1, cs%nup
                        k = mesh%rowcol_to_ind_ac(cs%up_rowcols(i, 1), cs%up_rowcols(i, 2))
                        upstream_cell_indices(i) = k
                        q_im1d2 = q_im1d2 + ac_qz(k, setup%nqz)
                        !q_im1d2 = q_im1d2 + max(0._sp, ac_qz(k, setup%nqz)) ! handle negative upstream inflows
                    end do
                else if (ics .eq. ifcs .and. .not. is_first_us_seg) then
                    q_im1d2 = 0._sp
                    do i = 1, size(us_cs)  
                        q_im1d2 = q_im1d2 + hy1d_q(ius_cs(i))
                    end do
                else
                    q_im1d2 = hy1d_q(ius_cs(1))
                end if

                !k = mesh%rowcol_to_ind_ac(cs%rowcol(1), cs%rowcol(2))
                !q_lat = ac_qtz(k, setup%nqz)
                !do i = 1, cs%nlat
                    !k = mesh%rowcol_to_ind_ac(cs%lat_rowcols(i, 1), cs%lat_rowcols(i, 2))
                    !q_lat = q_lat + ac_qz(k, setup%nqz)
                !end do
                q_lat = 0._sp
                if (cs%rowcol(1) /= -99) then ! no hydrological couplings....
                    k = mesh%rowcol_to_ind_ac(cs%rowcol(1), cs%rowcol(2))
                    cs_cell_index = k
                    q_lat = ac_qtz(k, setup%nqz)
                    !q_lat = max(0._sp, ac_qtz(k, setup%nqz)) ! handle negative rain
                end if
                n_lat_cells = cs%nlat
                do i = 1, cs%nlat
                    k = mesh%rowcol_to_ind_ac(cs%lat_rowcols(i, 1), cs%lat_rowcols(i, 2))
                    lateral_cell_indices(i) = k
                    q_lat = q_lat + ac_qz(k, setup%nqz)
                    !q_lat = q_lat + max(0._sp, ac_qz(k, setup%nqz)) ! handle negative lateral inflows
                end do

                x_i = cs%x
                b_ip1 = ds_cs%bathy
                h_ip1 = hy1d_h(ids_cs)
                if (is_ds_bc) then
                    call hy1d_get_downstream_boundary_condition_xbh(cs, us_cs, x_ip1, b_ip1, h_ip1)
                else
                    x_ip1 = ds_cs%x
                endif
                dx = x_i - x_ip1 ! curvilinera abscissa
                !dx = max(dx, 300._sp) ! FOR DEBUGGING ~ quantile 1%
                !dx = max(dx, 100._sp)
                !dx = 1000._sp ! DEBUGGING

                !x_ip1 = ds_cs%x
                !dx = x_i - x_ip1
                !a_tp1 = a_t + dt*(q_lat + q_im1d2 - q_ip1d2)/1000._sp ! should be dx : pag : add comment vs interp lineaire qui va bien vs sous pas de temps
                a_tp1 = a_t + dt*(q_lat + q_im1d2 - q_ip1d2)/dx
                if (a_tp1 .lt. 0._sp) then
                    a_tp1 = 0._sp
                    !hy1d_q(ics) = (q_lat + q_im1d2) + a_t/(dt/1000._sp)
                    hy1d_q(ics) = (q_lat + q_im1d2) + a_t/(dt/dx)
                end if
                hy1d_h(ics) = a_tp1/cs%level_widths(1)
                !hy1d_h(ics) = a_tp1/100._sp

                ! ADDED FOR DEBUG: threshold for small depths
                !if (hy1d_h(ics) < 1.0e-3_sp) then
                    !hy1d_h(ics) = 0._sp
                    !a_tp1 = 0._sp
                    !hy1d_q(ics) = 0._sp
                !end if

                ! Create string representations of indices
                !if (n_up_cells > 0) then
                    !write(up_inds_str, '(100I6)') (upstream_cell_indices(i), i=1,n_up_cells)
                !else
                    !up_inds_str = "-"
                !endif
                
                !if (n_lat_cells > 0) then
                    !write(lat_inds_str, '(100I6)') (lateral_cell_indices(i), i=1,n_lat_cells)
                !else
                    !lat_inds_str = "-"
                !endif
                
                ! Write values to CSV file
                !write(10, '(I5,",",I5,",",F30.16,",",I5,",",A,",",F30.16,",",F30.16,",",F30.16,",", &
                !&F30.16,",",F30.16,",",F30.16,",",F30.16,",",F30.16,",",F30.16)') &
                !time_step, hy1d_step, dt, ics, trim(adjustl(up_inds_str)), &
                !ac_qtz(cs_cell_index, setup%nqz), q_im1d2, q_lat, a_t, a_tp1, &
                !hy1d_h(ics), hy1d_q(ics), dx, cs%level_widths(1)
                

            end do
        end do

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

        !hy1d_h(:) = 1.0_sp
        call hy1d_non_inertial_get_dt(setup, mesh, hy1d_h, dt) ! pag: à chaque dt hydraulique en ppe ; cpdt premiere implem commode vs dt_hydrologie ; ok vs variations smooth ?

        !ntime_step = ceiling(setup%dt/dt)

        ! pag : replaced ceiling for tapenade
        ceil = setup%dt/dt
        if (ceil == int(ceil)) then
            ntime_step = int(ceil)
        else
            ntime_step = int(ceil) + 1
        end if
        ! pag

        do t = 1, ntime_step
            call hy1d_non_inertial_momemtum(mesh, dt, hy1d_h, hy1d_q, time_step, t) ! Pass time_step and t
            call hy1d_non_inertial_mass(setup, mesh, dt, ac_qtz, ac_qz, hy1d_h, hy1d_q,time_step,t) ! Pass t as hy1d_step counter and time_step as hydro_step counter
        end do
        
        ! Close file after all hydrological time steps are complete
        !if (time_step == setup%ntime_step) then
            !close(10)
            !close(11)  ! Close the WSE and slope file
        !endif

    end subroutine hy1d_non_inertial_time_step

end module md_hy1d_operator
