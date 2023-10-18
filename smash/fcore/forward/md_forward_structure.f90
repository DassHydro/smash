!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr4_lr_forward
!%      - gr4_lr_ss_forward
!%      - gr4_kw_forward
!%      - gr5_kw_forward
!%      - gr5_kw_forward
!%      - loieau_lr_forward
!%      - grd_lr_forward
!%      - vic3l_lr_forward

module md_forward_structure

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use mwd_sparse_matrix_manipulation !% only: sparse_matrix_to_matrix
    use md_gr_operator !% only: gr_interception, gr_production, gr_exchange, gr_threshold_exchange, &
    !% & gr_transfer
    use md_neural_ode_operator !% only: gr_ode_implicit_euler
    use md_vic3l_operator !% only: vic3l_canopy_evapotranspiration, vic3l_upper_soil_layer_evaporation, &
    !% & vic3l_infiltration, vic3l_drainage, vic3l_baseflow
    use md_routing_operator !% only: upstream_discharge, linear_routing, kinematic_wave1d

    implicit none

contains

    subroutine gr4_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: llr)
        !% - rr_states:     (1: hi), (2: hp), (3: ht), (4: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ci, cp, ct, kexc, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hi, hp, ht, hlr
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        ci = parameters%rr_parameters%values(:, :, 1)
        cp = parameters%rr_parameters%values(:, :, 2)
        ct = parameters%rr_parameters%values(:, :, 3)
        kexc = parameters%rr_parameters%values(:, :, 4)
        llr = parameters%rr_parameters%values(:, :, 5)

        hi = parameters%rr_initial_states%values(:, :, 1)
        hp = parameters%rr_initial_states%values(:, :, 2)
        ht = parameters%rr_initial_states%values(:, :, 3)
        hlr = parameters%rr_initial_states%values(:, :, 4)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_exchange(kexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = (qr + qd)

            end do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hi
            output%rr_final_states%values(:, :, 2) = hp
            output%rr_final_states%values(:, :, 3) = ht
            output%rr_final_states%values(:, :, 4) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine gr4_lr_forward

    subroutine gr4_lr_ss_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: llr)
        !% - rr_states:     (1: hi), (2: hp), (3: ht), (4: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ci, cp, ct, kexc, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hi, hp, ht, hlr
        real(sp) :: ei, pn, en, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!
        
        ci = parameters%rr_parameters%values(:, :, 1)
        cp = parameters%rr_parameters%values(:, :, 2)
        ct = parameters%rr_parameters%values(:, :, 3)
        kexc = parameters%rr_parameters%values(:, :, 4)
        llr = parameters%rr_parameters%values(:, :, 5)

        hi = parameters%rr_initial_states%values(:, :, 1)
        hp = parameters%rr_initial_states%values(:, :, 2)
        ht = parameters%rr_initial_states%values(:, :, 3)
        hlr = parameters%rr_initial_states%values(:, :, 4)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

!~             !$OMP parallel do schedule(static) num_threads(options%comm%ncpu) &
!~             !$OMP& shared(setup, mesh, input_data, parameters, output, options, returns, prcp, pet, qt) &
!~             !$OMP& private(i, row, col, ei, pn, en, pr, perc, l, prr, prd, qr, qd)
            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF NO PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, ei)

                    en = pet(row, col) - ei

                else
                    pn = 0._sp
                    en = 0._sp

                end if

                !% =============================================================================================== %!
                !%   Production and Transfer State-Space
                !% =============================================================================================== %!

                call gr_ode_implicit_euler(pn, en, cp(row, col), ct(row, col), kexc(row, col), &
                & hp(row, col), ht(row, col), qt(row, col))

            end do
!~             !$OMP end parallel do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hi
            output%rr_final_states%values(:, :, 2) = hp
            output%rr_final_states%values(:, :, 3) = ht
            output%rr_final_states%values(:, :, 4) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine gr4_lr_ss_forward

    subroutine gr4_kw_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: akw), (6: bkw)
        !% - rr_states:     (1: hi), (2: hp), (3: ht)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        integer, parameter :: zq = 2
        real(sp), dimension(mesh%nrow, mesh%ncol, zq) :: qt, q
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ci, cp, ct, kexc, akw, bkw
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hi, hp, ht
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, &
        & qlijm1, qlij, qijm1, qim1j, qij
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        ci = parameters%rr_parameters%values(:, :, 1)
        cp = parameters%rr_parameters%values(:, :, 2)
        ct = parameters%rr_parameters%values(:, :, 3)
        kexc = parameters%rr_parameters%values(:, :, 4)
        akw = parameters%rr_parameters%values(:, :, 5)
        bkw = parameters%rr_parameters%values(:, :, 6)

        hi = parameters%rr_initial_states%values(:, :, 1)
        hp = parameters%rr_initial_states%values(:, :, 2)
        ht = parameters%rr_initial_states%values(:, :, 3)

        q = 0._sp
        qt = 0._sp
        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Swapping Q Buffer
            !% =============================================================================================================== %!

            do i = 1, zq - 1

                q(:, :, i) = q(:, :, i + 1)
                qt(:, :, i) = qt(:, :, i + 1)

            end do

            q(:, :, zq) = 0._sp
            qt(:, :, zq) = 0._sp

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_exchange(kexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col, zq) = (qr + qd)

            end do

            qt(:, :, zq) = qt(:, :, zq)*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q(:, :, zq), qim1j)

                    qlijm1 = qt(row, col, zq - 1)
                    qlij = qt(row, col, zq)
                    qijm1 = q(row, col, zq - 1)

                    call kinematic_wave1d(setup%dt, mesh%dx(row, col), akw(row, col), bkw(row, col), &
                    & qlijm1, qlij, qim1j, qijm1, qij)

                    q(row, col, zq) = qij

                else

                    q(row, col, zq) = qt(row, col, zq)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2), zq)

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hi
            output%rr_final_states%values(:, :, 2) = hp
            output%rr_final_states%values(:, :, 3) = ht

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q(:, :, zq)
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine gr4_kw_forward

    subroutine gr5_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: aexc), (6: llr)
        !% - rr_states:     (1: hi), (2: hp), (3: ht), (4: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ci, cp, ct, kexc, aexc, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hi, hp, ht, hlr
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        ci = parameters%rr_parameters%values(:, :, 1)
        cp = parameters%rr_parameters%values(:, :, 2)
        ct = parameters%rr_parameters%values(:, :, 3)
        kexc = parameters%rr_parameters%values(:, :, 4)
        aexc = parameters%rr_parameters%values(:, :, 5)
        llr = parameters%rr_parameters%values(:, :, 6)

        hi = parameters%rr_initial_states%values(:, :, 1)
        hp = parameters%rr_initial_states%values(:, :, 2)
        ht = parameters%rr_initial_states%values(:, :, 3)
        hlr = parameters%rr_initial_states%values(:, :, 4)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_threshold_exchange(kexc(row, col), aexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = (qr + qd)

            end do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hi
            output%rr_final_states%values(:, :, 2) = hp
            output%rr_final_states%values(:, :, 3) = ht
            output%rr_final_states%values(:, :, 4) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine gr5_lr_forward

    subroutine gr5_kw_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: aexc), (6: akw), (7: bkw)
        !% - rr_states:     (1: hi), (2: hp), (3: ht)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        integer, parameter :: zq = 2
        real(sp), dimension(mesh%nrow, mesh%ncol, zq) :: qt, q
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ci, cp, ct, kexc, aexc, akw, bkw
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hi, hp, ht
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, &
        & qlijm1, qlij, qijm1, qim1j, qij
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        ci = parameters%rr_parameters%values(:, :, 1)
        cp = parameters%rr_parameters%values(:, :, 2)
        ct = parameters%rr_parameters%values(:, :, 3)
        kexc = parameters%rr_parameters%values(:, :, 4)
        aexc = parameters%rr_parameters%values(:, :, 5)
        akw = parameters%rr_parameters%values(:, :, 6)
        bkw = parameters%rr_parameters%values(:, :, 7)

        hi = parameters%rr_initial_states%values(:, :, 1)
        hp = parameters%rr_initial_states%values(:, :, 2)
        ht = parameters%rr_initial_states%values(:, :, 3)

        q = 0._sp
        qt = 0._sp
        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Swapping Q Buffer
            !% =============================================================================================================== %!

            do i = 1, zq - 1

                q(:, :, i) = q(:, :, i + 1)
                qt(:, :, i) = qt(:, :, i + 1)

            end do

            q(:, :, zq) = 0._sp
            qt(:, :, zq) = 0._sp

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), ci(row, col), hi(row, col), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_threshold_exchange(kexc(row, col), aexc(row, col), ht(row, col), l)

                else

                    pr = 0._sp
                    perc = 0._sp
                    l = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = 0.9_sp*(pr + perc) + l
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qd = max(0._sp, prd + l)

                qt(row, col, zq) = (qr + qd)

            end do

            qt(:, :, zq) = qt(:, :, zq)*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q(:, :, zq), qim1j)

                    qlijm1 = qt(row, col, zq - 1)
                    qlij = qt(row, col, zq)
                    qijm1 = q(row, col, zq - 1)

                    call kinematic_wave1d(setup%dt, mesh%dx(row, col), akw(row, col), bkw(row, col), &
                    & qlijm1, qlij, qim1j, qijm1, qij)

                    q(row, col, zq) = qij

                else

                    q(row, col, zq) = qt(row, col, zq)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2), zq)

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hi
            output%rr_final_states%values(:, :, 2) = hp
            output%rr_final_states%values(:, :, 3) = ht

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q(:, :, zq)
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine gr5_kw_forward

    subroutine loieau_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: ca), (2: cc), (3: kb), (4: llr)
        !% - rr_states:     (1: ha), (2: hc), (3: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ca, cc, kb, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: ha, hc, hlr
        real(sp) :: ei, pn, en, pr, perc, prr, prd, qr, qd, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        ca = parameters%rr_parameters%values(:, :, 1)
        cc = parameters%rr_parameters%values(:, :, 2)
        kb = parameters%rr_parameters%values(:, :, 3)
        llr = parameters%rr_parameters%values(:, :, 4)

        ha = parameters%rr_initial_states%values(:, :, 1)
        hc = parameters%rr_initial_states%values(:, :, 2)
        hlr = parameters%rr_initial_states%values(:, :, 3)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!
                    ei = min(pet(row, col), prcp(row, col))
                    pn = max(0._sp, prcp(row, col) - ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, ca(row, col), 9._sp/4._sp, ha(row, col), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = 0.9_sp*(pr + perc)
                prd = 0.1_sp*(pr + perc)

                call gr_transfer(4._sp, prcp(row, col), prr, cc(row, col), hc(row, col), qr)

                qd = max(0._sp, prd)

                qt(row, col) = kb(row, col)*(qr + qd)

            end do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), &
                    & qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = ha
            output%rr_final_states%values(:, :, 2) = hc
            output%rr_final_states%values(:, :, 3) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine loieau_lr_forward

    subroutine grd_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: cp), (2: ct), (3: llr)
        !% - rr_states:     (1: hp), (2: ht), (3: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: cp, ct, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hp, ht, hlr
        real(sp) :: ei, pn, en, pr, perc, prr, qr, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        cp = parameters%rr_parameters%values(:, :, 1)
        ct = parameters%rr_parameters%values(:, :, 2)
        llr = parameters%rr_parameters%values(:, :, 3)

        hp = parameters%rr_initial_states%values(:, :, 1)
        ht = parameters%rr_initial_states%values(:, :, 2)
        hlr = parameters%rr_initial_states%values(:, :, 3)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    ei = min(pet(row, col), prcp(row, col))

                    pn = max(0._sp, prcp(row, col) - ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, cp(row, col), 9._sp/4._sp, hp(row, col), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = pr + perc

                call gr_transfer(5._sp, prcp(row, col), prr, ct(row, col), ht(row, col), qr)

                qt(row, col) = qr

            end do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store final states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hp
            output%rr_final_states%values(:, :, 2) = ht
            output%rr_final_states%values(:, :, 3) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine grd_lr_forward

    subroutine vic3l_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - rr_parameters: (1: b), (2: cusl), (3: cmsl), (4: cbsl), (5: ks), (6: pbc), &
        !%                   (7: ds), (8: dsm), (9: ws), (10: llr)
        !% - rr_states:     (1: hcl), (2: husl), (3: hmsl), (4: hbsl), (5: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp), dimension(mesh%nrow, mesh%ncol) :: b, cusl, cmsl, cbsl, ks, pbc, ds, dsm, ws, llr
        real(sp), dimension(mesh%nrow, mesh%ncol) :: hcl, husl, hmsl, hbsl, hlr
        real(sp) :: pn, en, qr, qb, qup, qrout
        integer :: t, i, row, col, g, iret

        !% =================================================================================================================== %!
        !%  Initialise parameters
        !% =================================================================================================================== %!

        b = parameters%rr_parameters%values(:, :, 1)
        cusl = parameters%rr_parameters%values(:, :, 2)
        cmsl = parameters%rr_parameters%values(:, :, 3)
        cbsl = parameters%rr_parameters%values(:, :, 4)
        ks = parameters%rr_parameters%values(:, :, 5)
        pbc = parameters%rr_parameters%values(:, :, 6)
        ds = parameters%rr_parameters%values(:, :, 7)
        dsm = parameters%rr_parameters%values(:, :, 8)
        ws = parameters%rr_parameters%values(:, :, 9)
        llr = parameters%rr_parameters%values(:, :, 10)

        hcl = parameters%rr_initial_states%values(:, :, 1)
        husl = parameters%rr_initial_states%values(:, :, 2)
        hmsl = parameters%rr_initial_states%values(:, :, 3)
        hbsl = parameters%rr_initial_states%values(:, :, 4)
        hlr = parameters%rr_initial_states%values(:, :, 5)

        iret = 0
        do t = 1, setup%ntime_step !% [ DO TIME ]

            !% =============================================================================================================== %!
            !%  Getting Precipitation and PET at time step
            !% =============================================================================================================== %!

            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(t), prcp)
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(t), pet)

            else

                prcp = input_data%atmos_data%prcp(:, :, t)
                pet = input_data%atmos_data%pet(:, :, t)

            end if

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =================================================================================================== %!
                    !%   Canopy interception module
                    !% =================================================================================================== %!

                    ! Canopy maximum capacity is (0.2 * LAI). Here we fix maximum capacity to 1 mm
                    call vic3l_canopy_interception(prcp(row, col), pet(row, col), 1._sp, hcl(row, col), pn, en)

                    !% =================================================================================================== %!
                    !%   Upper soil layer evaporation module
                    !% =================================================================================================== %!

                    call vic3l_upper_soil_layer_evaporation(en, b(row, col), cusl(row, col), husl(row, col))

                    !% =================================================================================================== %!
                    !%   Infiltration module
                    !% =================================================================================================== %!

                    call vic3l_infiltration(pn, b(row, col), cusl(row, col), cmsl(row, col), husl(row, col), &
                    & hmsl(row, col), qr)

                    !% =================================================================================================== %!
                    !%   Vertical drainage module
                    !% =================================================================================================== %!

                    call vic3l_drainage(cusl(row, col), cmsl(row, col), cbsl(row, col), ks(row, col), pbc(row, col), &
                    & husl(row, col), hmsl(row, col), hbsl(row, col))

                else

                    qr = 0._sp

                end if

                !% ======================================================================================================= %!
                !%   Baseflow module
                !% ======================================================================================================= %!

                call vic3l_baseflow(cbsl(row, col), ds(row, col), dsm(row, col), ws(row, col), hbsl(row, col), qb)

                qt(row, col) = qr + qb

            end do

            qt = qt*1e-3_sp*mesh%dx*mesh%dy/setup%dt

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                row = mesh%path(1, i)
                col = mesh%path(2, i)

                if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

                if (mesh%flwacc(row, col) .gt. mesh%dx(row, col)*mesh%dy(row, col)) then !% [ IF BC ]

                    !% =================================================================================================== %!
                    !%   Routing module
                    !% =================================================================================================== %!

                    call upstream_discharge(mesh%nrow, mesh%ncol, row, col, mesh%dx(row, col), &
                    & mesh%dy(row, col), mesh%flwacc(row, col), mesh%flwdir, q, qup)

                    call linear_routing(setup%dt, mesh%dx(row, col), mesh%dy(row, col), mesh%flwacc(row, col), &
                    & llr(row, col), hlr(row, col), qup, qrout)

                    q(row, col) = qrout + qt(row, col)

                else

                    q(row, col) = qt(row, col)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

            !% =============================================================================================================== %!
            !%   Store states
            !% =============================================================================================================== %!

            output%rr_final_states%values(:, :, 1) = hcl
            output%rr_final_states%values(:, :, 2) = husl
            output%rr_final_states%values(:, :, 3) = hmsl
            output%rr_final_states%values(:, :, 4) = hbsl
            output%rr_final_states%values(:, :, 5) = hlr

            !% =============================================================================================================== %!
            !%   Store optional returns
            !% =============================================================================================================== %!

            !$AD start-exclude
            if (allocated(returns%mask_time_step)) then
                if (returns%mask_time_step(i)) then
                    iret = iret + 1
                    if (returns%rr_states_flag) returns%rr_states(iret) = output%rr_final_states
                    if (returns%q_domain_flag) returns%q_domain(:, :, iret) = q
                end if
            end if
            !$AD end-exclude

        end do !% [ END DO TIME ]

    end subroutine vic3l_lr_forward

end module md_forward_structure
