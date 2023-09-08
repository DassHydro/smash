!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr4_lr_forward
!%      - gr4_kw_forward
!%      - grd_lr_forward

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
    use md_gr_operator !% only: gr_interception, gr_production, gr_exchange, &
    !% & gr_transfer
    use md_routing_operator !% only: upstream_discharge, linear_routing, kinematic_wave1d

    implicit none

contains

    subroutine gr4_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - opr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: llr)
        !% - opr_states:     (1: hi), (2: hp), (3: ht), (4: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, qup, qrout
        integer :: t, i, row, col, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

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

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), parameters%opr_parameters%values(row, col, 1), &
                    & parameters%opr_initial_states%values(row, col, 1), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, parameters%opr_parameters%values(row, col, 2), 9._sp/4._sp, &
                    & parameters%opr_initial_states%values(row, col, 2), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_exchange(parameters%opr_parameters%values(row, col, 4), &
                    & parameters%opr_initial_states%values(row, col, 3), l)

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

                call gr_transfer(5._sp, prcp(row, col), prr, parameters%opr_parameters%values(row, col, 3), &
                & parameters%opr_initial_states%values(row, col, 3), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = (qr + qd)

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
                    & parameters%opr_parameters%values(row, col, 5), parameters%opr_initial_states%values(row, col, 4), &
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

                output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr4_lr_forward

    subroutine gr4_kw_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - opr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: akw), (6: bkw)
        !% - opr_states:     (1: hi), (2: hp), (3: ht)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        integer, parameter :: zq = 2
        real(sp), dimension(mesh%nrow, mesh%ncol, zq) :: qt, q
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, &
        & qlijm1, qlij, qijm1, qim1j, qij
        integer :: t, i, row, col, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        q = 0._sp
        qt = 0._sp

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

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), parameters%opr_parameters%values(row, col, 1), &
                    & parameters%opr_initial_states%values(row, col, 1), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, parameters%opr_parameters%values(row, col, 2), 9._sp/4._sp, &
                    & parameters%opr_initial_states%values(row, col, 2), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_exchange(parameters%opr_parameters%values(row, col, 4), &
                    & parameters%opr_initial_states%values(row, col, 3), l)

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

                call gr_transfer(5._sp, prcp(row, col), prr, parameters%opr_parameters%values(row, col, 3), &
                & parameters%opr_initial_states%values(row, col, 3), qr)

                qd = max(0._sp, prd + l)

                qt(row, col, zq) = (qr + qd)

            end do
!~             !$OMP end parallel do

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

                    call kinematic_wave1d(setup%dt, mesh%dx(row, col), parameters%opr_parameters%values(row, col, 5), &
                    & parameters%opr_parameters%values(row, col, 6), qlijm1, qlij, qim1j, qijm1, qij)

                    q(row, col, zq) = qij

                else

                    q(row, col, zq) = qt(row, col, zq)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2), zq)

            end do

        end do !% [ END DO TIME ]

    end subroutine gr4_kw_forward

    subroutine gr5_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - opr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: aexc), (6: llr)
        !% - opr_states:     (1: hi), (2: hp), (3: ht), (4: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, qup, qrout
        integer :: t, i, row, col, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

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

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), parameters%opr_parameters%values(row, col, 1), &
                    & parameters%opr_initial_states%values(row, col, 1), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, parameters%opr_parameters%values(row, col, 2), 9._sp/4._sp, &
                    & parameters%opr_initial_states%values(row, col, 2), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_threshold_exchange(parameters%opr_parameters%values(row, col, 4),&
                    & parameters%opr_initial_states%values(row, col, 3), parameters%opr_parameters%values(row, col, 5), l)

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

                call gr_transfer(5._sp, prcp(row, col), prr, parameters%opr_parameters%values(row, col, 3), &
                & parameters%opr_initial_states%values(row, col, 3), qr)

                qd = max(0._sp, prd + l)

                qt(row, col) = (qr + qd)

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
                    & parameters%opr_parameters%values(row, col, 6), parameters%opr_initial_states%values(row, col, 4), &
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

                output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr5_lr_forward

    subroutine gr5_kw_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - opr_parameters: (1: ci), (2: cp), (3: ct), (4: kexc), (5: texc), (6: akw), (7: bkw)
        !% - opr_states:     (1: hi), (2: hp), (3: ht)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        integer, parameter :: zq = 2
        real(sp), dimension(mesh%nrow, mesh%ncol, zq) :: qt, q
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet
        real(sp) :: ei, pn, en, pr, perc, l, prr, prd, qr, qd, &
        & qlijm1, qlij, qijm1, qim1j, qij
        integer :: t, i, row, col, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        q = 0._sp
        qt = 0._sp

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

                if (prcp(row, col) .ge. 0._sp .and. pet(row, col) .ge. 0._sp) then !% [ IF PRCP GAP ]

                    !% =============================================================================================== %!
                    !%   Interception module
                    !% =============================================================================================== %!

                    call gr_interception(prcp(row, col), pet(row, col), parameters%opr_parameters%values(row, col, 1), &
                    & parameters%opr_initial_states%values(row, col, 1), pn, ei)

                    en = pet(row, col) - ei

                    !% =============================================================================================== %!
                    !%   Production module
                    !% =============================================================================================== %!

                    call gr_production(pn, en, parameters%opr_parameters%values(row, col, 2), 9._sp/4._sp, &
                    & parameters%opr_initial_states%values(row, col, 2), pr, perc)

                    !% =============================================================================================== %!
                    !%   Exchange module
                    !% =============================================================================================== %!

                    call gr_threshold_exchange(parameters%opr_parameters%values(row, col, 4),&
                    & parameters%opr_initial_states%values(row, col, 3), parameters%opr_parameters%values(row, col, 5), l)

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

                call gr_transfer(5._sp, prcp(row, col), prr, parameters%opr_parameters%values(row, col, 3), &
                & parameters%opr_initial_states%values(row, col, 3), qr)

                qd = max(0._sp, prd + l)

                qt(row, col, zq) = (qr + qd)

            end do
!~             !$OMP end parallel do

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

                    call kinematic_wave1d(setup%dt, mesh%dx(row, col), parameters%opr_parameters%values(row, col, 6), &
                    & parameters%opr_parameters%values(row, col, 7), qlijm1, qlij, qim1j, qijm1, qij)

                    q(row, col, zq) = qij

                else

                    q(row, col, zq) = qt(row, col, zq)

                end if !% [ END IF BC ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2), zq)

            end do

        end do !% [ END DO TIME ]

    end subroutine gr5_kw_forward

    subroutine grd_lr_forward(setup, mesh, input_data, parameters, output, options, returns)
        !% Note:
        !% - opr_parameters: (1: cp), (2: ct), (3: llr)
        !% - opr_states:     (1: hp), (2: ht), (3: hlr)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp, pet, q, qt
        real(sp) :: ei, pn, en, pr, perc, prr, qr, qup, qrout
        integer :: t, i, row, col, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

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
!~             !$OMP& private(i, row, col, ei, pn, en, pr, perc, prr, qr)
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

                    call gr_production(pn, en, parameters%opr_parameters%values(row, col, 1), 9._sp/4._sp, &
                    & parameters%opr_initial_states%values(row, col, 1), pr, perc)

                else

                    pr = 0._sp
                    perc = 0._sp

                end if !% [ END IF PRCP GAP ]

                !% =================================================================================================== %!
                !%   Transfer module
                !% =================================================================================================== %!

                prr = pr + perc

                call gr_transfer(5._sp, prcp(row, col), prr, parameters%opr_parameters%values(row, col, 2), &
                & parameters%opr_initial_states%values(row, col, 2), qr)

                qt(row, col) = qr

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
                    & parameters%opr_parameters%values(row, col, 3), parameters%opr_initial_states%values(row, col, 3), &
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

                output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine grd_lr_forward

end module md_forward_structure
