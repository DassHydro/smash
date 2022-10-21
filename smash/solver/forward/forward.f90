subroutine forward(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)
    
    use mwd_common !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: Hyper_ParametersDT
    use mwd_states !% only: Hyper_StatesDT
    use mwd_output !% only: OutputDT
    use md_operator !% only: GR_interception, GR_production, GR_exchange, &
    !% & GR_transferN, upstream_discharge, sparse_upstream_discharge, GR_transfer1
    use mwd_cost !% only: compute_cost, hyper_compuste_cost
    
    implicit none

    !% =================================================================================================================== %!
    !%   Derived Type Variables (shared)
    !% =================================================================================================================== %!

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(in) :: parameters, parameters_bgd
    type(StatesDT), intent(inout) :: states, states_bgd
    type(OutputDT), intent(inout) :: output
    real(sp), intent(inout) :: cost
    
    !% =================================================================================================================== %!
    !%   Local Variables (private)
    !% =================================================================================================================== %!
    
    type(StatesDT) :: states_imd
    real(sp), dimension(:,:), allocatable :: q
    real(sp), dimension(:), allocatable :: sparse_q
    real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prl, prd, &
    & qd, qr, ql, qt, qup, qrout
    integer :: t, i, row, col, k, g
    
    cost = 0._sp
    states_imd = states
    
    if (setup%sparse_storage) then
        
        allocate(sparse_q(mesh%nac))
        
    else
        
        allocate(q(mesh%nrow, mesh%ncol))
    
    end if
    
    !% =================================================================================================================== %!
    !%   Begin subroutine
    !% =================================================================================================================== %!

    do t=1, setup%ntime_step !% [ DO TIME ]
    
        do i=1, mesh%nrow * mesh%ncol !% [ DO SPACE ]
        
        !% =============================================================================================================== %!
        !%   Local Variables Initialisation for time step (t) and cell (i)
        !% =============================================================================================================== %!
        
            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
            prl = 0._sp
            prd = 0._sp
            qd = 0._sp
            qr = 0._sp
            ql = 0._sp
            qup = 0._sp
            qrout = 0._sp
            
            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of drained area 
            !% =========================================================================================================== %!
            
            if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]
            
                row = mesh%path(1, i)
                col = mesh%path(2, i)
                if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)
                
                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!
                
                if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]
                        
                    if (setup%sparse_storage) then
                    
                        prcp = input_data%sparse_prcp(k, t)
                        pet = input_data%sparse_pet(k, t)
                    
                    else
                    
                        prcp = input_data%prcp(row, col, t)
                        pet = input_data%pet(row, col, t)
                    
                    end if
                    
                    if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]
                
                        !% =============================================================================================== %!
                        !%   Interception module case [ 0 - 1 ]
                        !% =============================================================================================== %!
                    
                        select case(setup%interception_module)
                        
                        case(0)
                        
                            ei = min(pet, prcp)
                            
                            pn = max(0._sp, prcp - ei)
                            
                        case(1)

                            call GR_interception(prcp, pet, parameters%ci(row, col), states%hi(row, col), pn, ei)
                        
                        end select
                        
                        en = pet - ei
                        
                        !% =============================================================================================== %!
                        !%   Production module case [ 0 ]
                        !% =============================================================================================== %!
                        
                        select case(setup%production_module)

                        case(0)
                        
                            call GR_production(pn, en, parameters%cp(row, col), parameters%beta(row, col), &
                            & states%hp(row, col), pr, perc)
                            
                        end select
                            
                        !% =============================================================================================== %!
                        !%   Exchange module case [ 0 - 1 ]
                        !% =============================================================================================== %!
                    
                        select case(setup%exchange_module)
                        
                        case(0)
                        
                            l = 0._sp
                            
                        case(1)
                        
                            call GR_exchange(parameters%exc(row, col), states%hft(row, col), l)
                        
                        end select
                        
                    end if !% [ END IF PRCP GAP ]
                    
                    !% =================================================================================================== %!
                    !%   Transfer module case [ 0 ]
                    !% =================================================================================================== %!
                    
                    select case(setup%transfer_module)
                    
                    case(0)
                    
                        prr = parameters%alpha(row, col) * (pr + perc) + l
                        prd = (1._sp - parameters%alpha(row, col)) * (pr + perc)
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        qd = max(0._sp, prd + l)
                        
                        
                    case(1)
                    
                        prr = 0.9_sp * parameters%alpha(row, col) * (pr + perc) + l
                        prl = 0.9_sp * (1._sp - parameters%alpha(row, col)) * (pr + perc)
                        prd = 0.1_sp * (pr + perc)
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        call GR_transferN(5._sp, prcp, prl, parameters%cst(row, col), states%hst(row, col), ql)
                        
                        qd = max(0._sp, prd + l)

                    end select
                    
                    qt = (qd + qr + ql)
                    
                    !% =================================================================================================== %!
                    !%   Routing module case [ 0 - 1 ]
                    !% =================================================================================================== %!
                    
                    select case(setup%routing_module)
                
                    case(0)
                    
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, mesh%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flwdir, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                        
                            sparse_q(k) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flwdir, mesh%drained_area, row, col, q, qup)
                        
                            q(row, col) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt
                    
                        end if
                        
                    case(1)
                        
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, mesh%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flwdir, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                            sparse_q(k) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flwdir, mesh%drained_area, row, col, q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)
                        
                            q(row, col) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt
                    
                        end if
                
                    end select
                
                end if !% [ END IF ACTIVE CELL ]
                
            end if !% [ END IF PATH ]
        
        end do !% [ END DO SPACE ]
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge at gauge
        !% =============================================================================================================== %!
        
        do g=1, mesh%ng
        
            row = mesh%gauge_pos(g, 1)
            col = mesh%gauge_pos(g, 2)
            
            if (setup%sparse_storage) then
                
                k = mesh%rowcol_to_ind_sparse(row, col)
                
                output%qsim(g, t) = sparse_q(k)

            else
            
                output%qsim(g, t) = q(row, col)
            
            end if
        
        end do
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge on domain (optional)
        !% =============================================================================================================== %!
        
        if (setup%save_qsim_domain) then
        
            if (setup%sparse_storage) then
            
                output%sparse_qsim_domain(:, t) = sparse_q
                
            else
            
                output%qsim_domain(:, :, t) = q
            
            end if
        
        end if
        
        !% =============================================================================================================== %!
        !%   Store simulated net rainfall on domain (optional)
        !%   The net rainfall over a surface is a fictitious quantity that corresponds to 
        !%   the part of the rainfall water depth that actually causes runoff. 
        !% =============================================================================================================== %!
        
        if (setup%save_net_prcp_domain) then
        
            if (setup%sparse_storage) then
            
                output%sparse_net_prcp_domain(:, t) = qt
                
            else
            
                output%net_prcp_domain(:, :, t) = qt
            
            end if
        
        end if
        
    end do !% [ END DO TIME ]
    
    !% =============================================================================================================== %!
    !%   Store states at final time step and reset states
    !% =============================================================================================================== %!
    
    output%fstates = states
    states = states_imd
    
    !% =================================================================================================================== %!
    !%   Compute J
    !% =================================================================================================================== %!
    
    call compute_cost(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)
        
end subroutine forward

!% Subroutine is a copy of forward
!% Find a way to avoid a full copy
!% WARNING: Differentiated module
subroutine hyper_forward(setup, mesh, input_data, &
    & hyper_parameters, hyper_parameters_bgd, hyper_states, &
    & hyper_states_bgd, output, cost)

    use mwd_common !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: Hyper_ParametersDT, ParametersDT_initialise, hyper_parameters_to_parameters
    use mwd_states !% only: Hyper_StatesDT, StatesDT_initialise, hyper_states_to_states
    use mwd_output !% only: OutputDT
    use md_operator !% only: GR_interception, GR_production, GR_exchange, &
    !% & GR_transferN, upstream_discharge, sparse_upstream_discharge, GR_transfer1
    use mwd_cost !% only: compute_cost
    
    implicit none
    
    !% =================================================================================================================== %!
    !%   Derived Type Variables (shared)
    !% =================================================================================================================== %!

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(inout) :: input_data 
    type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_bgd
    type(Hyper_StatesDT), intent(inout) :: hyper_states, hyper_states_bgd
    type(OutputDT), intent(inout) :: output
    real(sp), intent(inout) :: cost
    
    !% =================================================================================================================== %!
    !%   Local Variables (private)
    !% =================================================================================================================== %!

    type(ParametersDT) :: parameters
    type(StatesDT) :: states
    real(sp), dimension(:,:), allocatable :: q
    real(sp), dimension(:), allocatable :: sparse_q
    real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prl, prd, &
    & qd, qr, ql, qt, qup, qrout
    integer :: t, i, row, col, k, g
    
    !$AD NOCHECKPOINT
    call ParametersDT_initialise(parameters, mesh)
    
    !$AD NOCHECKPOINT
    call StatesDT_initialise(states, mesh)
    
    cost = 0._sp
    
    call hyper_parameters_to_parameters(hyper_parameters, parameters, setup, input_data)
    call hyper_states_to_states(hyper_states, states, setup, input_data)
    
     if (setup%sparse_storage) then
        
        allocate(sparse_q(mesh%nac))
        
     else
        
        allocate(q(mesh%nrow, mesh%ncol))
    
    end if
    
    !% =================================================================================================================== %!
    !%   Begin subroutine
    !% =================================================================================================================== %!

    do t=1, setup%ntime_step !% [ DO TIME ]
    
        do i=1, mesh%nrow * mesh%ncol !% [ DO SPACE ]
        
        !% =============================================================================================================== %!
        !%   Local Variables Initialisation for time step (t) and cell (i)
        !% =============================================================================================================== %!
        
            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
            prl = 0._sp
            prd = 0._sp
            qd = 0._sp
            qr = 0._sp
            ql = 0._sp
            qup = 0._sp
            qrout = 0._sp
            
            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of drained area 
            !% =========================================================================================================== %!
            
            if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]
            
                row = mesh%path(1, i)
                col = mesh%path(2, i)
                if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)
                
                !% ======================================================================================================= %!
                !%   Global/Local active cell
                !% ======================================================================================================= %!
                
                if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]
                        
                    if (setup%sparse_storage) then
                    
                        prcp = input_data%sparse_prcp(k, t)
                        pet = input_data%sparse_pet(k, t)
                    
                    else
                    
                        prcp = input_data%prcp(row, col, t)
                        pet = input_data%pet(row, col, t)
                    
                    end if
                    
                    if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]
                
                        !% =============================================================================================== %!
                        !%   Interception module case [ 0 - 1 ]
                        !% =============================================================================================== %!
                    
                        select case(setup%interception_module)
                        
                        case(0)
                        
                            ei = min(pet, prcp)
                            
                            pn = max(0._sp, prcp - ei)
                            
                        case(1)

                            call GR_interception(prcp, pet, parameters%ci(row, col), states%hi(row, col), pn, ei)
                        
                        end select
                        
                        en = pet - ei
                        
                        !% =============================================================================================== %!
                        !%   Production module case [ 0 ]
                        !% =============================================================================================== %!
                        
                        select case(setup%production_module)

                        case(0)
                        
                            call GR_production(pn, en, parameters%cp(row, col), parameters%beta(row, col), &
                            & states%hp(row, col), pr, perc)
                            
                        end select
                            
                        !% =============================================================================================== %!
                        !%   Exchange module case [ 0 - 1 ]
                        !% =============================================================================================== %!
                    
                        select case(setup%exchange_module)
                        
                        case(0)
                        
                            l = 0._sp
                            
                        case(1)
                        
                            call GR_exchange(parameters%exc(row, col), states%hft(row, col), l)
                        
                        end select
                        
                    end if !% [ END IF PRCP GAP ]
                    
                    !% =================================================================================================== %!
                    !%   Transfer module case [ 0 ]
                    !% =================================================================================================== %!
                    
                    select case(setup%transfer_module)
                    
                    case(0)
                    
                        prr = parameters%alpha(row, col) * (pr + perc) + l
                        prd = (1._sp - parameters%alpha(row, col)) * (pr + perc)
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        qd = max(0._sp, prd + l)
                        
                        
                    case(1)
                    
                        prr = 0.9_sp * parameters%alpha(row, col) * (pr + perc) + l
                        prl = 0.9_sp * (1._sp - parameters%alpha(row, col)) * (pr + perc)
                        prd = 0.1_sp * (pr + perc)
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        call GR_transferN(5._sp, prcp, prl, parameters%cst(row, col), states%hst(row, col), ql)
                        
                        qd = max(0._sp, prd + l)

                    end select
                    
                    qt = (qd + qr + ql)
                    
                    !% =================================================================================================== %!
                    !%   Routing module case [ 0 - 1 ]
                    !% =================================================================================================== %!
                    
                    select case(setup%routing_module)
                
                    case(0)
                    
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, mesh%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flwdir, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                        
                            sparse_q(k) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flwdir, mesh%drained_area, row, col, q, qup)
                        
                            q(row, col) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt
                    
                        end if
                        
                    case(1)
                        
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, mesh%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flwdir, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                            sparse_q(k) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flwdir, mesh%drained_area, row, col, q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)
                        
                            q(row, col) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * mesh%dx * mesh%dx * 0.001_sp / setup%dt
                    
                        end if
                
                    end select
                
                end if !% [ END IF ACTIVE CELL ]
                
            end if !% [ END IF PATH ]
        
        end do !% [ END DO SPACE ]
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge at gauge
        !% =============================================================================================================== %!
        
        do g=1, mesh%ng
        
            row = mesh%gauge_pos(g, 1)
            col = mesh%gauge_pos(g, 2)
            
            if (setup%sparse_storage) then
                
                k = mesh%rowcol_to_ind_sparse(row, col)
                
                output%qsim(g, t) = sparse_q(k)

            else
            
                output%qsim(g, t) = q(row, col)
            
            end if
        
        end do
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge on domain (optional)
        !% =============================================================================================================== %!
        
        if (setup%save_qsim_domain) then
        
            if (setup%sparse_storage) then
            
                output%sparse_qsim_domain(:, t) = sparse_q
                
            else
            
                output%qsim_domain(:, :, t) = q
            
            end if
        
        end if
        
        !% =============================================================================================================== %!
        !%   Store simulated net rainfall on domain (optional)
        !%   The net rainfall over a surface is a fictitious quantity that corresponds to 
        !%   the part of the rainfall water depth that actually causes runoff. 
        !% =============================================================================================================== %!
        
        if (setup%save_net_prcp_domain) then
        
            if (setup%sparse_storage) then
            
                output%sparse_net_prcp_domain(:, t) = qt
                
            else
            
                output%net_prcp_domain(:, :, t) = qt
            
            end if
        
        end if
        
    end do !% [ END DO TIME ]
    
    !% =============================================================================================================== %!
    !%   Store states at final time step
    !% =============================================================================================================== %!
    
    output%fstates = states
    
    !% =================================================================================================================== %!
    !%   Compute J
    !% =================================================================================================================== %!
    
    call hyper_compute_cost(setup, mesh, input_data, hyper_parameters, &
    & hyper_parameters_bgd, hyper_states, hyper_states_bgd, output, cost)

end subroutine hyper_forward
