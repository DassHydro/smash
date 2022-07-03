subroutine forward(setup, mesh, input_data, parameters, states, output, cost)

    !% =================================================================================================================== %!
    !%   Module import ('only' commented because of issues in adjoint model)
    !% =================================================================================================================== %!
    
    use mwd_common !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    use md_operator !% only: GR_interception, GR_production, GR_exchange, &
    !% & GR_transferN, upstream_discharge, sparse_upstream_discharge, GR_transfer1
    use mwd_cost !% only: compute_jobs

    implicit none

    !% =================================================================================================================== %!
    !%   Derived Type Variables (shared)
    !% =================================================================================================================== %!

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(in) :: parameters
    type(StatesDT), intent(inout) :: states
    type(OutputDT), intent(inout) :: output
    real(sp), intent(inout) :: cost
    
    !% =================================================================================================================== %!
    !%   Local Variables (private)
    !===================================================================================================================== %!
    
    real(sp), dimension(:,:), allocatable :: q
    real(sp), dimension(:), allocatable :: sparse_q
    real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
    & qd, qr, ql, qt, qup, qrout
    integer :: t, i, row, col, k, g
    
    cost = 0._sp
    
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
        !%   Local Variables Initialization for time step (t) and cell (i)
        !% =============================================================================================================== %!
        
            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
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
                
                if (mesh%global_active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]
                        
                    if (setup%sparse_storage) then
                    
                        prcp = input_data%sparse_prcp(k, t)
                        pet = input_data%sparse_pet(k, t)
                    
                    else
                    
                        prcp = input_data%prcp(row, col, t)
                        pet = input_data%pet(row, col, t)
                    
                    end if
                    
                    if (prcp .ge. 0) then !% [ IF PRCP GAP ]
                
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
                        prd = (pr + perc) - prr
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        qd = max(0._sp, prd + l)

                    end select
                    
                    qt = (qd + qr + ql)
                    
                    !% =================================================================================================== %!
                    !%   Routing module case [ 0 - 1 ]
                    !% =================================================================================================== %!
                    
                    select case(setup%routing_module)
                
                    case(0)
                    
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, setup%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flow, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                        
                            sparse_q(k) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, setup%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flow, mesh%drained_area, row, col, q, qup)
                        
                            q(row, col) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt
                    
                        end if
                        
                    case(1)
                        
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, setup%dx, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flow, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, sparse_q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                            sparse_q(k) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, setup%dx, mesh%nrow,&
                            &  mesh%ncol, mesh%flow, mesh%drained_area, row, col, q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)
                        
                            q(row, col) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt
                    
                        end if
                
                    end select
                
                end if !% [ END IF ACTIVE CELL ]
                
            end if !% [ END IF PATH ]
        
        end do !% [ END DO SPACE ]
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge at gauge
        !% =============================================================================================================== %!
        
        do g=1, mesh%ng
        
            row = mesh%gauge_pos(1, g)
            col = mesh%gauge_pos(2, g)
            
            if (setup%sparse_storage) then
                
                k = mesh%rowcol_to_ind_sparse(row, col)
                
                output%qsim(g, t) = sparse_q(k)

            else
            
                output%qsim(g, t) = q(row, col)
            
            end if
        
        end do
        
        !% =============================================================================================================== %!
        !%   Store simulated discharge at domain (optional)
        !% =============================================================================================================== %!
        
        if (setup%save_qsim_domain) then
        
            if (setup%sparse_storage) then
            
                output%sparse_qsim_domain(:, t) = sparse_q
                
            else
            
                output%qsim_domain(:, :, t) = q
            
            end if
        
        end if
        
    end do !% [ END DO TIME ]
    
    !% =================================================================================================================== %!
    !%   Compute J
    !% =================================================================================================================== %!
    
    call compute_jobs(setup, mesh, input_data, output, cost)
    
end subroutine forward
