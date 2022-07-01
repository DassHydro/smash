subroutine forward(setup, mesh, input_data, parameters, states, output, cost)
    
    use md_common !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT, parameters_derived_type_to_matrix
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    
    use md_operator !% only: GR_interception, GR_production, GR_exchange, &
    !% & GR_transferN, upstream_discharge, sparse_upstream_discharge, GR_transfer1
    use mwd_cost !% only: compute_jobs

    implicit none

    type(SetupDT), intent(in) :: setup
    type(MeshDT), intent(in) :: mesh
    type(Input_DataDT), intent(in) :: input_data
    type(ParametersDT), intent(in) :: parameters
    type(StatesDT), intent(inout) :: states
    type(OutputDT), intent(inout) :: output
    real(sp), intent(inout) :: cost
    
    integer :: t, i, row, col, k, g, row_g, col_g, k_g
    real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
    & qd, qr, ql, qt, qup, qrout
    
    real(sp), dimension(:,:,:), allocatable :: q
    real(sp), dimension(:,:), allocatable :: sparse_q
    
    if (setup%sparse_storage) then
    
        allocate(sparse_q(mesh%nac, setup%ntime_step))
    
    else
        
        allocate(q(mesh%nrow, mesh%ncol, setup%ntime_step))
    
    end if

    cost = 0._sp
    
    do t=1, setup%ntime_step
    
        do i=1, mesh%nrow * mesh%ncol
        
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
            
            if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then
            
                row = mesh%path(1, i)
                col = mesh%path(2, i)
                if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)
                
                if (mesh%global_active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then
                        
                    if (setup%sparse_storage) then
                    
                        prcp = input_data%sparse_prcp(k, t)
                        pet = input_data%sparse_pet(k, t)
                    
                    else
                    
                        prcp = input_data%prcp(row, col, t)
                        pet = input_data%pet(row, col, t)
                    
                    end if
                    
                    if (prcp .ge. 0) then
                
!% -------------------------------- Interception module case
                        select case(setup%interception_module)
                        
                        case(0)
                        
                            ei = min(pet, prcp)
                            
                            pn = max(0._sp, prcp - ei)
                            
                        case(1)

                            call GR_interception(prcp, pet, parameters%ci(row, col), states%hi(row, col), pn, ei)
                        
                        end select
                        
                        en = pet - ei
                        
                        select case(setup%production_module)
                        
!% -------------------------------- Production module case
                        case(0)
                        
                            call GR_production(pn, en, parameters%cp(row, col), parameters%beta(row, col), &
                            & states%hp(row, col), pr, perc)
                            
                        end select

!% -------------------------------- Exchange module case
                        select case(setup%exchange_module)
                        
                        case(0)
                        
                            l = 0._sp
                            
                        case(1)
                        
                            call GR_exchange(parameters%exc(row, col), states%hft(row, col), l)
                        
                        end select
                        
                    end if !% {end if: prcp ge 0}
                    
!% ---------------------------- Transfer module case
                    select case(setup%transfer_module)
                    
                    case(0)
                    
                        prr = parameters%alpha(row, col) * (pr + perc) + l
                        prd = (pr + perc) - prr
                        
                        call GR_transferN(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)
                        
                        qd = max(0._sp, prd + l)

                    end select
                    
                    qt = (qd + qr + ql)
                    
!% ------------------------ Routing module case
                    select case(setup%routing_module)
                
                    case(0)
                    
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, setup%dx, setup%ntime_step, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flow, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, t, sparse_q, qup)
                        
                            sparse_q(k, t) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, setup%dx, setup%ntime_step, mesh%nrow,&
                            &  mesh%ncol, mesh%flow, mesh%drained_area, row, col, t, q, qup)
                        
                            q(row, col, t) = (qt + qup * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt
                    
                        end if
                        
                    case(1)
                        
                        if (setup%sparse_storage) then
                    
                            call sparse_upstream_discharge(setup%dt, setup%dx, setup%ntime_step, &
                            & mesh%nrow, mesh%ncol, mesh%nac, mesh%flow, mesh%drained_area, &
                            & mesh%rowcol_to_ind_sparse, row, col, t, sparse_q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hr(row, col), qrout)

                            sparse_q(k, t) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt

                        else

                            call upstream_discharge(setup%dt, setup%dx, setup%ntime_step, mesh%nrow,&
                            &  mesh%ncol, mesh%flow, mesh%drained_area, row, col, t, q, qup)
                            
                            call GR_transfer1(setup%dt, qup, parameters%lr(row, col), states%hr(row, col), qrout)
                        
                            q(row, col, t) = (qt + qrout * real(mesh%drained_area(row, col) - 1))&
                            & * setup%dx * setup%dx * 0.001_sp / setup%dt
                    
                        end if
                
                    end select
                
                end if !% {end if: global active cell}
                
            end if !% {end if: path}
        
        end do !% {end do: space}
        
    end do !% {end do: time}
    
    do g=1, mesh%ng
    
        row_g = mesh%gauge_pos(1, g)
        col_g = mesh%gauge_pos(2, g)
        
        if (setup%sparse_storage) then
            
            k_g = mesh%rowcol_to_ind_sparse(row_g, col_g)
            
            output%qsim(g, :) = sparse_q(k_g, :)
            
        
        else
        
            output%qsim(g, :) = q(row_g, col_g, :)
        
        end if
    
    end do
    
    call compute_jobs(setup, mesh, input_data, output, cost)
    
end subroutine forward
