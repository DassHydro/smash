!%    This module (wrap) `mw_run` encapsulates all SMASH run (type, subroutines, functions)
module mw_run
    
    use m_common, only: sp, dp, lchar, np, np
    use mw_setup, only: SetupDT
    use mw_mesh, only: MeshDT
    use mw_input_data, only: Input_DataDT
    use mw_parameters, only: ParametersDT
    use mw_states, only: StatesDT
    use mw_output, only: OutputDT
    
    use m_operator, only: GR_interception, GR_production, GR_exchange, &
    & GR_transferN, upstream_discharge, sparse_upstream_discharge
    use mw_cost, only: compute_jobs
    
!~     use :: omp_lib
    
    implicit none
    
    contains
    
        subroutine direct_model(setup, mesh, input_data, parameters, states, output, cost)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(in) :: input_data
            type(ParametersDT), intent(in) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(out) :: cost
            
            integer :: t, i, row, col, k, g, row_g, col_g, k_g
            real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
            & qd, qr, ql, qt, qup
            
!~             real(sp) :: jobs
            
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
            
        end subroutine direct_model
      
!~         subroutine adjoint_model()
        
!~         end subroutine adjoint_model
        
!~         subroutine tangent_linear_model()
        
!~         end subroutine tangent_linear_model

end module mw_run
