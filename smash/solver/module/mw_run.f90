!%    This module (wrap) `mw_run` encapsulates all SMASH run (type, subroutines, functions)
module mw_run
    
    use m_common, only: sp, dp, lchar, np, np
    use mw_setup, only: SetupDT
    use mw_mesh, only: MeshDT
    use mw_input_data, only: Input_DataDT
    use mw_parameters, only: ParametersDT
    use mw_states, only: StatesDT
    use mw_output, only: OutputDT
    use m_operator, only: GR_production
    
    use, intrinsic :: omp_lib
    
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
            
            integer :: t, i, row, col, k
            real(sp) :: prcp, pet, ei, pth, en, pr, perc
            
            cost = 0._sp
            
            print*, "IN DIRECT MODEL"
!~             27 13 (row, col) exu
            
            do t=1, setup%ntime_step
            
                k = 0
            
                do i=1, mesh%nrow * mesh%ncol
                
                    ei = 0._sp
                    pth = 0._sp
                    en = 0._sp
                    pr = 0._sp
                    perc = 0._sp
                    
                    if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then
                    
                        row = mesh%path(1, i)
                        col = mesh%path(2, i)
                        
                        if (mesh%global_active_cell(row, col) .eq. 1) then
                        
                            k = k + 1
                            
                            if (mesh%local_active_cell(row, col) .eq. 1) then
                                
                                if (setup%sparse_storage) then
                                
                                    prcp = input_data%sparse_prcp(k, t)
                                    pet = input_data%sparse_pet(k, t)
                                
                                else
                                
                                    prcp = input_data%prcp(row, col, t)
                                    pet = input_data%pet(row, col, t)
                                
                                end if
                                
                                ei = min(pet, prcp)
                                
                                pth = max(0._sp, prcp - ei)
                                
                                en = pet - ei
                                
                                call GR_production(pth, en, parameters%cp(row, col), 1000._sp, states%hp(row, col), pr, perc)
                            
                            end if
                        
                        end if
                        
                    end if
                
                end do
                
            end do
            
        end subroutine direct_model
        
!~         subroutine adjoint_model()
        
!~         end subroutine adjoint_model
        
!~         subroutine tangent_linear_model()
        
!~         end subroutine tangent_linear_model

end module mw_run
