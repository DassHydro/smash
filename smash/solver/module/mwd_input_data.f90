!%      This module `mwd_input_data` encapsulates all SMASH input_data.
!%      This module is wrapped and differentiated.
!%
!%      Input_DataDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``qobs``                 Oberserved discharge at gauge               [m3/s]
!%      ``prcp``                 Precipitation field                         [mm]
!%      ``pet``                  Potential evapotranspiration field          [mm]
!%      ``descriptor``           Descriptor map(s) field                     [(descriptor dependent)]
!%      ``sparse_prcp``          Sparse precipitation field                  [mm] 
!%      ``sparse_pet``           Spase potential evapotranspiration field    [mm]
!%      ``mean_prcp``            Mean precipitation at gauge                 [mm]
!%      ``mean_pet``             Mean potential evapotranspiration at gauge  [mm]
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] Input_DataDT_initialise
!%      [2] input_data_copy
!%      [3] compute_mean_forcing

module mwd_input_data

    use mwd_common !% only: sp, dp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    
    implicit none

    type Input_DataDT
    
        real(sp), dimension(:,:), allocatable :: qobs
        real(sp), dimension(:,:,:), allocatable :: prcp
        real(sp), dimension(:,:,:), allocatable :: pet
        
        real(sp), dimension(:,:,:), allocatable :: descriptor
        
        real(sp), dimension(:,:), allocatable :: sparse_prcp
        real(sp), dimension(:,:), allocatable :: sparse_pet
        
        real(sp), dimension(:,:), allocatable :: mean_prcp
        real(sp), dimension(:,:), allocatable :: mean_pet
    
    end type Input_DataDT
    
    contains
    
        subroutine Input_DataDT_initialise(input_data, setup, mesh)
        
            implicit none
            
            type(Input_DataDT), intent(inout) :: input_data
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            
            if (mesh%ng .gt. 0) then

                allocate(input_data%qobs(mesh%ng, setup%ntime_step))
                input_data%qobs = -99._sp
                
            end if
            
            if (setup%sparse_storage) then
            
                allocate(input_data%sparse_prcp(mesh%nac, &
                & setup%ntime_step))
                input_data%sparse_prcp = -99._sp
                allocate(input_data%sparse_pet(mesh%nac, &
                & setup%ntime_step))
                input_data%sparse_pet = -99._sp
                
            else
            
                allocate(input_data%prcp(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%prcp = -99._sp
                allocate(input_data%pet(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%pet = -99._sp
            
            end if
            
            if (setup%nd .gt. 0) then
            
                allocate(input_data%descriptor(mesh%nrow, mesh%ncol, setup%nd))
                
            end if
            
            if (setup%mean_forcing) then
            
                allocate(input_data%mean_prcp(mesh%ng, &
                & setup%ntime_step))
                input_data%mean_prcp = -99._sp
                allocate(input_data%mean_pet(mesh%ng, setup%ntime_step))
                input_data%mean_pet = -99._sp
                
            end if
            
        end subroutine Input_DataDT_initialise
        
        
!%      TODO comment        
        subroutine input_data_copy(input_data_in, &
        & input_data_out)
            
            implicit none
            
            type(Input_DataDT), intent(in) :: input_data_in
            type(Input_DataDT), intent(out) :: input_data_out
            
            input_data_out = input_data_in
        
        end subroutine input_data_copy
      
      
!%      TODO comment 
        subroutine compute_mean_forcing(setup, mesh, input_data)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            
            integer, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: &
            & mask_gauge
            real(sp), dimension(mesh%ng) :: cml_prcp, cml_pet
            integer :: t, col, row, g, k, n, i
            
            mask_gauge = 0
            
            do g=1, mesh%ng
            
                call mask_upstream_cells(mesh%gauge_pos(1, g), &
                & mesh%gauge_pos(2, g), mesh, mask_gauge(:, : ,g))
            
            end do

            do t=1, setup%ntime_step
            
                k = 0
                cml_prcp = 0._sp
                cml_pet = 0._sp
                
                do i=1, mesh%nrow * mesh%ncol
                
                    if (mesh%path(1, i) .gt. 0 .and. &
                    & mesh%path(2, i) .gt. 0) then
                    
                        row = mesh%path(1, i)
                        col = mesh%path(2, i)
                        
                        if (mesh%active_cell(row, col) .eq. &
                        & 1) then
                        
                            k = k + 1
                            
                            do g=1, mesh%ng
                                
                                if (mask_gauge(row, col, g) .eq. 1) then
                                
                                    if (setup%sparse_storage) then
                                        
                                        cml_prcp(g) = cml_prcp(g) + &
                                        & input_data%sparse_prcp(k, t)
                                        cml_pet(g) = cml_pet(g) + &
                                        & input_data%sparse_pet(k, t)
                                        
                                    else
                                    
                                        cml_prcp(g) = cml_prcp(g) + &
                                        & input_data%prcp(row, col, t)
                                        cml_pet(g) = cml_pet(g) + &
                                        & input_data%pet(row, col, t)
                                    
                                    end if
                                    
                                end if
                            
                            end do
                        
                        end if
                        
                    end if
                    
                end do
                    
                do g=1, mesh%ng
            
                    n = count(mask_gauge(:, :, g) .eq. 1)
                    
                    input_data%mean_prcp(g, t) = cml_prcp(g) / n
                    input_data%mean_pet(g, t) = cml_pet(g) / n
            
                end do
                
            end do
        
        end subroutine compute_mean_forcing
    
end module mwd_input_data
