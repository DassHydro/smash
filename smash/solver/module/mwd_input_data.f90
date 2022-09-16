!%      This module `mwd_input_data` encapsulates all SMASH input_data.
!%      This module is wrapped and differentiated.
!%
!%      Prcp_IndiceDT type: DOI: doi:10.5194/hess-15-3767-2011 (Zoccatelli, 2011)
!%      
!%      md1 = p1 / (p0 * g1)
!%      md2 = (1 / (g2 - g1**2)) * ((p2 / p0) - (p1 / p0)**2)
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``p0``                   The 0-th spatial moment of catchment prcp  [mm/dt]
!%      ``p1``                   The 1-th spatial moment of catchment prpc  [mm2/dt]
!%      ``p2``                   The 2-th spatial moment of catchment prpc  [mm3/dt]
!%      ``g1``                   The 1-th spatial moment of flow distance   [mm]
!%      ``g2``                   The 2-th spatial moment of flow distance   [mm2]
!%      ``md1``                  The 1-th scaled moment                     [-]
!%      ``md2``                  The 2-th scaled moment                     [-]
!%      ======================== =======================================
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
!%      [1] Prcp_IndiceDT_initialise
!%      [1] Input_DataDT_initialise
!%      [2] input_data_copy
!%      [3] compute_mean_forcing
!%      [4] compute_prcp_indice

module mwd_input_data

    use mwd_common !% only: sp, dp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT, mask_gauge
    
    implicit none
    
    type Prcp_IndiceDT
    
        real(sp), dimension(:,:), allocatable :: p0
        real(sp), dimension(:,:), allocatable :: p1
        real(sp), dimension(:,:), allocatable :: p2
        
        real(sp), dimension(:), allocatable :: g1
        real(sp), dimension(:), allocatable :: g2
        
        real(sp), dimension(:,:), allocatable :: md1
        real(sp), dimension(:,:), allocatable :: md2
        
        real(sp), dimension(:,:), allocatable :: std
    
    end type Prcp_IndiceDT
    
    
    type Input_DataDT
    
        real(sp), dimension(:,:), allocatable :: qobs
        real(sp), dimension(:,:,:), allocatable :: prcp
        real(sp), dimension(:,:,:), allocatable :: pet
        
        real(sp), dimension(:,:,:), allocatable :: descriptor
        
        real(sp), dimension(:,:), allocatable :: sparse_prcp
        real(sp), dimension(:,:), allocatable :: sparse_pet
        
        real(sp), dimension(:,:), allocatable :: mean_prcp
        real(sp), dimension(:,:), allocatable :: mean_pet

        type(Prcp_IndiceDT) :: prcp_indice
    
    end type Input_DataDT
    
    
    contains
    
        subroutine Prcp_IndiceDT_initialise(prcp_indice, setup, mesh)
            
            implicit none
            
            type(Prcp_IndiceDT) :: prcp_indice
            type(SetupDT) :: setup
            type(MeshDT) :: mesh
            
            allocate(prcp_indice%p0(mesh%ng, setup%ntime_step))
            prcp_indice%p0 = -99._sp
            allocate(prcp_indice%p1(mesh%ng, setup%ntime_step))
            prcp_indice%p1 = -99._sp
            allocate(prcp_indice%p2(mesh%ng, setup%ntime_step))
            prcp_indice%p2 = -99._sp
            
            allocate(prcp_indice%g1(mesh%ng))
            prcp_indice%g1 = -99._sp
            allocate(prcp_indice%g2(mesh%ng))
            prcp_indice%g2 = -99._sp
        
            allocate(prcp_indice%md1(mesh%ng, setup%ntime_step))
            prcp_indice%md1 = -99._sp
            allocate(prcp_indice%md2(mesh%ng, setup%ntime_step))
            prcp_indice%md2 = -99._sp
        
            allocate(prcp_indice%std(mesh%ng, setup%ntime_step))
            prcp_indice%std = -99._sp
        
        end subroutine Prcp_IndiceDT_initialise
        
    
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
            
                allocate(input_data%descriptor(mesh%nrow, mesh%ncol, &
                & setup%nd))
                
            end if
            
            if (setup%mean_forcing) then
            
                allocate(input_data%mean_prcp(mesh%ng, &
                & setup%ntime_step))
                input_data%mean_prcp = -99._sp
                allocate(input_data%mean_pet(mesh%ng, setup%ntime_step))
                input_data%mean_pet = -99._sp
                
            end if
            
            if (setup%prcp_indice .and. mesh%ng .gt. 0) then
            
                call Prcp_IndiceDT_initialise(input_data%prcp_indice, &
                & setup, mesh)
            
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
!%      Refactorize this with check for no data inside mean
!%      Allow mean_forcing without gauge
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
        
        
!%      TODO comment
        subroutine compute_prcp_indice(setup, mesh, input_data)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            
            logical, dimension(mesh%nrow, mesh%ncol) :: mask
            integer, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: g3d_mask
            real(sp), dimension(mesh%nrow, mesh%ncol) :: dflwdst
            integer :: i, j
            real(sp) :: minv_n, sum_r, sum_r2, sum_d, sum_d2, sum_rd, &
            & sum_rd2, mean_r, p0, p1, p2, g1, g2, md1, md2, std
            
            call mask_gauge(mesh, g3d_mask)
            
            do i=1, mesh%ng
                
                dflwdst = mesh%flwdst - &
                & mesh%flwdst(mesh%gauge_pos(1, i), mesh%gauge_pos(2, i))
                
                do j=1, setup%ntime_step
                    
                    mask = (input_data%prcp(:,:,j) .ge. 0 .and. g3d_mask(:,:,i) .eq. 1)
                
                    minv_n = 1._sp / count(mask)
                    
                    sum_r = sum(input_data%prcp(:,:,j), mask=mask)
                    
                    !% Do not compute indices if there is no precipitation
                    if (sum_r .gt. 0._sp) then
                        
                        sum_r2 = sum(input_data%prcp(:,:,j) * input_data%prcp(:,:,j), mask=mask)
                        sum_d = sum(dflwdst, mask=mask)
                        sum_d2 = sum(dflwdst * dflwdst, mask=mask)
                        sum_rd = sum(input_data%prcp(:,:,j) * dflwdst, mask=mask)
                        sum_rd2 = sum(input_data%prcp(:,:,j) * dflwdst * dflwdst, mask=mask)
                        
                        mean_r = sum_r * minv_n
                        
                        p0 = minv_n * sum_r
                        input_data%prcp_indice%p0(i, j) = p0
                        
                        p1 = minv_n * sum_rd
                        input_data%prcp_indice%p1(i, j) = p1
                        
                        p2 = minv_n * sum_rd2
                        input_data%prcp_indice%p2(i, j) = p2
                        
                        g1 = minv_n * sum_d
                        input_data%prcp_indice%g1(i) = g1
                        
                        g2 = minv_n * sum_d2
                        input_data%prcp_indice%g2(i) = g2
                        
                        md1 = p1 / (p0 * g1)
                        input_data%prcp_indice%md1(i, j) = md1
                    
                        md2 = (1._sp / (g2 - g1 * g1)) * ((p2 / p0) - (p1 / p0) * (p1 / p0))
                        input_data%prcp_indice%md2(i, j) = md2
                        
                        std = sqrt((minv_n * sum_r2) - (mean_r * mean_r))
                        input_data%prcp_indice%std(i, j) = std
                    
                    end if
                    
                end do 
        
            end do
            
        end subroutine compute_prcp_indice

end module mwd_input_data
