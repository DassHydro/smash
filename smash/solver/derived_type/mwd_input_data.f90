!%      This module `mwd_input_data` encapsulates all SMASH input_data.
!%      This module is wrapped and differentiated.
!%
!%      Prcp_IndiceDT type: 
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
!%      ``std``                  Standard deviation of catchment prcp       [mm/dt]
!%      ``wf``                   Width function                             [-]
!%      ``pwf``                  Precipitation width function               [-]
!%      ``vg``                   Vertical gap                               [-]
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
!%      ``prcp_indice``          Precipitation indices (Prcp_IndicesDT)
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] Prcp_IndiceDT_initialise
!%      [2] Input_DataDT_initialise

module mwd_input_data

    use md_common !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    
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
        
        real(sp), dimension(:,:), allocatable :: flwdst_qtl
        
        real(sp), dimension(:,:), allocatable :: wf
        real(sp), dimension(:,:,:), allocatable :: pwf
        
        real(sp), dimension(:,:), allocatable :: vg
    
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
        
            !% Notes
            !% -----
            !%
            !% Prcp_IndiceDT initialisation subroutine
            
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
            
            allocate(prcp_indice%flwdst_qtl(mesh%ng, 11))
            prcp_indice%flwdst_qtl = -99._sp
            
            allocate(prcp_indice%wf(mesh%ng, 11))
            prcp_indice%wf = -99._sp
            allocate(prcp_indice%pwf(mesh%ng, setup%ntime_step, 11))
            prcp_indice%pwf = -99._sp

            allocate(prcp_indice%vg(mesh%ng, setup%ntime_step))
            prcp_indice%vg = -99._sp
            
        
        end subroutine Prcp_IndiceDT_initialise
        
    
        subroutine Input_DataDT_initialise(input_data, setup, mesh)
        
            !% Notes
            !% -----
            !%
            !% Input_DataDT initialisation subroutine
        
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
            
            if (setup%mean_forcing .and. mesh%ng .gt. 0) then
            
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

end module mwd_input_data
