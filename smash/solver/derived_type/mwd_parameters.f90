!%      This module `mwd_parameters` encapsulates all SMASH parameters.
!%      This module is wrapped and differentiated.
!%
!%      ParametersDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``ci``                   GR interception parameter          [mm]    (default: 1)     ]0, +Inf[
!%      ``cp``                   GR production parameter            [mm]    (default: 200)   ]0, +Inf[
!%      ``beta``                 GR percolation parameter           [-]     (default: 1000)  ]0, +Inf[
!%      ``cft``                  GR fast transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``cst``                  GR slow transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``alpha``                GR transfer partitioning parameter [-]     (default: 0.9)   ]0, 1[
!%      ``exc``                  GR exchange parameter              [mm/dt] (default: 0)     ]-Inf, +Inf[
!%
!%      ``b``                    VIC infiltration parameter         [-]     (default: 0.3)   ]0, +Inf[
!%      ``cusl1``                VIC upper soil layer 1 capacity    [mm]    (default: 100)   ]0, +Inf[
!%      ``cusl2``                VIC upper soil layer 2 capacity    [mm]    (default: 500)   ]0, +Inf[
!%      ``clsl``                 VIC lower soil layer capacity      [mm]    (default: 2000)  ]0, +Inf[
!%      ``ks``                   VIC sat. hydraulic conductivity    [mm/dt] (default: 20)    ]0, +Inf[
!%      ``ds``                   VIC baseflow ds parameter          [-]     (default: 0.02)  ]0, 1[
!%      ``dsm``                  VIC baseflow max discharge         [mm/dt] (default: 0.33)  ]0, +Inf[
!%      ``ws``                   VIC baseflow linear threshold      [-]     (default: 0.8)  ]0, 1[
!%
!%      ``lr``                   Linear routing parameter           [min]   (default: 5)     ]0, +Inf[
!%      ======================== =======================================
!%      
!%      Hyper_ParametersDT type:
!%      
!%      It contains same Parameters variables (see above)
!%
!%      contains
!%
!%      [1]  ParametersDT_initialise
!%      [2]  Hyper_ParametersDT_initialise
!%      [3]  parameters_to_matrix
!%      [4]  matrix_to_parameters
!%      [5]  vector_to_parameters
!%      [6]  set0_parameters
!%      [7]  set1_parameters
!%      [8]  hyper_parameters_to_matrix
!%      [10] matrix_to_hyper_parameters
!%      [11] set0_hyper_parameters
!%      [12] set1_hyper_parameters
!%      [13] hyper_parameters_to_parameters

module mwd_parameters

    use md_constant !% only: sp, GNP
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    
    implicit none
    
    type ParametersDT
    
        !% Notes
        !% -----
        !% ParametersDT Derived Type.
        
        ! GR
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        
        ! VIC
        real(sp), dimension(:,:), allocatable :: b
        real(sp), dimension(:,:), allocatable :: cusl1
        real(sp), dimension(:,:), allocatable :: cusl2
        real(sp), dimension(:,:), allocatable :: clsl
        real(sp), dimension(:,:), allocatable :: ks
        real(sp), dimension(:,:), allocatable :: ds
        real(sp), dimension(:,:), allocatable :: dsm
        real(sp), dimension(:,:), allocatable :: ws
        
        ! Routing
        real(sp), dimension(:,:), allocatable :: lr
        
    end type ParametersDT
    
    type Hyper_ParametersDT
    
        !% Notes
        !% -----
        !% Hyper_ParametersDT Derived Type.
    
        ! GR
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        
        ! VIC
        real(sp), dimension(:,:), allocatable :: b
        real(sp), dimension(:,:), allocatable :: cusl1
        real(sp), dimension(:,:), allocatable :: cusl2
        real(sp), dimension(:,:), allocatable :: clsl
        real(sp), dimension(:,:), allocatable :: ks
        real(sp), dimension(:,:), allocatable :: ds
        real(sp), dimension(:,:), allocatable :: dsm
        real(sp), dimension(:,:), allocatable :: ws
        
        ! Routing
        real(sp), dimension(:,:), allocatable :: lr
    
    end type Hyper_ParametersDT
    
    contains
        
        subroutine ParametersDT_initialise(this, mesh)
        
            !% Notes
            !% -----
            !% ParametersDT initialisation subroutine.
        
            implicit none
            
            type(ParametersDT), intent(inout) :: this
            type(MeshDT), intent(in) :: mesh
            
            allocate(this%ci(mesh%nrow, mesh%ncol))
            allocate(this%cp(mesh%nrow, mesh%ncol))
            allocate(this%beta(mesh%nrow, mesh%ncol))
            allocate(this%cft(mesh%nrow, mesh%ncol))
            allocate(this%cst(mesh%nrow, mesh%ncol))
            allocate(this%alpha(mesh%nrow, mesh%ncol))
            allocate(this%exc(mesh%nrow, mesh%ncol))
            
            allocate(this%b(mesh%nrow, mesh%ncol))
            allocate(this%cusl1(mesh%nrow, mesh%ncol))
            allocate(this%cusl2(mesh%nrow, mesh%ncol))
            allocate(this%clsl(mesh%nrow, mesh%ncol))
            allocate(this%ks(mesh%nrow, mesh%ncol))
            allocate(this%ds(mesh%nrow, mesh%ncol))
            allocate(this%dsm(mesh%nrow, mesh%ncol))
            allocate(this%ws(mesh%nrow, mesh%ncol))
            
            allocate(this%lr(mesh%nrow, mesh%ncol))
            
            this%ci    = 1._sp
            this%cp    = 200._sp
            this%beta  = 1000._sp
            this%cft   = 500._sp
            this%cst   = 500._sp
            this%alpha = 0.9_sp
            this%exc   = 0._sp
            
            this%b     = 0.3_sp
            this%cusl1 = 100._sp
            this%cusl2 = 500._sp
            this%ks    = 20._sp
            this%clsl  = 2000._sp
            this%ds    = 0.02_sp
            this%dsm   = 0.33_sp
            this%ws    = 0.8_sp
            
            this%lr    = 5._sp
 
        end subroutine ParametersDT_initialise
        
        
        subroutine Hyper_ParametersDT_initialise(this, setup)
        
            !% Notes
            !% -----
            !% Hyper_ParametersDT initialisation subroutine.
        
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: this
            type(SetupDT), intent(in) :: setup
            
            
            allocate(this%ci(setup%optimize%nhyper, 1))
            allocate(this%cp(setup%optimize%nhyper, 1))
            allocate(this%beta(setup%optimize%nhyper, 1))
            allocate(this%cft(setup%optimize%nhyper, 1))
            allocate(this%cst(setup%optimize%nhyper, 1))
            allocate(this%alpha(setup%optimize%nhyper, 1))
            allocate(this%exc(setup%optimize%nhyper, 1))
            
            allocate(this%b(setup%optimize%nhyper, 1))
            allocate(this%cusl1(setup%optimize%nhyper, 1))
            allocate(this%cusl2(setup%optimize%nhyper, 1))
            allocate(this%clsl(setup%optimize%nhyper, 1))
            allocate(this%ks(setup%optimize%nhyper, 1))
            allocate(this%ds(setup%optimize%nhyper, 1))
            allocate(this%dsm(setup%optimize%nhyper, 1))
            allocate(this%ws(setup%optimize%nhyper, 1))
            
            allocate(this%lr(setup%optimize%nhyper, 1))
 
        end subroutine Hyper_ParametersDT_initialise


end module mwd_parameters
