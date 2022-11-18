!%      This module `mwd_states` encapsulates all SMASH states.
!%      This module is wrapped and differentiated.
!%
!%      StatesDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``hi``                   Interception state       [-]   (default: 0.01)   ]0, 1[
!%      ``hp``                   Production state         [-]   (default: 0.01)   ]0, 1[
!%      ``hft``                  Fast transfer state      [-]   (default: 0.01)   ]0, 1[
!%      ``hst``                  Slow transfer state      [-]   (default: 0.01)   ]0, 1[
!%
!%      ``husl1``                Upper soil layer 1 state [-]   (default: 0.01)   ]0, 1[
!%      ``husl2``                Upper soil layer 2 state [-]   (default: 0.01)   ]0, 1[
!%      ``hlsl``                 Lower soil layer state   [-]   (default: 0.01)   ]0, 1[
!%
!%      ``hlr``                  Linear routing state     [mm]  (default: 1e-6)   ]0, +Inf[
!%      ======================== =======================================
!%
!%      Hyper_StatesDT type:
!%      
!%      It contains same States variables (see above)
!%
!%      contains
!%
!%      [1] StatesDT_initialise
!%      [2] Hyper_StatesDT_initialise
!%      [3] states_to_matrix
!%      [4] matrix_to_states
!%      [5] vector_to_states
!%      [6] set0_states
!%      [7] set1_states
!%      [8]  hyper_states_to_matrix
!%      [10] matrix_to_hyper_states
!%      [11] set0_hyper_states
!%      [12] set1_hyper_states
!%      [13] hyper_states_to_states

module mwd_states

    use md_constant !% only: sp, ns
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    
    implicit none

    type StatesDT
    
        !% Notes
        !% -----
        !% StatesDT Derived Type.
        
        ! GR
        real(sp), dimension(:,:), allocatable :: hi
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hst
        
        ! VIC
        real(sp), dimension(:,:), allocatable :: husl1
        real(sp), dimension(:,:), allocatable :: husl2
        real(sp), dimension(:,:), allocatable :: hlsl
        
        ! Routing
        real(sp), dimension(:,:), allocatable :: hlr
        
    end type StatesDT
    
    type Hyper_StatesDT
    
        !% Notes
        !% -----
        !% Hyper_StatesDT Derived Type.
    
        ! GR
        real(sp), dimension(:,:), allocatable :: hi
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hst
        
        ! VIC
        real(sp), dimension(:,:), allocatable :: husl1
        real(sp), dimension(:,:), allocatable :: husl2
        real(sp), dimension(:,:), allocatable :: hlsl
        
        ! Routing
        real(sp), dimension(:,:), allocatable :: hlr
    
    end type Hyper_StatesDT
    
    contains
        
        subroutine StatesDT_initialise(this, mesh)
        
            !% Notes
            !% -----
            !% StatesDT initialisation subroutine.
        
            implicit none
            
            type(StatesDT), intent(inout) :: this
            type(MeshDT), intent(in) :: mesh
            
            allocate(this%hi(mesh%nrow, mesh%ncol))
            allocate(this%hp(mesh%nrow, mesh%ncol))
            allocate(this%hft(mesh%nrow, mesh%ncol))
            allocate(this%hst(mesh%nrow, mesh%ncol))
            
            allocate(this%husl1(mesh%nrow, mesh%ncol))
            allocate(this%husl2(mesh%nrow, mesh%ncol))
            allocate(this%hlsl(mesh%nrow, mesh%ncol))
            
            allocate(this%hlr(mesh%nrow, mesh%ncol))
            
            this%hi  = 0.01_sp
            this%hp  = 0.01_sp
            this%hft = 0.01_sp
            this%hst = 0.01_sp
            
            this%husl1 = 0.01_sp
            this%husl2 = 0.01_sp
            this%hlsl = 0.01_sp
            
            this%hlr = 0.000001_sp
            
        end subroutine StatesDT_initialise
        
        
        subroutine Hyper_StatesDT_initialise(this, setup)
        
            !% Notes
            !% -----
            !% Hyper_StatesDT initialisation subroutine.
        
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: this
            type(SetupDT), intent(in) :: setup
            
            allocate(this%hi(setup%optimize%nhyper, 1))
            allocate(this%hp(setup%optimize%nhyper, 1))
            allocate(this%hft(setup%optimize%nhyper, 1))
            allocate(this%hst(setup%optimize%nhyper, 1))
            
            allocate(this%husl1(setup%optimize%nhyper, 1))
            allocate(this%husl2(setup%optimize%nhyper, 1))
            allocate(this%hlsl(setup%optimize%nhyper, 1))
            
            allocate(this%hlr(setup%optimize%nhyper, 1))
 
        end subroutine Hyper_StatesDT_initialise

end module mwd_states
