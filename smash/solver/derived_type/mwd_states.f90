!%      This module `mwd_states` encapsulates all SMASH states.
!%      This module is wrapped and differentiated.
!%
!%      StatesDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``hi``                   Interception state    [-]   (default: 0.01)   ]0, 1[
!%      ``hp``                   Production state      [-]   (default: 0.01)   ]0, 1[
!%      ``hft``                  Fast transfer state   [-]   (default: 0.01)   ]0, 1[
!%      ``hst``                  Slow transfer state   [-]   (default: 0.01)   ]0, 1[
!%      ``hlr``                  Linear routing state  [mm]  (default: 0.01)   ]0, +Inf[
!%      ======================== =======================================
!%
!%      Hyper_StatesDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``hi``                   Interception state    [-]   (default: 0.01)   ]0, 1[
!%      ``hp``                   Production state      [-]   (default: 0.01)   ]0, 1[
!%      ``hft``                  Fast transfer state   [-]   (default: 0.01)   ]0, 1[
!%      ``hst``                  Slow transfer state   [-]   (default: 0.01)   ]0, 1[
!%      ``hlr``                  Linear routing state  [mm]  (default: 0.01)   ]0, +Inf[
!%      ======================== =======================================
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

    use md_common !% only: sp, ns
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    
    implicit none

    type StatesDT
        
        real(sp), dimension(:,:), allocatable :: hi
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hst
        real(sp), dimension(:,:), allocatable :: hlr
        
    end type StatesDT
    
    type Hyper_StatesDT
    
        real(sp), dimension(:,:), allocatable :: hi
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hst
        real(sp), dimension(:,:), allocatable :: hlr
    
    end type Hyper_StatesDT
    
    contains
        
        subroutine StatesDT_initialise(states, mesh)
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            type(StatesDT), intent(inout) :: states
            
            integer :: nrow, ncol
            
            nrow = mesh%nrow
            ncol = mesh%ncol
            
            allocate(states%hi(nrow, ncol))
            allocate(states%hp(nrow, ncol))
            allocate(states%hft(nrow, ncol))
            allocate(states%hst(nrow, ncol))
            allocate(states%hlr(nrow, ncol))
            
            states%hi  = 0.01_sp
            states%hp  = 0.01_sp
            states%hft = 0.01_sp
            states%hst = 0.01_sp
            states%hlr = 0.000001_sp
            
        end subroutine StatesDT_initialise
        
        
        subroutine Hyper_StatesDT_initialise(hyper_states, setup)
        
            !% Notes
            !% -----
            !%
            !% Hyper_StatesDT initialisation subroutine
        
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            type(SetupDT), intent(in) :: setup
            
            integer :: n
            
            select case(trim(setup%mapping))
            
            case("hyper-linear")
            
                n = (1 + setup%nd)
                
            case("hyper-polynomial")
            
                n = (1 + 2 * setup%nd)
                
            end select
            
            allocate(hyper_states%hi(n, 1))
            allocate(hyper_states%hp(n, 1))
            allocate(hyper_states%hft(n, 1))
            allocate(hyper_states%hst(n, 1))
            allocate(hyper_states%hlr(n, 1))
 
        end subroutine Hyper_StatesDT_initialise

end module mwd_states
