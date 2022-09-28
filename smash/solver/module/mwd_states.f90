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
!%      contains
!%
!%      [1] StatesDT_initialise
!%      [2] states_to_matrix
!%      [3] matrix_to_states
!%      [4] vector_to_states
!%      [5] set0_states
!%      [6] set1_states

module mwd_states

    use mwd_common !% only: sp, ns
    use mwd_mesh  !% only: MeshDT
    
    implicit none

    type StatesDT
        
        real(sp), dimension(:,:), allocatable :: hi
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hst
        real(sp), dimension(:,:), allocatable :: hlr
        
    end type StatesDT
    
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
        
        
!%      TODO comment 
        subroutine states_to_matrix(states, matrix)
        
            implicit none
            
            type(StatesDT), intent(in) :: states
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2), ns), intent(inout) :: matrix
            
            matrix(:,:,1) = states%hi(:,:)
            matrix(:,:,2) = states%hp(:,:)
            matrix(:,:,3) = states%hft(:,:)
            matrix(:,:,4) = states%hst(:,:)
            matrix(:,:,5) = states%hlr(:,:)
        
        end subroutine states_to_matrix
        
        
!%      TODO comment
        subroutine matrix_to_states(matrix, states)
            
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2), ns), intent(in) :: matrix
            
            states%hi(:,:) = matrix(:,:,1)
            states%hp(:,:) = matrix(:,:,2)
            states%hft(:,:) = matrix(:,:,3)
            states%hst(:,:) = matrix(:,:,4)
            states%hlr(:,:) = matrix(:,:,5)
        
        end subroutine matrix_to_states
        
        
!%      TODO comment
        subroutine vector_to_states(vector, states)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(ns), intent(in) :: vector
            
            states%hi = vector(1)
            states%hp = vector(2)
            states%hft = vector(3)
            states%hst = vector(4)
            states%hlr = vector(5)
        
        end subroutine vector_to_states
        
        
!%      TODO comment        
        subroutine set0_states(states)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            
            real(sp), dimension(ns) :: vector0
            
            vector0 = 0._sp
            
            call vector_to_states(vector0, states)
        
        end subroutine set0_states

        
!%      TODO comment        
        subroutine set1_states(states)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            
            real(sp), dimension(ns) :: vector1
            
            vector1 = 1._sp
            
            call vector_to_states(vector1, states)
        
        end subroutine set1_states

end module mwd_states
