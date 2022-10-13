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
        
        
!%      TODO comment
        subroutine hyper_states_to_matrix(hyper_states, matrix)
        
            implicit none
            
            type(Hyper_StatesDT), intent(in) :: hyper_states
            real(sp), dimension(size(hyper_states%hp, 1), &
            & size(hyper_states%hp, 2), ns), intent(inout) :: matrix

            matrix(:,:,1) = hyper_states%hi(:,:)
            matrix(:,:,2) = hyper_states%hp(:,:)
            matrix(:,:,3) = hyper_states%hft(:,:)
            matrix(:,:,4) = hyper_states%hst(:,:)
            matrix(:,:,5) = hyper_states%hlr(:,:)

        end subroutine hyper_states_to_matrix

        
!%      TODO comment
        subroutine matrix_to_hyper_states(matrix, hyper_states)
        
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            real(sp), dimension(size(hyper_states%hp, 1), &
            & size(hyper_states%hp, 2), ns), intent(in) :: matrix
            
            hyper_states%hi(:,:) = matrix(:,:,1)
            hyper_states%hp(:,:) = matrix(:,:,2)
            hyper_states%hft(:,:) = matrix(:,:,3)
            hyper_states%hst(:,:) = matrix(:,:,4)
            hyper_states%hlr(:,:) = matrix(:,:,5)
        
        end subroutine matrix_to_hyper_states
        
        
!%      TODO comment
        subroutine set0_hyper_states(hyper_states)
            
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            
            real(sp), dimension(size(hyper_states%hp, 1), &
            & size(hyper_states%hp, 2), ns) :: matrix
            
            matrix = 0._sp
            
            call matrix_to_hyper_states(matrix, hyper_states)
        
        end subroutine set0_hyper_states
        
        
!%      TODO comment
        subroutine set1_hyper_states(hyper_states)
            
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            
            real(sp), dimension(size(hyper_states%hp, 1), &
            & size(hyper_states%hp, 2), ns) :: matrix
            
            matrix = 1._sp
            
            call matrix_to_hyper_states(matrix, hyper_states)
        
        end subroutine set1_hyper_states
  
        
!%     TODO comment
        subroutine hyper_states_to_states(hyper_states, &
        & states, setup, input_data)
        
            implicit none
            
            type(Hyper_StatesDT), intent(in) :: hyper_states
            type(StatesDT), intent(inout) :: states
            type(SetupDT), intent(in) :: setup
            type(Input_DataDT), intent(in) :: input_data
            
            real(sp), dimension(size(hyper_states%hp, 1), &
            & size(hyper_states%hp, 2), ns) :: hyper_states_matrix
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2), np) :: states_matrix
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2)) :: d, dpb
            integer :: i, j
            real(sp) :: a, b
            
            call hyper_states_to_matrix(hyper_states, hyper_states_matrix)
            call states_to_matrix(states, states_matrix)
            
            !% Add mask later here
            !% 1 in dim2 will be replace with k and apply where on Omega
            do i=1, ns
            
                states_matrix(:,:,i) = hyper_states_matrix(1, 1, i)
                
                do j=1, setup%nd
                
                    d = input_data%descriptor(:,:,j)
            
                    a = hyper_states_matrix(2 * j, 1, i)
                    b = hyper_states_matrix(2 * j + 1, 1, i)
                    dpb = d ** b
                
                    states_matrix(:,:,i) = states_matrix(:,:,i) + a * dpb
            
                end do
                
                where (states_matrix(:,:,i) .lt. setup%lb_states(i))
                    
                    states_matrix(:,:,i) = setup%lb_states(i)
                
                end where
            
                where (states_matrix(:,:,i) .gt. setup%ub_states(i))
                    
                    states_matrix(:,:,i) = setup%ub_states(i)
                
                end where
            
            end do
            
            call matrix_to_states(states_matrix, states)
        
        end subroutine hyper_states_to_states

end module mwd_states
