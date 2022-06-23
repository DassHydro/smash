!%    This module `mw_states` encapsulates all SMASH states
module mw_states

    use m_common, only: sp, dp, lchar, np, ns
    use mw_setup, only: SetupDT
    use mw_mesh, only: MeshDT
    
    implicit none
    
    public :: StatesDT, states_derived_type_to_matrix, &
    & matrix_to_states_derived_type, vector_to_states_derived_type
    
    type StatesDT
        
        real(sp), dimension(:,:), allocatable :: hp
        real(sp), dimension(:,:), allocatable :: hft
        real(sp), dimension(:,:), allocatable :: hr
        
    end type StatesDT
    
    contains
        
        subroutine StatesDT_initialise(states, setup, mesh)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(StatesDT), intent(inout) :: states
            
            integer :: nrow, ncol
            
            nrow = mesh%nrow
            ncol = mesh%ncol
            
            allocate(states%hp(nrow, ncol))
            allocate(states%hft(nrow, ncol))
            allocate(states%hr(nrow, ncol))
            
            call vector_to_states_derived_type(&
            & setup%default_states, states)

        end subroutine StatesDT_initialise
        
        
        subroutine states_derived_type_to_matrix(states, matrix)
        
            implicit none
            
            type(StatesDT), intent(in) :: states
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2), ns), intent(inout) :: matrix
            
            matrix(:,:,1) = states%hp(:,:)
            matrix(:,:,2) = states%hft(:,:)
            matrix(:,:,3) = states%hr(:,:)
        
        end subroutine states_derived_type_to_matrix
        
        
        subroutine matrix_to_states_derived_type(matrix, states)
            
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2), ns), intent(in) :: matrix
            
            states%hp(:,:) = matrix(:,:,1)
            states%hft(:,:) = matrix(:,:,2)
            states%hr(:,:) = matrix(:,:,3)
        
        end subroutine matrix_to_states_derived_type
        
        
        subroutine vector_to_states_derived_type(vector, states)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(np), intent(in) :: vector
            
            states%hp = vector(1)
            states%hft = vector(2)
            states%hr = vector(3)
        
        end subroutine vector_to_states_derived_type

end module mw_states
