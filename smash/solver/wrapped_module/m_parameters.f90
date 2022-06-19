!%    This module `m_parameters` encapsulates all SMASH parameters
module m_parameters

    use m_common, only: sp, dp, lchar, np, ns
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    
    implicit none
    
    public :: ParametersDT, parameters_derived_type_to_matrix, &
    & matrix_to_parameters_derived_type, &
    & vector_to_parameters_derived_type
    
    type ParametersDT
        
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: lr
        
    end type ParametersDT
    
    contains
        
        subroutine ParametersDT_initialise(parameters, setup, mesh)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(ParametersDT), intent(inout) :: parameters
            
            integer :: nrow, ncol
            
            nrow = mesh%nrow
            ncol = mesh%ncol
            
            allocate(parameters%cp(nrow, ncol))
            allocate(parameters%cft(nrow, ncol))
            allocate(parameters%lr(nrow, ncol))
            
            call vector_to_parameters_derived_type(&
            & setup%default_parameters, parameters)

        end subroutine ParametersDT_initialise
        
        
        subroutine parameters_derived_type_to_matrix(parameters, matrix)
        
            implicit none
            
            type(ParametersDT), intent(in) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(inout) :: matrix
            
            matrix(:,:,1) = parameters%cp(:,:)
            matrix(:,:,2) = parameters%cft(:,:)
            matrix(:,:,3) = parameters%lr(:,:)
        
        end subroutine parameters_derived_type_to_matrix
        
        
        subroutine matrix_to_parameters_derived_type(matrix, parameters)
            
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(in) :: matrix
            
            parameters%cp(:,:) = matrix(:,:,1)
            parameters%cft(:,:) = matrix(:,:,2)
            parameters%lr(:,:) = matrix(:,:,3)
        
        end subroutine matrix_to_parameters_derived_type
        
        
        subroutine vector_to_parameters_derived_type(vector, parameters)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(np), intent(in) :: vector
            
            parameters%cp = vector(1)
            parameters%cft = vector(2)
            parameters%lr = vector(3)
        
        end subroutine vector_to_parameters_derived_type

end module m_parameters
