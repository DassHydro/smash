!%      This module `mwd_parameters` encapsulates all SMASH parameters.
!%      This module is wrapped and differentiated.
!%
!%      ParametersDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``ci``                   Interception parameter          [mm]    (default: 1)     ]0, +Inf[
!%      ``cp``                   Production parameter            [mm]    (default: 200)   ]0, +Inf[
!%      ``beta``                 Percolation parameter           [-]     (default: 1000)  ]0, +Inf[
!%      ``cft``                  Fast transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``cst``                  Slow transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``alpha``                Transfer partitioning parameter [-]     (default: 0.9)   ]0, 1[
!%      ``exc``                  Exchange parameter              [mm/dt] (default: 0)     ]-Inf, +Inf[
!%      ``lr``                   Linear routing parameter        [min]   (default: 5)     ]0, +Inf[
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1] ParametersDT_initialise
!%      [2] parameters_to_matrix
!%      [3] matrix_to_parameters
!%      [4] vector_to_parameters
!%      [5] set0_parameters
!%      [6] set1_parameters

module mwd_parameters

    use mwd_common !% only: sp, np
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    
    implicit none
    
    type ParametersDT
        
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        real(sp), dimension(:,:), allocatable :: lr
        
    end type ParametersDT
    
    type Hyper_ParametersDT
    
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        real(sp), dimension(:,:), allocatable :: lr
    
    end type Hyper_ParametersDT
    
    contains
        
        subroutine ParametersDT_initialise(parameters, mesh)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT initialisation subroutine
        
            implicit none
            
            type(MeshDT), intent(in) :: mesh
            type(ParametersDT), intent(inout) :: parameters
            
            integer :: nrow, ncol
            
            nrow = mesh%nrow
            ncol = mesh%ncol
            
            allocate(parameters%ci(nrow, ncol))
            allocate(parameters%cp(nrow, ncol))
            allocate(parameters%beta(nrow, ncol))
            allocate(parameters%cft(nrow, ncol))
            allocate(parameters%cst(nrow, ncol))
            allocate(parameters%alpha(nrow, ncol))
            allocate(parameters%exc(nrow, ncol))
            allocate(parameters%lr(nrow, ncol))
            
            parameters%ci    = 1._sp
            parameters%cp    = 200._sp
            parameters%beta  = 1000._sp
            parameters%cft   = 500._sp
            parameters%cst   = 500._sp
            parameters%alpha = 0.9_sp
            parameters%exc   = 0._sp
            parameters%lr    = 5._sp
 
        end subroutine ParametersDT_initialise
        
        
        subroutine Hyper_ParametersDT_initialise(hyper_parameters, setup, mesh)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT initialisation subroutine
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            
            integer :: n
            
            n = np * (1 + 2 * setup%nd)
            
            allocate(hyper_parameters%ci(n, 1))
            allocate(hyper_parameters%cp(n, 1))
            allocate(hyper_parameters%beta(n, 1))
            allocate(hyper_parameters%cft(n, 1))
            allocate(hyper_parameters%cst(n, 1))
            allocate(hyper_parameters%alpha(n, 1))
            allocate(hyper_parameters%exc(n, 1))
            allocate(hyper_parameters%lr(n, 1))
            
            hyper_parameters%ci    = 0._sp
            hyper_parameters%cp    = 0._sp
            hyper_parameters%beta  = 0._sp
            hyper_parameters%cft   = 0._sp
            hyper_parameters%cst   = 0._sp
            hyper_parameters%alpha = 0._sp
            hyper_parameters%exc   = 0._sp
            hyper_parameters%lr    = 0._sp
 
        end subroutine Hyper_ParametersDT_initialise
        
!%      TODO comment  
        subroutine parameters_to_matrix(parameters, matrix)
        
            implicit none
            
            type(ParametersDT), intent(in) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(inout) :: matrix
            
            matrix(:,:,1) = parameters%ci(:,:)
            matrix(:,:,2) = parameters%cp(:,:)
            matrix(:,:,3) = parameters%beta(:,:)
            matrix(:,:,4) = parameters%cft(:,:)
            matrix(:,:,5) = parameters%cst(:,:)
            matrix(:,:,6) = parameters%alpha(:,:)
            matrix(:,:,7) = parameters%exc(:,:)
            matrix(:,:,8) = parameters%lr(:,:)
        
        end subroutine parameters_to_matrix
        
        
!%      TODO comment  
        subroutine matrix_to_parameters(matrix, parameters)
            
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(in) :: matrix
            
            parameters%ci(:,:) = matrix(:,:,1)
            parameters%cp(:,:) = matrix(:,:,2)
            parameters%beta(:,:) = matrix(:,:,3)
            parameters%cft(:,:) = matrix(:,:,4)
            parameters%cst(:,:) = matrix(:,:,5)
            parameters%alpha(:,:) = matrix(:,:,6)
            parameters%exc(:,:) = matrix(:,:,7)
            parameters%lr(:,:) = matrix(:,:,8)
        
        end subroutine matrix_to_parameters
        
        
!%      TODO comment  
        subroutine vector_to_parameters(vector, parameters)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(np), intent(in) :: vector
            
            parameters%ci = vector(1)
            parameters%cp = vector(2)
            parameters%beta = vector(3)
            parameters%cft = vector(4)
            parameters%cst = vector(5)
            parameters%alpha = vector(6)
            parameters%exc = vector(7)
            parameters%lr = vector(8)
        
        end subroutine vector_to_parameters
        
        
!%      TODO comment  
        subroutine set0_parameters(parameters)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            
            real(sp), dimension(np) :: vector0
            
            vector0 = 0._sp
            
            call vector_to_parameters(vector0, parameters)
        
        end subroutine set0_parameters
        
        
!%      TODO comment  
        subroutine set1_parameters(parameters)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            
            real(sp), dimension(np) :: vector1
            
            vector1 = 1._sp
            
            call vector_to_parameters(vector1, parameters)
        
        end subroutine set1_parameters
        
        
!%      TODO comment
        subroutine hyper_parameters_to_matrix(hyper_parameters, matrix)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters
            real(sp), dimension(size(hyper_parameters%cp, 1), &
            & size(hyper_parameters%cp, 2), np), intent(inout) :: matrix

            matrix(:,:,1) = hyper_parameters%ci(:,:)
            matrix(:,:,2) = hyper_parameters%cp(:,:)
            matrix(:,:,3) = hyper_parameters%beta(:,:)
            matrix(:,:,4) = hyper_parameters%cft(:,:)
            matrix(:,:,5) = hyper_parameters%cst(:,:)
            matrix(:,:,6) = hyper_parameters%alpha(:,:)
            matrix(:,:,7) = hyper_parameters%exc(:,:)
            matrix(:,:,8) = hyper_parameters%lr(:,:)
        
        end subroutine hyper_parameters_to_matrix

        
!%      TODO comment
        subroutine matrix_to_hyper_parameters(matrix, hyper_parameters)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            real(sp), dimension(size(hyper_parameters%cp, 1), &
            & size(hyper_parameters%cp, 2), np), intent(in) :: matrix
            
            hyper_parameters%ci(:,:) = matrix(:,:,1)
            hyper_parameters%cp(:,:) = matrix(:,:,2)
            hyper_parameters%beta(:,:) = matrix(:,:,3)
            hyper_parameters%cft(:,:) = matrix(:,:,4)
            hyper_parameters%cst(:,:) = matrix(:,:,5)
            hyper_parameters%alpha(:,:) = matrix(:,:,6)
            hyper_parameters%exc(:,:) = matrix(:,:,7)
            hyper_parameters%lr(:,:) = matrix(:,:,8)
        
        end subroutine matrix_to_hyper_parameters
  

!%      TODO comment
        subroutine set0_hyper_parameters(hyper_parameters)
            
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            
            real(sp), dimension(size(hyper_parameters%cp, 1), &
            & size(hyper_parameters%cp, 2), np) :: matrix
            
            matrix = 0._sp
            
            call matrix_to_hyper_parameters(matrix, hyper_parameters)
        
        end subroutine set0_hyper_parameters
        
        
!%      TODO comment
        subroutine set1_hyper_parameters(hyper_parameters)
            
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            
            real(sp), dimension(size(hyper_parameters%cp, 1), &
            & size(hyper_parameters%cp, 2), np) :: matrix
            
            matrix = 1._sp
            
            call matrix_to_hyper_parameters(matrix, hyper_parameters)
        
        end subroutine set1_hyper_parameters
        
        
!%      TODO comment
        subroutine hyper_parameters_to_parameters(hyper_parameters, &
        & parameters, setup, input_data)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters
            type(ParametersDT), intent(inout) :: parameters
            type(SetupDT), intent(in) :: setup
            type(Input_DataDT), intent(in) :: input_data
            
            real(sp), dimension(size(hyper_parameters%cp, 1), &
            & size(hyper_parameters%cp, 2), np) :: hyper_parameters_matrix
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np) :: parameters_matrix
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2)) :: d, dpb
            integer :: i, j
            real(sp) :: a, b
            
            call hyper_parameters_to_matrix(hyper_parameters, hyper_parameters_matrix)
            call parameters_to_matrix(parameters, parameters_matrix)
            
            !% Add mask later here
            !% 1 in dim2 will be replace with k and apply where on Omega
            do i=1, np
            
                parameters_matrix(:,:,i) = hyper_parameters_matrix(1, 1, i)
                
                do j=1, setup%nd
                
                    d = input_data%descriptor(:,:,j)
            
                    a = hyper_parameters_matrix(2 * j, 1, i)
                    b = hyper_parameters_matrix(2 * j + 1, 1, i)
                    dpb = d ** b
                    
                    parameters_matrix(:,:,i) = parameters_matrix(:,:,i) + a * dpb
                
                end do
                
                where (parameters_matrix(:,:,i) .lt. setup%lb_parameters(i))
                    
                    parameters_matrix(:,:,i) = setup%lb_parameters(i)
                
                end where
            
                where (parameters_matrix(:,:,i) .gt. setup%ub_parameters(i))
                    
                    parameters_matrix(:,:,i) = setup%ub_parameters(i)
                
                end where
            
            end do

            call matrix_to_parameters(parameters_matrix, parameters)
        
        end subroutine hyper_parameters_to_parameters
        
end module mwd_parameters
