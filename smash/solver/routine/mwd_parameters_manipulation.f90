module mwd_parameters_manipulation
    
    use md_constant
    use mwd_setup
    use mwd_input_data
    use mwd_parameters
    
    implicit none
    
    interface set_parameters
        
        module procedure set0d_parameters
        module procedure set1d_parameters
        module procedure set3d_parameters
    
    end interface set_parameters
    
    interface set_hyper_parameters
    
        module procedure set0d_hyper_parameters
        module procedure set1d_hyper_parameters
        module procedure set3d_hyper_parameters
    
    end interface set_hyper_parameters
    
    contains
    
!%      TODO comment  
        subroutine get_parameters(parameters, a)
        
            implicit none
            
            type(ParametersDT), intent(in) :: parameters
            real(sp), dimension(:,:,:), intent(inout) :: a
            
            a(:,:,1) = parameters%ci(:,:)
            a(:,:,2) = parameters%cp(:,:)
            a(:,:,3) = parameters%beta(:,:)
            a(:,:,4) = parameters%cft(:,:)
            a(:,:,5) = parameters%cst(:,:)
            a(:,:,6) = parameters%alpha(:,:)
            a(:,:,7) = parameters%exc(:,:)
            a(:,:,8) = parameters%lr(:,:)
        
        end subroutine get_parameters
        
        
        subroutine set3d_parameters(parameters, a)
            
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(:,:,:), intent(in) :: a
            
            parameters%ci(:,:)    = a(:,:,1)
            parameters%cp(:,:)    = a(:,:,2)
            parameters%beta(:,:)  = a(:,:,3)
            parameters%cft(:,:)   = a(:,:,4)
            parameters%cst(:,:)   = a(:,:,5)
            parameters%alpha(:,:) = a(:,:,6)
            parameters%exc(:,:)   = a(:,:,7)
            parameters%lr(:,:)    = a(:,:,8)
        
        end subroutine set3d_parameters
        
        
        subroutine set1d_parameters(parameters, a)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(np), intent(in) :: a
            
            parameters%ci(:,:)    = a(1)
            parameters%cp(:,:)    = a(2)
            parameters%beta(:,:)  = a(3)
            parameters%cft(:,:)   = a(4)
            parameters%cst(:,:)   = a(5)
            parameters%alpha(:,:) = a(6)
            parameters%exc(:,:)   = a(7)
            parameters%lr(:,:)    = a(8)
        
        end subroutine set1d_parameters
        
        
        subroutine set0d_parameters(parameters, a)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), intent(in) :: a
            
            real(sp), dimension(np) :: a1d
            
            a1d(:) = a
            
            call set1d_parameters(parameters, a1d)
        
        end subroutine set0d_parameters
        
        
        subroutine get_hyper_parameters(hyper_parameters, a)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(in) :: hyper_parameters
            real(sp), dimension(:,:,:), intent(inout) :: a

            a(:,:,1) = hyper_parameters%ci(:,:)
            a(:,:,2) = hyper_parameters%cp(:,:)
            a(:,:,3) = hyper_parameters%beta(:,:)
            a(:,:,4) = hyper_parameters%cft(:,:)
            a(:,:,5) = hyper_parameters%cst(:,:)
            a(:,:,6) = hyper_parameters%alpha(:,:)
            a(:,:,7) = hyper_parameters%exc(:,:)
            a(:,:,8) = hyper_parameters%lr(:,:)
        
        end subroutine get_hyper_parameters
        
        
        subroutine set3d_hyper_parameters(hyper_parameters, a)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            real(sp), dimension(:,:,:), intent(in) :: a
            
            hyper_parameters%ci(:,:)    = a(:,:,1)
            hyper_parameters%cp(:,:)    = a(:,:,2)
            hyper_parameters%beta(:,:)  = a(:,:,3)
            hyper_parameters%cft(:,:)   = a(:,:,4)
            hyper_parameters%cst(:,:)   = a(:,:,5)
            hyper_parameters%alpha(:,:) = a(:,:,6)
            hyper_parameters%exc(:,:)   = a(:,:,7)
            hyper_parameters%lr(:,:)    = a(:,:,8)
        
        end subroutine set3d_hyper_parameters
        
        
        subroutine set1d_hyper_parameters(hyper_parameters, a)
            
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            real(sp), dimension(np), intent(in) :: a
            
            hyper_parameters%ci(:,:)    = a(1)
            hyper_parameters%cp(:,:)    = a(2)
            hyper_parameters%beta(:,:)  = a(3)
            hyper_parameters%cft(:,:)   = a(4)
            hyper_parameters%cst(:,:)   = a(5)
            hyper_parameters%alpha(:,:) = a(6)
            hyper_parameters%exc(:,:)   = a(7)
            hyper_parameters%lr(:,:)    = a(8)
        
        end subroutine set1d_hyper_parameters
        
        
        subroutine set0d_hyper_parameters(hyper_parameters, a)
        
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
            real(sp), intent(in) :: a
            
            real(sp), dimension(np) :: a1d
            
            a1d(:) = a
            
            call set1d_hyper_parameters(hyper_parameters, a1d)
        
        end subroutine set0d_hyper_parameters

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
            
            call get_hyper_parameters(hyper_parameters, hyper_parameters_matrix)
            call get_parameters(parameters, parameters_matrix)
            
            !% Add mask later here
            !% 1 in dim2 will be replace with k and apply where on Omega
            do i=1, np
            
                parameters_matrix(:,:,i) = hyper_parameters_matrix(1, 1, i)
                
                do j=1, setup%nd
                
                    d = input_data%descriptor(:,:,j)
            
                    select case(trim(setup%optimize%mapping))
                    
                    case("hyper-linear")
                    
                        a = hyper_parameters_matrix(j + 1, 1, i)
                        b = 1._sp
                        
                    case("hyper-polynomial")
                        
                        a = hyper_parameters_matrix(2 * j, 1, i)
                        b = hyper_parameters_matrix(2 * j + 1, 1, i)
                    
                    end select
                    
                    dpb = d ** b
                    
                    parameters_matrix(:,:,i) = parameters_matrix(:,:,i) + a * dpb
                
                end do
                
                !% sigmoid transformation lambda = 1
                parameters_matrix(:,:,i) = (setup%optimize%ub_parameters(i) - setup%optimize%lb_parameters(i)) &
                & * (1._sp / (1._sp + exp(- parameters_matrix(:,:,i)))) + setup%optimize%lb_parameters(i)
            
            end do

            call set_parameters(parameters, parameters_matrix)
        
        end subroutine hyper_parameters_to_parameters

end module mwd_parameters_manipulation
