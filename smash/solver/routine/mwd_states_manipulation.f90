module mwd_states_manipulation
    
    use md_common
    use mwd_setup
    use mwd_input_data
    use mwd_states
    
    implicit none
    
    interface set_states
        
        module procedure set0d_states
        module procedure set1d_states
        module procedure set3d_states
    
    end interface set_states
    
    interface set_hyper_states
    
        module procedure set0d_hyper_states
        module procedure set1d_hyper_states
        module procedure set3d_hyper_states
    
    end interface set_hyper_states
    
    contains
    
!%      TODO comment  
        subroutine get_states(states, a)
        
            implicit none
            
            type(StatesDT), intent(in) :: states
            real(sp), dimension(:,:,:), intent(inout) :: a
            
            a(:,:,1) = states%hi(:,:)
            a(:,:,2) = states%hp(:,:)
            a(:,:,3) = states%hft(:,:)
            a(:,:,4) = states%hst(:,:)
            a(:,:,5) = states%hlr(:,:)
        
        end subroutine get_states
        
        
        subroutine set3d_states(states, a)
            
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(:,:,:), intent(in) :: a
            
            states%hi(:,:)  = a(:,:,1)
            states%hp(:,:)  = a(:,:,2)
            states%hft(:,:) = a(:,:,3)
            states%hst(:,:) = a(:,:,4)
            states%hlr(:,:) = a(:,:,5)
        
        end subroutine set3d_states
        
        
        subroutine set1d_states(states, a)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), dimension(ns), intent(in) :: a
            
            states%hi(:,:)  = a(1)
            states%hp(:,:)  = a(2)
            states%hft(:,:) = a(3)
            states%hst(:,:) = a(4)
            states%hlr(:,:) = a(5)
        
        end subroutine set1d_states
        
        
        subroutine set0d_states(states, a)
        
            implicit none
            
            type(StatesDT), intent(inout) :: states
            real(sp), intent(in) :: a
            
            real(sp), dimension(ns) :: a1d
            
            a1d(:) = a
            
            call set1d_states(states, a1d)
        
        end subroutine set0d_states
        
        
        subroutine get_hyper_states(hyper_states, a)
        
            implicit none
            
            type(Hyper_StatesDT), intent(in) :: hyper_states
            real(sp), dimension(:,:,:), intent(inout) :: a

            a(:,:,1) = hyper_states%hi(:,:)
            a(:,:,2) = hyper_states%hp(:,:)
            a(:,:,3) = hyper_states%hft(:,:)
            a(:,:,4) = hyper_states%hst(:,:)
            a(:,:,5) = hyper_states%hlr(:,:)

        end subroutine get_hyper_states
        
        
        subroutine set3d_hyper_states(hyper_states, a)
        
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            real(sp), dimension(:,:,:), intent(in) :: a
            
            hyper_states%hi(:,:)  = a(:,:,1)
            hyper_states%hp(:,:)  = a(:,:,2)
            hyper_states%hft(:,:) = a(:,:,3)
            hyper_states%hst(:,:) = a(:,:,4)
            hyper_states%hlr(:,:) = a(:,:,5)
        
        end subroutine set3d_hyper_states
        
        
        subroutine set1d_hyper_states(hyper_states, a)
            
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            real(sp), dimension(ns), intent(in) :: a
            
            hyper_states%hi(:,:)  = a(1)
            hyper_states%hp(:,:)  = a(2)
            hyper_states%hft(:,:) = a(3)
            hyper_states%hst(:,:) = a(4)
            hyper_states%hlr(:,:) = a(5)
        
        end subroutine set1d_hyper_states
        
        
        subroutine set0d_hyper_states(hyper_states, a)
        
            implicit none
            
            type(Hyper_StatesDT), intent(inout) :: hyper_states
            real(sp), intent(in) :: a
            
            real(sp), dimension(ns) :: a1d
            
            a1d(:) = a
            
            call set1d_hyper_states(hyper_states, a1d)
        
        end subroutine set0d_hyper_states

!%      TODO comment
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
            & size(states%hp, 2), ns) :: states_matrix
            real(sp), dimension(size(states%hp, 1), &
            & size(states%hp, 2)) :: d, dpb
            integer :: i, j
            real(sp) :: a, b
            
            call get_hyper_states(hyper_states, hyper_states_matrix)
            call get_states(states, states_matrix)
            
            !% Add mask later here
            !% 1 in dim2 will be replace with k and apply where on Omega
            do i=1, ns
            
                states_matrix(:,:,i) = hyper_states_matrix(1, 1, i)
                
                do j=1, setup%nd
                
                    d = input_data%descriptor(:,:,j)
            
                    select case(trim(setup%mapping))
                    
                    case("hyper-linear")
                    
                        a = hyper_states_matrix(j + 1, 1, i)
                        b = 1._sp
                        
                    case("hyper-polynomial")
                        
                        a = hyper_states_matrix(2 * j, 1, i)
                        b = hyper_states_matrix(2 * j + 1, 1, i)
                    
                    end select
                    
                    dpb = d ** b
                    
                    states_matrix(:,:,i) = states_matrix(:,:,i) + a * dpb
                
                end do
                
                !% sigmoid transformation lambda = 1
                states_matrix(:,:,i) = (setup%ub_states(i) - setup%lb_states(i)) &
                & * (1._sp / (1._sp + exp(- states_matrix(:,:,i)))) + setup%lb_states(i)
            
            end do

            call set_states(states, states_matrix)
        
        end subroutine hyper_states_to_states

end module mwd_states_manipulation
