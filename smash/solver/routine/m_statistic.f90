!%      This module `m_statistic` encapsulates all SMASH statistic.
!%      
!%      quantile interface:
!%
!%      module procedure quantile0d_i
!%      module procedure quantile0d_r
!%      module procedure quantile1d_i
!%      module procedure quantile1d_r
!%
!%      mean interface:
!%
!%      module procedure mean1d_i
!%      module procedure mean1d_r
!%      module procedure mean2d_i
!%      module procedure mean2d_r
!%
!%      variance interface:
!%
!%      module procedure variance1d_i
!%      module procedure variance1d_r
!%      module procedure variance2d_i
!%      module procedure variance2d_r
!%
!%      std interface:
!%
!%      module procedure std1d_i
!%      module procedure std1d_r
!%      module procedure std2d_i
!%      module procedure std2d_r
!%
!%      contains
!%      
!%      [1]  quantile0d_i
!%      [2]  quantile0d_r
!%      [3]  quantile1d_i
!%      [4]  quantile1d_r
!%      [5]  mean1d_i
!%      [6] mean1d_r
!%      [7] mean2d_i
!%      [8] mean2d_r
!%      [9] variance1d_i
!%      [10] variance1d_r
!%      [11] variance2d_i
!%      [12] variance2d_r
!%      [13] std1d_i
!%      [14] std1d_r
!%      [15] std2d_i
!%      [16] std2d_r

module m_statistic

    use md_constant, only: sp
    use m_sort, only: quicksort
    
    implicit none
    
    interface quantile
    
        module procedure quantile0d_i
        module procedure quantile0d_r
        module procedure quantile1d_i
        module procedure quantile1d_r
    
    end interface quantile
    
    
    interface mean
        
        module procedure mean1d_i
        module procedure mean1d_r
        module procedure mean2d_i
        module procedure mean2d_r
        
    end interface mean
    
    
    interface variance
    
        module procedure variance1d_i
        module procedure variance1d_r
        module procedure variance2d_i
        module procedure variance2d_r
    
    end interface
    
    
    interface std
    
        module procedure std1d_i
        module procedure std1d_r
        module procedure std2d_i
        module procedure std2d_r
    
    end interface
    
    contains

        subroutine quantile0d_i(a, q, res)
        
            !% Notes
            !% -----
            !%
            !% Quantile subroutine
            !%
            !% Given an integer array of dim(1), a single precision quantile value,
            !% it returns the value associated to the quantile value.
            !% Linear interpolation is applied.
            !% 0: gives array(0)
            !% 1: gives array(size(array))
            
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), intent(in) :: q
            real(sp), intent(inout) :: res
            
            integer, dimension(size(a)) :: b
            integer :: n, qt
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort(b)
            
            if (q .ge. 1._sp) then
            
                res = b(n)
                
            else
            
                div = q * real(n - 1, kind=sp) + 1._sp
                qt = floor(div)
                r = mod(div, real(qt, kind=sp))
                    
                res = (1._sp - r) * b(qt) + r * b(qt + 1)
            
            end if
        
        end subroutine quantile0d_i
        
        
        subroutine quantile0d_r(a, q, res)
        
            !% Notes
            !% -----
            !%
            !% Quantile subroutine
            !%
            !% Given a single precision array of dim(1), a single precision quantile value,
            !% it returns a single precision value associated to the quantile value.
            !% Linear interpolation is applied.
            !% 0: gives array(0)
            !% 1: gives array(size(array))
            
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), intent(in) :: q
            real(sp), intent(inout) :: res
            
            real(sp), dimension(size(a)) :: b
            integer :: n, qt
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort(b)
            
            if (q .ge. 1._sp) then
            
                res = b(n)
                
            else
            
                div = q * real(n - 1, kind=sp) + 1._sp
                qt = floor(div)
                r = mod(div, real(qt, kind=sp))
                    
                res = (1._sp - r) * b(qt) + r * b(qt + 1)
            
            end if
        
        end subroutine quantile0d_r
        
        
        subroutine quantile1d_i(a, q, res)
        
            !% Notes
            !% -----
            !%
            !% Quantile subroutine
            !%
            !% Given an integer array of dim(1), a single precision quantile array of dim(1),
            !% it returns a single precision array of dim(1) and size(quantile) associated to the quantile value.
            !% Linear interpolation is applied.
            !% 0: gives array(0)
            !% 1: gives array(size(array))
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(in) :: q
            real(sp), dimension(size(q)), intent(inout) :: res
            
            integer, dimension(size(a)) :: b
            integer :: n, qt, i
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort(b)
            
            do i=1, size(q)
            
                if (q(i) .ge. 1._sp) then
                
                    res(i) = b(n)
                
                else
                
                    div = q(i) * real(n - 1, kind=sp) + 1._sp
                    qt = floor(div)
                    r = mod(div, real(qt, kind=sp))
                    
                    res(i) = (1._sp - r) * b(qt) + r * b(qt + 1)

                end if
            
            end do
        
        end subroutine quantile1d_i
        
        
        subroutine quantile1d_r(a, q, res)
            
            !% Notes
            !% -----
            !%
            !% Quantile subroutine
            !%
            !% Given a single precision array of dim(1), a single precision quantile array of dim(1),
            !% it returns a single precision array of dim(1) and size(quantile) associated to the quantile value.
            !% Linear interpolation is applied.
            !% 0: gives array(0)
            !% 1: gives array(size(array))
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(in) :: q
            real(sp), dimension(size(q)), intent(inout) :: res
            
            real(sp), dimension(size(a)) :: b
            integer :: n, qt, i
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort(b)
            
            do i=1, size(q)
            
                if (q(i) .ge. 1._sp) then
                
                    res(i) = b(n)
                
                else
                
                    div = q(i) * real(n - 1, kind=sp) + 1._sp
                    qt = floor(div)
                    r = mod(div, real(qt, kind=sp))
                    
                    res(i) = (1._sp - r) * b(qt) + r * b(qt + 1)

                end if
            
            end do
        
        end subroutine quantile1d_r
        
        
!%      TODO comment
        subroutine mean1d_i(a, res)
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
        
            res = sum(a) / size(a)
        
        end subroutine mean1d_i
        
        
!%      TODO comment
        subroutine mean1d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
        
            res = sum(a) / size(a)
        
        end subroutine mean1d_r
        
        
!%      TODO comment
        subroutine mean2d_i(a, res)
        
            implicit none
            
            integer, dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
        
            res = sum(a) / size(a)
        
        end subroutine mean2d_i
        
        
!%      TODO comment
        subroutine mean2d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
        
            res = sum(a) / size(a)
        
        end subroutine mean2d_r
        
        
!%      TODO comment
        subroutine variance1d_i(a, res)
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            real(sp) :: sum_a, sum_a2
            integer :: n
            
            sum_a = sum(a)
            sum_a2 = sum(a * a)
            n = size(a)
            
            res = (sum_a2 - (sum_a * sum_a) / n) / (n  - 1)
            
        end subroutine variance1d_i
        
        
!%      TODO comment
        subroutine variance1d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            real(sp) :: sum_a, sum_a2
            integer :: n
            
            sum_a = sum(a)
            sum_a2 = sum(a * a)
            n = size(a)
            
            res = (sum_a2 - (sum_a * sum_a) / n) / (n  - 1)
            
        end subroutine variance1d_r
        
        
!%      TODO comment
        subroutine variance2d_i(a, res)
        
            implicit none
            
            integer, dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            real(sp) :: sum_a, sum_a2
            integer :: n
            
            sum_a = sum(a)
            sum_a2 = sum(a * a)
            n = size(a)
            
            res = (sum_a2 - (sum_a * sum_a) / n) / (n  - 1)
            
        end subroutine variance2d_i
        
        
!%      TODO comment
        subroutine variance2d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            real(sp) :: sum_a, sum_a2
            integer :: n
            
            sum_a = sum(a)
            sum_a2 = sum(a * a)
            n = size(a)
            
            res = (sum_a2 - (sum_a * sum_a) / n) / (n  - 1)
            
        end subroutine variance2d_r
        
        
!%      TODO comment
        subroutine std1d_i(a, res)
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            call variance1d_i(a, res)
            
            res = sqrt(res)
            
        end subroutine std1d_i
        
        
!%      TODO comment
        subroutine std1d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            call variance1d_r(a, res)
            
            res = sqrt(res)
            
        end subroutine std1d_r
        
        
!%      TODO comment
        subroutine std2d_i(a, res)
        
            implicit none
            
            integer, dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            call variance2d_i(a, res)
            
            res = sqrt(res)
            
        end subroutine std2d_i
        
        
!%      TODO comment
        subroutine std2d_r(a, res)
        
            implicit none
            
            real(sp), dimension(:,:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            call variance2d_r(a, res)
            
            res = sqrt(res)
            
        end subroutine std2d_r
        
end module m_statistic

