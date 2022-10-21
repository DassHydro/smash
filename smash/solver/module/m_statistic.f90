!%      This module `m_statistic` encapsulates all SMASH statistic.
!%
!%      insertionsort interface:
!%
!%      module procedure insertionsort_i
!%      module procedure insertionsort_r
!%
!%      quicksort interface:
!%
!%      module procedure quicksort_i
!%      module procedure quicksort_r
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
!%      [1]  insertionsort_i
!%      [2]  insertionsort_r
!%      [3]  quicksort_i
!%      [4]  quicksort_r
!%      [5]  quantile0d_i
!%      [6]  quantile0d_r
!%      [7]  quantile1d_i
!%      [8]  quantile1d_r
!%      [9]  mean1d_i
!%      [10] mean1d_r
!%      [11] mean2d_i
!%      [12] mean2d_r
!%      [13] variance1d_i
!%      [14] variance1d_r
!%      [15] variance2d_i
!%      [16] variance2d_r
!%      [17] std1d_i
!%      [18] std1d_r
!%      [19] std2d_i
!%      [20] std2d_r

module m_statistic

    use mwd_common, only: sp
    
    implicit none
    
    interface insertionsort
    
        module procedure insertionsort_i
        module procedure insertionsort_r
    
    end interface insertionsort
    
    
    interface quicksort
    
        module procedure quicksort_i
        module procedure quicksort_r
    
    end interface quicksort
    
    
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
    
        subroutine insertionsort_i(a)
        
            !% Notes
            !% -----
            !%
            !% Insertion sort subroutine
            !%
            !% Given an integer array of dim(1),
            !% it returns the sorted integer array in-place
            
            implicit none
            
            integer, dimension(:), intent(inout) :: a
            
            integer :: imd
            integer :: n, i, j
            
            n =  size(a)
            
            do j=2, n
                
                imd = a(j)
                
                do i=j-1, 1,-1
                
                    if (a(i) .le. imd) goto 1
                    a(i+1) = a(i)
                
                end do 
                
                i = 0
        1       a(i+1) = imd 
            
            end do
        
        end subroutine insertionsort_i
        
        
        subroutine insertionsort_r(a)
            
            !% Notes
            !% -----
            !%
            !% Insertion sort subroutine
            !%
            !% Given a single precision array of dim(1),
            !% it returns the sorted single precision array in-place
            
            implicit none
            
            real(sp), dimension(:), intent(inout) :: a
            
            real(sp) :: imd
            integer :: n, i, j
            
            n = size(a)
            
            do j=2, n
                
                imd = a(j)
                
                do i=j-1, 1,-1
                
                    if (a(i) .le. imd) goto 1
                    a(i+1) = a(i)
                
                end do 
                
                i = 0
        1       a(i+1) = imd 
            
            end do
        
        end subroutine insertionsort_r
        
        
        recursive subroutine quicksort_i(a)
        
            !% Notes
            !% -----
            !%
            !% Quicksort sort subroutine
            !%
            !% Given an integer array of dim(1),
            !% it returns the sorted integer array in-place.
            !% It uses insertion sort for array of size lower than or equal to 20
            
            implicit none
            
            integer, dimension(:), intent(inout) :: a
            
            integer :: x, t
            integer :: first = 1, last
            integer :: n, i, j
            
            n = size(a)
            
            !% Insertion sort for array size lower than or equal to 20
            if (n .le. 20) then
                
                call insertionsort_i(a)

            else
            
                last = n
                x = a((first+last) / 2)
                i = first
                j = last

                do
                    do while (a(i) .lt. x)
                       
                        i = i + 1
                    
                    end do
                
                    do while (x .lt. a(j))
                        
                        j = j - 1
                    
                    end do
                    
                    if (i .ge. j) exit
                    
                    t = a(i);  a(i) = a(j);  a(j) = t
                    i= i + 1
                    j= j - 1
                    
                end do

                if (first .lt. i - 1) call quicksort_i(a(first : i - 1))
                if (j + 1 .lt. last)  call quicksort_i(a(j + 1 : last))
                
            end if
            
            end subroutine quicksort_i
        
        
        recursive subroutine quicksort_r(a)
        
            !% Notes
            !% -----
            !%
            !% Quicksort sort subroutine
            !%
            !% Given a single precision array of dim(1),
            !% it returns the sorted single precision array in-place.
            !% It uses insertion sort for array of size lower than or equal to 20
            
            implicit none
            
            real(sp), dimension(:), intent(inout) :: a
            
            real(sp) :: x, t
            integer :: first = 1, last
            integer :: n, i, j
            
            n = size(a)
            
            !% Insertion sort for array size lower than or equal to 20
            if (n .le. 20) then
            
                call insertionsort_r(a)
                
            else

                last = n
                x = a((first+last) / 2)
                i = first
                j = last

                do
                    do while (a(i) .lt. x)
                       
                        i = i + 1
                    
                    end do
                
                    do while (x .lt. a(j))
                        
                        j = j - 1
                    
                    end do
                    
                    if (i .ge. j) exit
                    
                    t = a(i);  a(i) = a(j);  a(j) = t
                    i= i + 1
                    j= j - 1
                    
                end do

                if (first .lt. i - 1) call quicksort_r(a(first : i - 1))
                if (j + 1 .lt. last)  call quicksort_r(a(j + 1 : last))
                
            end if
            
            end subroutine quicksort_r
        
        
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

