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
        
end module m_statistic

