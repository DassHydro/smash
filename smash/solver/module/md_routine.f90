!%      This module `md_routine` encapsulates all SMASH routine.
!%      This module is differentiated.
!%
!%      contains


module md_routine

    use mwd_common
    
    implicit none
    
    interface median1d
    
        module procedure median1d_i
        module procedure median1d_r
    
    end interface median1d
    
    
    interface quantile1d
    
        module procedure quantile1d_i
        module procedure quantile1d_r
    
    end interface quantile1d
    

    interface percentile1d
    
        module procedure percentile1d_i
        module procedure percentile1d_r
    
    end interface percentile1d
        
    contains
    
        subroutine insertionsort_i(a)
            
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
        
        
        subroutine median1d_i(a, res)
            
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            integer, dimension(size(a)) :: b
            integer :: n
            
            n = size(a)
            
            b = a

            call quicksort_i(b)
            
            if (mod(n, 2) .gt. 0) then
            
                res = real(b((n + 1) / 2), kind=sp)
                
            else
                
                res = real((b(n / 2) + b((n / 2) + 1)), kind=sp) / 2._sp
            
            end if

        end subroutine median1d_i
        
        
        subroutine median1d_r(a, res)
            
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), intent(inout) :: res
            
            real(sp), dimension(size(a)) :: b
            integer :: n
            
            n = size(a)
            
            b = a

            call quicksort_r(b)
            
            if (mod(n, 2) .gt. 0) then
            
                res = b((n + 1) / 2)
                
            else
                
                res = (b(n / 2) + b((n / 2) + 1)) / 2
            
            end if

        end subroutine median1d_r
        
        
        subroutine quantile1d_i(a, q, res)
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(in) :: q
            real(sp), dimension(size(q)), intent(inout) :: res
            
            integer, dimension(size(a)) :: b
            integer :: n, qt, i
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort_i(b)
            
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
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(in) :: q
            real(sp), dimension(size(q)), intent(inout) :: res
            
            real(sp), dimension(size(a)) :: b
            integer :: n, qt, i
            real(sp) :: div, r
            
            n = size(a)
            b = a
            
            call quicksort_r(b)
            
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
        
        
        subroutine percentile1d_i(a, p, res)
        
            implicit none
            
            integer, dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(inout) :: p
            real(sp), dimension(size(p)), intent(inout) :: res
            
            call quantile1d_i(a, p / 100._sp, res)
            
        end subroutine percentile1d_i
        
        
        subroutine percentile1d_r(a, p, res)
        
            implicit none
            
            real(sp), dimension(:), intent(in) :: a
            real(sp), dimension(:), intent(inout) :: p
            real(sp), dimension(size(p)), intent(inout) :: res
            
            call quantile1d_r(a, p / 100._sp, res)
            
        end subroutine percentile1d_r

    
end module md_routine

