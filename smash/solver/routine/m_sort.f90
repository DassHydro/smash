!%      This module `m_sort` encapsulates all SMASH m_sort
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

module m_sort

    use md_constant, only: sp
    
    implicit none
    
    interface insertionsort
    
        module procedure insertionsort_i
        module procedure insertionsort_r
    
    end interface insertionsort
    
    
    interface quicksort
    
        module procedure quicksort_i
        module procedure quicksort_r
    
    end interface quicksort
    
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


end module m_sort
