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
!%      flatten2d interface:
!%      
!%      module procedure flatten2d_i      
!%      module procedure flatten2d_r      
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
!%      [9]  flatten2d_i
!%      [10] flatten2d_r

module m_statistic

    use mwd_common !% only: sp, dp, lchar
    
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
    
    
    interface flatten2d
        
        module procedure flatten2d_i
        module procedure flatten2d_r
    
    end interface flatten2d
    
    
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
        
        
        subroutine quantile0d_i(a, q, res)
            
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
        
        
        subroutine flatten2d_i(mat2d, a, mask)
            
            implicit none
            
            integer, dimension(:,:), intent(in) :: mat2d
            integer, dimension(:), allocatable, intent(inout) :: a
            logical, optional, &
            & dimension(size(mat2d, 1), size(mat2d, 2)), intent(in) :: mask
            
            logical, dimension(size(mat2d, 1), size(mat2d, 2)) :: mask_value
            
            integer :: i, j, n
            
            if (present(mask)) then
            
                mask_value = mask
                
            else
            
                mask_value = .true.
                
            end if
            
            if (allocated(a)) deallocate(a)
            
            allocate(a(count(mask_value)))
            
            n = 1
            
            do i=1, size(mat2d, 2)
            
                do j=1, size(mat2d, 1)
                
                    if (mask_value(j, i)) then
                        
                        a(n) = mat2d(j, i) 
                        n = n + 1
                    
                    end if
 
                end do
            
            end do
        
        end subroutine flatten2d_i
        
        
        subroutine flatten2d_r(mat2d, a, mask)
            
            implicit none
            
            real(sp), dimension(:,:), intent(in) :: mat2d
            real(sp), dimension(:), allocatable, intent(inout) :: a
            logical, optional, &
            & dimension(size(mat2d, 1), size(mat2d, 2)), intent(in) :: mask
            
            logical, dimension(size(mat2d, 1), size(mat2d, 2)) :: mask_value
            
            integer :: i, j, n
            
            if (present(mask)) then
            
                mask_value = mask
                
            else
            
                mask_value = .true.
                
            end if
            
            if (allocated(a)) deallocate(a)
            
            allocate(a(count(mask_value)))
            
            n = 1
            
            do i=1, size(mat2d, 2)
            
                do j=1, size(mat2d, 1)
                
                    if (mask_value(j, i)) then
                        
                        a(n) = mat2d(j, i) 
                        n = n + 1
                    
                    end if
 
                end do
            
            end do
        
        end subroutine flatten2d_r

end module m_statistic

