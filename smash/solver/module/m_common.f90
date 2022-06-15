module m_common
    
    implicit none
    
    public :: sp, dp, lchar
    
!~     integer, parameter :: dp = kind(0.d0)
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128

end module m_common
