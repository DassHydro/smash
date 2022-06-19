module m_common
    
    implicit none
    
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128

    integer, parameter :: np = 3
    integer, parameter :: ns = 3
    
    character(len=10), dimension(np), parameter :: list_parameters = [&
    "cp        ",&
    "cft       ",&
    "lr        "]

end module m_common
