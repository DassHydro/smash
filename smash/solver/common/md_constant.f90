!%      (MD) Module Differentiated.
!%
!%      Variables
!%      ---------
!%
!%      - sp
!%          Single precision value
!%      - dp
!%          Double precision value
!%      - lchar
!%          Characeter length value
!%      - nopr_parameters
!%          Number of parameters in Opr_Parameters
!%      - nopr_states
!%          Number of states in Opr_States

module md_constant

    implicit none

    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128
    
    integer, parameter :: nopr_parameters = 6
    integer, parameter :: nopr_states = 5

end module md_constant
