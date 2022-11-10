!%      This module `md_constant` encapsulates all SMASH constant value.
!%      This module is differentiated.
!%
!%      md_constant variables
!%
!%      </> Public
!%      ======================= ========================================
!%      `Variables`             Description
!%      ======================= ========================================
!%      ``sp``                  Single precision value
!%      ``dp``                  Double precision value
!%      ``lchar``               Characeter length value
!%      ``np``                  Total number of SMASH parameters (ParametersDT)
!%      ``ns``                  Total number of SMASH states (StatesDT)
!%      ======================= ========================================

module md_constant
    
    implicit none
    
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128
    
    integer, parameter :: np = 16
    integer, parameter :: ns = 8

end module md_constant
