!%      This module `md_kind` encapsulates all SMASH kind value.
!%      This module is differentiated.
!%
!%      md_kind variables
!%
!%      </> Public
!%      ======================= ========================================
!%      `Variables`             Description
!%      ======================= ========================================
!%      ``sp``                  Single precision value
!%      ``dp``                  Double precision value
!%      ``lchar``               Characeter length value
!%      ``np``                  Number of SMASH parameters (ParametersDT)
!%      ``ns``                  Number of SMASH states (StatesDT)
!%      ======================= ========================================

module md_kind
    
    implicit none
    
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128
    
    integer, parameter :: np = 8
    integer, parameter :: ns = 5

end module md_kind
