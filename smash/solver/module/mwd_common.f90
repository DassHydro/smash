!%      This module `md_common` encapsulates all SMASH common.
!%      This module is differentiated.

!%      md_common variables
!%
!%      ======================= ========================================
!%      `variables`             Description
!%      ======================= ========================================
!%      ``sp``                  Single precision value
!%      ``dp``                  Double precision value
!%      ``lchar``               Characeter length value
!%      ``np``                  Number of SMASH parameters (ParametersDT)
!%      ``ns``                  Number of SMASH states (StatesDT)
!%      ======================= ========================================

module mwd_common
    
    implicit none
    
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128
    
    integer, parameter :: np = 8
    integer, parameter :: ns = 5
    
    character(10), dimension(np) :: name_parameters = &
        
        & (/"ci        ",&
        &   "cp        ",&
        &   "beta      ",&
        &   "cft       ",&
        &   "cst       ",&
        &   "alpha     ",&
        &   "exc       ",&
        &   "lr        "/)
        
    character(10), dimension(ns) :: name_states = &
    
        & (/"hi        ",&
        &   "hp        ",&
        &   "hft       ",&
        &   "hst       ",&
        &   "hlr       "/)

end module mwd_common
