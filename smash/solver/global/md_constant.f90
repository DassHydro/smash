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
!%
!%      ``GNP``                  Total number of SMASH parameters
!%      ``GNS``                  Total number of SMASH states
!%      ``GPARAMETERS_NAME``     Name of SMASH parameters
!%      ``GSTATES_NAME``         Name of SMASH states
!%      ``GLB_PARAMETERS``       Lower bound of SMASH parameters
!%      ``GUB_PARAMETERS``       Upper bound of SMASH parameters
!%      ``GLB_STATES``           Lower bound of SMASH states
!%      ``GUB_STATES``           Upper bound of SMASH states
!%      ======================= ========================================

module md_constant
    
    implicit none
    
    !% Keep precision constants in lower case
    integer, parameter :: sp = 4
    integer, parameter :: dp = 8
    integer, parameter :: lchar = 128
    
    integer, parameter :: GNP = 16
    integer, parameter :: GNS = 8
    
    character(10), dimension(GNP), parameter :: GPARAMETERS_NAME = &
        
        & (/"ci        ",&
        &   "cp        ",&
        &   "beta      ",&
        &   "cft       ",&
        &   "cst       ",&
        &   "alpha     ",&
        &   "exc       ",&
        
        &   "b         ",&
        &   "cusl1     ",&
        &   "cusl2     ",&
        &   "clsl      ",&
        &   "ks        ",&
        &   "ds        ",&
        &   "dsm       ",&
        &   "ws        ",&
        
        &   "lr        "/)
        
    character(10), dimension(GNS), parameter :: GSTATES_NAME = &
    
        & (/"hi        ",&
        &   "hp        ",&
        &   "hft       ",&
        &   "hst       ",&
        
        &   "husl1     ",&
        &   "husl2     ",&
        &   "hlsl      ",&
        
        &   "hlr       "/)
        
    real(sp), dimension(GNP), parameter :: GLB_PARAMETERS = &
        
        & (/1e-6_sp ,& !% ci
        &   1e-6_sp ,& !% cp
        &   1e-6_sp ,& !% beta
        &   1e-6_sp ,& !% cft
        &   1e-6_sp ,& !% cst
        &   1e-6_sp ,& !% alpha
        &   -50._sp ,& !% exc
        
        &   1e-6_sp ,& !% b
        &   1e-6_sp ,& !% cusl1
        &   1e-6_sp ,& !% cusl2
        &   1e-6_sp ,& !% clsl
        &   1e-6_sp ,& !% ks
        &   1e-6_sp ,& !% ds
        &   1e-6_sp ,& !% dsm
        &   1e-6_sp ,& !% ws
        
        &   1e-6_sp/)  !% lr
        
    real(sp), dimension(GNP), parameter :: GUB_PARAMETERS = &
        
        & (/1e2_sp      ,&  !% ci
        &   1e3_sp      ,&  !% cp
        &   1e3_sp      ,&  !% beta
        &   1e3_sp      ,&  !% cft
        &   1e4_sp      ,&  !% cst
        &   0.999999_sp ,&  !% alpha
        &   50._sp      ,&  !% exc
        
        &   1e1_sp      ,&  !% b
        &   2e3_sp      ,&  !% cusl1
        &   2e3_sp      ,&  !% cusl2
        &   2e3_sp      ,&  !% clsl
        &   1e4_sp      ,&  !% ks
        &   0.999999_sp ,&  !% ds
        &   30._sp      ,&  !% dsm
        &   0.999999_sp ,&  !% ws
        
        &   1e3_sp/)        !% lr
        
    real(sp), dimension(GNS), parameter :: GLB_STATES = &
        
        & (/1e-6_sp ,& !% hi
        &   1e-6_sp ,& !% hp
        &   1e-6_sp ,& !% hft
        &   1e-6_sp ,& !% hst
        
        &   1e-6_sp ,& !% husl1
        &   1e-6_sp ,& !% husl2
        &   1e-6_sp ,& !% hlsl
        
        &   1e-6_sp/)  !% hlr
        
    real(sp), dimension(GNS), parameter :: GUB_STATES = &
        
        & (/0.999999_sp ,& !% hi
        &   0.999999_sp ,& !% hp
        &   0.999999_sp ,& !% hft
        &   0.999999_sp ,& !% hst
        
        &   0.999999_sp ,& !% husl1
        &   0.999999_sp ,& !% husl2
        &   0.999999_sp ,& !% hlsl
        
        &   10000._sp/)    !% hlr

end module md_constant
