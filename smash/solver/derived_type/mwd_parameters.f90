!%      This module `mwd_parameters` encapsulates all SMASH parameters.
!%      This module is wrapped and differentiated.
!%
!%      ParametersDT type:
!%      
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``ci``                   Interception parameter          [mm]    (default: 1)     ]0, +Inf[
!%      ``cp``                   Production parameter            [mm]    (default: 200)   ]0, +Inf[
!%      ``beta``                 Percolation parameter           [-]     (default: 1000)  ]0, +Inf[
!%      ``cft``                  Fast transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``cst``                  Slow transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``alpha``                Transfer partitioning parameter [-]     (default: 0.9)   ]0, 1[
!%      ``exc``                  Exchange parameter              [mm/dt] (default: 0)     ]-Inf, +Inf[
!%      ``lr``                   Linear routing parameter        [min]   (default: 5)     ]0, +Inf[
!%      ======================== =======================================
!%      
!%      Hyper_ParametersDT type:
!%
!%      </> Public
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``ci``                   Interception parameter          [mm]    (default: 1)     ]0, +Inf[
!%      ``cp``                   Production parameter            [mm]    (default: 200)   ]0, +Inf[
!%      ``beta``                 Percolation parameter           [-]     (default: 1000)  ]0, +Inf[
!%      ``cft``                  Fast transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``cst``                  Slow transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``alpha``                Transfer partitioning parameter [-]     (default: 0.9)   ]0, 1[
!%      ``exc``                  Exchange parameter              [mm/dt] (default: 0)     ]-Inf, +Inf[
!%      ``lr``                   Linear routing parameter        [min]   (default: 5)     ]0, +Inf[
!%      ======================== =======================================
!%
!%      contains
!%
!%      [1]  ParametersDT_initialise
!%      [2]  Hyper_ParametersDT_initialise
!%      [3]  parameters_to_matrix
!%      [4]  matrix_to_parameters
!%      [5]  vector_to_parameters
!%      [6]  set0_parameters
!%      [7]  set1_parameters
!%      [8]  hyper_parameters_to_matrix
!%      [10] matrix_to_hyper_parameters
!%      [11] set0_hyper_parameters
!%      [12] set1_hyper_parameters
!%      [13] hyper_parameters_to_parameters

module mwd_parameters

    use md_common !% only: sp, np
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    
    implicit none
    
    type ParametersDT
        
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        real(sp), dimension(:,:), allocatable :: lr
        
    end type ParametersDT
    
    type Hyper_ParametersDT
    
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        real(sp), dimension(:,:), allocatable :: lr
    
    end type Hyper_ParametersDT
    
    contains
        
        subroutine ParametersDT_initialise(this, mesh)
        
            !% Notes
            !% -----
            !%
            !% ParametersDT initialisation subroutine
        
            implicit none
            
            type(ParametersDT), intent(inout) :: this
            type(MeshDT), intent(in) :: mesh
            
            allocate(this%ci(mesh%nrow, mesh%ncol))
            allocate(this%cp(mesh%nrow, mesh%ncol))
            allocate(this%beta(mesh%nrow, mesh%ncol))
            allocate(this%cft(mesh%nrow, mesh%ncol))
            allocate(this%cst(mesh%nrow, mesh%ncol))
            allocate(this%alpha(mesh%nrow, mesh%ncol))
            allocate(this%exc(mesh%nrow, mesh%ncol))
            allocate(this%lr(mesh%nrow, mesh%ncol))
            
            this%ci    = 1._sp
            this%cp    = 200._sp
            this%beta  = 1000._sp
            this%cft   = 500._sp
            this%cst   = 500._sp
            this%alpha = 0.9_sp
            this%exc   = 0._sp
            this%lr    = 5._sp
 
        end subroutine ParametersDT_initialise
        
        
        subroutine Hyper_ParametersDT_initialise(this, setup)
        
            !% Notes
            !% -----
            !%
            !% Hyper_ParametersDT initialisation subroutine
        
            implicit none
            
            type(Hyper_ParametersDT), intent(inout) :: this
            type(SetupDT), intent(in) :: setup
            
            integer :: n
            
            select case(trim(setup%optimize%mapping))
            
            case("hyper-linear")
            
                n = (1 + setup%nd)
                
            case("hyper-polynomial")
            
                n = (1 + 2 * setup%nd)
                
            end select
            
            allocate(this%ci(n, 1))
            allocate(this%cp(n, 1))
            allocate(this%beta(n, 1))
            allocate(this%cft(n, 1))
            allocate(this%cst(n, 1))
            allocate(this%alpha(n, 1))
            allocate(this%exc(n, 1))
            allocate(this%lr(n, 1))
 
        end subroutine Hyper_ParametersDT_initialise


end module mwd_parameters
