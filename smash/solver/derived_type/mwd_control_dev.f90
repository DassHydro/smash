!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ControlDT_dev
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``x``                      Control vector
!%          ``x_bkg``                  Control vector background 
!%          ``l``                      Control vector lower bound
!%          ``u``                      Control vector upper bound
!%          ``nbd``                    Control vector kind of bound
!%
!§      Subroutine
!%      ----------
!%
!%      - ControlDT_dev_initialise
!%      - ControlDT_dev_copy

module mwd_control_dev

    use md_constant_dev !% only: sp
    
    implicit none
    
    type ControlDT_dev
    
        real(sp), dimension(:), allocatable :: x
        real(sp), dimension(:), allocatable :: x_bkg
        real(sp), dimension(:), allocatable :: l
        real(sp), dimension(:), allocatable :: u
        integer, dimension(:), allocatable :: nbd
    
    end type ControlDT_dev
    
contains

    subroutine ControlDT_dev_initialise(this)
    
        implicit none
    
        type(ControlDT_dev), intent(inout) :: this
    
    end subroutine ControlDT_dev_initialise
    
        subroutine ControlDT_dev_copy(this, this_copy)
    
        implicit none
        
        type(ControlDT_dev), intent(in) :: this
        type(ControlDT_dev), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine ControlDT_dev_copy
    
end module mwd_control_dev
