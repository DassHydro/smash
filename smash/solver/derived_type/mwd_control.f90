!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ControlDT
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
!%      - ControlDT_initialise
!%      - ControlDT_copy

module mwd_control

    use md_constant !% only: sp
    
    implicit none
    
    type ControlDT
    
        real(sp), dimension(:), allocatable :: x
        real(sp), dimension(:), allocatable :: x_bkg
        real(sp), dimension(:), allocatable :: l
        real(sp), dimension(:), allocatable :: u
        integer, dimension(:), allocatable :: nbd
    
    end type ControlDT
    
contains

    subroutine ControlDT_initialise(this)
    
        implicit none
    
        type(ControlDT), intent(inout) :: this
    
    end subroutine ControlDT_initialise
    
        subroutine ControlDT_copy(this, this_copy)
    
        implicit none
        
        type(ControlDT), intent(in) :: this
        type(ControlDT), intent(out) :: this_copy
        
        this_copy = this
    
    end subroutine ControlDT_copy
    
end module mwd_control
