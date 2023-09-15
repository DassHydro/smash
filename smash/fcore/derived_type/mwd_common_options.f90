!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Common_OptionsDT
!%        Common options passed by user
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``ncpu``                 Number of CPUs                   (default: 1)
!%          ``verbose``              Enable verbose                   (default: .true.)
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Common_OptionsDT_initialise
!%      - Common_OptionsDT_copy

module mwd_common_options

    implicit none

    type Common_OptionsDT

        integer :: ncpu = 1
        logical :: verbose = .true.

    end type Common_OptionsDT

contains

    subroutine Common_OptionsDT_initialise(this)

        implicit none

        type(Common_OptionsDT), intent(inout) :: this

    end subroutine Common_OptionsDT_initialise

    subroutine Common_OptionsDT_copy(this, this_copy)

        implicit none

        type(Common_OptionsDT), intent(in) :: this
        type(Common_OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Common_OptionsDT_copy

end module mwd_common_options
