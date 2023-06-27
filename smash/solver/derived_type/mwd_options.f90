!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - OptionsDT
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``comm``                 Common_OptionsDT
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - OptionsDT_initialise
!%      - OptionsDT_copy

module mwd_options

    use mwd_setup !% only: SetupDT
    use mwd_common_options !% only: Common_OptionsDT, Common_OptionsDT_initialise
    use mwd_optimize_options !% only: Optimize_OptionsDT, Optimize_OptionsDT_initialise

    implicit none

    type OptionsDT

        type(Common_OptionsDT) :: comm
        type(Optimize_OptionsDT) :: optimize

    end type OptionsDT

contains

    subroutine OptionsDT_initialise(this, setup)

        implicit none

        type(OptionsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup

        call Common_OptionsDT_initialise(this%comm)
        call Optimize_OptionsDT_initialise(this%optimize, setup)

    end subroutine OptionsDT_initialise

    subroutine OptionsDT_copy(this, this_copy)

        implicit none

        type(OptionsDT), intent(in) :: this
        type(OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine OptionsDT_copy

end module mwd_options
