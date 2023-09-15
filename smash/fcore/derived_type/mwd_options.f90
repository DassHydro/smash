!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - OptionsDT
!%          Container for all user options (optimize, cost, common)
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``optimize``             Optimize_OptionsDT
!%          ``cost``                 Cost_OptionsDT
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
    use mwd_mesh !% only: MeshDT
    use mwd_cost_options !% only: Cost_OptionsDT, Cost_OptionsDT_initialise
    use mwd_optimize_options !% only: Optimize_OptionsDT, Optimize_OptionsDT_initialise
    use mwd_common_options !% only: Common_OptionsDT, Common_OptionsDT_initialise

    implicit none

    type OptionsDT

        type(Optimize_OptionsDT) :: optimize
        type(Cost_OptionsDT) :: cost
        type(Common_OptionsDT) :: comm

    end type OptionsDT

contains

    subroutine OptionsDT_initialise(this, setup, mesh, njoc, njrc)

        implicit none

        type(OptionsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: njoc, njrc

        call Cost_OptionsDT_initialise(this%cost, setup, mesh, njoc, njrc)
        call Optimize_OptionsDT_initialise(this%optimize, setup)
        call Common_OptionsDT_initialise(this%comm)

    end subroutine OptionsDT_initialise

    subroutine OptionsDT_copy(this, this_copy)

        implicit none

        type(OptionsDT), intent(in) :: this
        type(OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine OptionsDT_copy

end module mwd_options
