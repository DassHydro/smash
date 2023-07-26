!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Cost_OptionsDT
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          variant                  Cost variant ("cls" or "bys")
!%          jobs_cmpt                  Cost variant ("cls" or "bys")
!%          wjobs_cmpt                  Cost variant ("cls" or "bys")
!%          wjreg                    Base weight for regularization
!%          jreg_cmpt                  Cost variant ("cls" or "bys")
!%          wjreg_cmpt                  Cost variant ("cls" or "bys")
!%          gauge                  Cost variant ("cls" or "bys")
!%          wgauge                  Cost variant ("cls" or "bys")
!%          end_warmup                  Cost variant ("cls" or "bys")
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Cost_OptionsDT_initialise
!%      - Cost_OptionsDT_copy

module mwd_cost_options

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Cost_OptionsDT

        character(lchar) :: variant = "..." !$F90W char

        integer :: njoc = -99
        character(lchar), dimension(:), allocatable :: jobs_cmpt !$F90W char-array
        real(sp), dimension(:), allocatable :: wjobs_cmpt

        integer :: njrc = -99
        real(sp) :: wjreg = -99._sp
        character(lchar), dimension(:), allocatable :: jreg_cmpt !$F90W char-array
        real(sp), dimension(:), allocatable :: wjreg_cmpt

        integer, dimension(:), allocatable :: gauge
        real(sp), dimension(:), allocatable :: wgauge

        integer :: end_warmup = -99 !$F90W index

        integer, dimension(:, :), allocatable :: mask_event

    end type Cost_OptionsDT

contains

    subroutine Cost_OptionsDT_initialise(this, setup, mesh, njoc, njrc)

        implicit none

        type(Cost_OptionsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: njoc, njrc

        this%njoc = njoc
        this%njrc = njrc

        allocate (this%jobs_cmpt(this%njoc))
        this%jobs_cmpt = "..."

        allocate (this%wjobs_cmpt(this%njoc))
        this%wjobs_cmpt = -99._sp

        allocate (this%jreg_cmpt(this%njrc))
        this%jreg_cmpt = "..."

        allocate (this%wjreg_cmpt(this%njrc))
        this%wjreg_cmpt = -99._sp

        allocate (this%gauge(mesh%ng))
        this%gauge = -99

        allocate (this%wgauge(mesh%ng))
        this%wgauge = -99._sp

        allocate (this%mask_event(mesh%ng, setup%ntime_step))
        this%mask_event = -99

    end subroutine Cost_OptionsDT_initialise

    subroutine Cost_OptionsDT_copy(this, this_copy)

        implicit none

        type(Cost_OptionsDT), intent(in) :: this
        type(Cost_OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Cost_OptionsDT_copy

end module mwd_cost_options
