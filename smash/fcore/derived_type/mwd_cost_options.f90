!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Cost_OptionsDT
!%          Cost options passed by user to define the output cost
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``bayesian``             Enable bayesian cost computation
!%          ``njoc``                 Number of jobs components
!%          ``jobs_cmpt``            Jobs components
!%          ``wjobs_cmpt``           Weight jobs components
!%          ``njrc``                 Number of jreg components
!%          ``wjreg``                Base weight for regularization
!%          ``jreg_cmpt``            Jreg components
!%          ``wjreg_cmpt``           Weight jreg components
!%          ``nog``                  Number of optimized gauges
!%          ``gauge``                Optimized gauges
!%          ``wgauge``               Weight optimized gauges
!%          ``end_warmup``           End Warmup index
!%          ``n_event ``             Number of flood events
!%          ``mask_event  ``         Mask info by segmentation algorithm
!%          ``control_prior``        Array of PriorType (from mwd_bayesian_tools)
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Cost_OptionsDT_initialise
!%      - Cost_OptionsDT_copy

module mwd_cost_options

    use mwd_bayesian_tools !% only PriorType, PriorType_initialise
    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Cost_OptionsDT

        logical :: bayesian = .false.

        integer :: njoc = -99
        character(lchar), dimension(:), allocatable :: jobs_cmpt !$F90W char-array
        real(sp), dimension(:), allocatable :: wjobs_cmpt
        character(lchar), dimension(:), allocatable :: jobs_cmpt_tfm !$F90W char-array

        integer :: njrc = -99
        real(sp) :: wjreg = -99._sp
        character(lchar), dimension(:), allocatable :: jreg_cmpt !$F90W char-array
        real(sp), dimension(:), allocatable :: wjreg_cmpt

        integer :: nog = -99
        integer, dimension(:), allocatable :: gauge
        real(sp), dimension(:), allocatable :: wgauge

        integer :: end_warmup = -99 !$F90W index

        integer, dimension(:), allocatable :: n_event
        integer, dimension(:, :), allocatable :: mask_event

        type(PriorType), dimension(:), allocatable :: control_prior

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

        allocate (this%jobs_cmpt_tfm(this%njoc))
        this%jobs_cmpt_tfm = "..."

        allocate (this%jreg_cmpt(this%njrc))
        this%jreg_cmpt = "..."

        allocate (this%wjreg_cmpt(this%njrc))
        this%wjreg_cmpt = -99._sp

        allocate (this%gauge(mesh%ng))
        this%gauge = -99

        allocate (this%wgauge(mesh%ng))
        this%wgauge = -99._sp

        allocate (this%n_event(mesh%ng))
        this%n_event = -99

        allocate (this%mask_event(mesh%ng, setup%ntime_step))
        this%mask_event = -99

    end subroutine Cost_OptionsDT_initialise

    subroutine Cost_OptionsDT_copy(this, this_copy)

        implicit none

        type(Cost_OptionsDT), intent(in) :: this
        type(Cost_OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Cost_OptionsDT_copy

    subroutine Cost_OptionsDT_alloc_control_prior(this, n, npar)

        implicit none

        type(Cost_OptionsDT), intent(inout) :: this
        integer, intent(in) :: n
        integer, dimension(n), intent(in) :: npar

        integer :: i
        allocate (this%control_prior(n))

        do i = 1, n
            call PriorType_initialise(this%control_prior(i), npar(i))
        end do

    end subroutine Cost_OptionsDT_alloc_control_prior

end module mwd_cost_options
