!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - ReturnsDT
!%          Usefull quantities returned by the hydrological model other than response variables themselves.
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``nmts``                 Number of time step to return
!%          ``mask_time_step``       Mask of time step
!%          ``rr_states``            Array of Rr_StatesDT
!%          ``rr_states_flag``       Return flag of rr_states
!%          ``q_domain``             Array of discharge
!%          ``q_domain_flag``        Return flag of q_domain
!%          ``cost``                 Cost value
!%          ``cost_flag``            Return flag of cost
!%          ``jobs``                 Jobs value
!%          ``jobs_flag``            Return flag of jobs
!%          ``jreg``                 Jreg value
!%          ``jreg_flag``            Return flag of jreg
!%          ``log_lkh``              Log_lkh value
!%          ``log_lkh_flag``         Return flag of log_lkh
!%          ``log_prior``            Log_prior value
!%          ``log_prior_flag``       Return flag of log_prior
!%          ``log_h``                Log_h value
!%          ``log_h_flag``           Return flag of log_h
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - ReturnsDT_initialise
!%      - ReturnsDT_copy

module mwd_returns

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_rr_states !%only: RR_StatesDT, RR_StatesDT_initialise

    implicit none

    type ReturnsDT

        integer :: nmts

        logical, dimension(:), allocatable :: mask_time_step
        integer, dimension(:), allocatable :: time_step_to_returns_time_step !$F90W index-array

        type(Rr_StatesDT), dimension(:), allocatable :: rr_states
        logical :: rr_states_flag = .false.

        real(sp), dimension(:, :, :), allocatable :: q_domain
        logical :: q_domain_flag = .false.

        real(sp) :: cost
        logical :: cost_flag = .false.

        real(sp) :: jobs
        logical :: jobs_flag = .false.

        real(sp) :: jreg
        logical :: jreg_flag = .false.

        real(sp) :: log_lkh
        logical :: log_lkh_flag = .false.

        real(sp) :: log_prior
        logical :: log_prior_flag = .false.

        real(sp) :: log_h
        logical :: log_h_flag = .false.

        real(sp), dimension(:, :, :, :), allocatable :: internal_fluxes
        logical :: internal_fluxes_flag = .false.

    end type ReturnsDT

contains

    subroutine ReturnsDT_initialise(this, setup, mesh, nmts, keys)

        implicit none

        type(ReturnsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: nmts
        character, dimension(:, :), intent(in) :: keys

        integer :: i, j
        character(lchar), dimension(size(keys, 2)) :: wkeys

        wkeys = ""
        do i = 1, size(keys, 2)
            do j = 1, size(keys, 1)
                wkeys(i) (j:j) = keys(j, i)
            end do
        end do

        this%nmts = nmts

        allocate (this%mask_time_step(setup%ntime_step))
        this%mask_time_step = .false.

        allocate (this%time_step_to_returns_time_step(setup%ntime_step))
        this%time_step_to_returns_time_step = -99

        ! Variable inside forward run are pre allocated
        ! Variable inside optimize will be allocated on the fly
        do i = 1, size(wkeys)

            select case (wkeys(i))

            case ("rr_states")
                this%rr_states_flag = .true.
                allocate (this%rr_states(this%nmts))
                do j = 1, this%nmts
                    call RR_StatesDT_initialise(this%rr_states(j), setup, mesh)
                end do

            case ("q_domain")
                this%q_domain_flag = .true.
                allocate (this%q_domain(mesh%nrow, mesh%ncol, this%nmts))
                this%q_domain = -99._sp

            case ("cost")
                this%cost_flag = .true.

            case ("jobs")
                this%jobs_flag = .true.

            case ("jreg")
                this%jreg_flag = .true.

            case ("log_lkh")
                this%log_lkh_flag = .true.

            case ("log_prior")
                this%log_prior_flag = .true.

            case ("log_h")
                this%log_h_flag = .true.

            case ("internal_fluxes")
                this%internal_fluxes_flag = .true.
                allocate (this%internal_fluxes(mesh%nrow, mesh%ncol, this%nmts, setup%n_internal_fluxes))

            end select

        end do

    end subroutine ReturnsDT_initialise

    subroutine ReturnsDT_copy(this, this_copy)

        implicit none

        type(ReturnsDT), intent(in) :: this
        type(ReturnsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ReturnsDT_copy

end module mwd_returns
