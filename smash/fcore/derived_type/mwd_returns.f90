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
!%          ``iter_cost``            Array of cost iteration
!%          ``iter_cost_flag``       Return flag of iter_cost
!%          ``iter_projg``           Array of infinity norm of projected gradient iteration
!%          ``iter_projg_flag``      Return flag of iter_projg
!%          ``control_vector``       Array of control vector
!%          ``control_vector_flag``  Return flag of control_vector
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
!%          ``serr_mu``              Serr mu value
!%          ``serr_mu_flag``         Return flag of serr_mu
!%          ``serr_sigma``           Serr sigma value
!%          ``serr_sigma_flag``      Return flag of serr_sigma
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
    use mwd_rr_states !%only: Rr_StatesDT

    implicit none

    type ReturnsDT

        integer :: nmts

        logical, dimension(:), allocatable :: mask_time_step

        type(Rr_StatesDT), dimension(:), allocatable :: rr_states
        logical :: rr_states_flag = .false.

        real(sp), dimension(:, :, :), allocatable :: q_domain
        logical :: q_domain_flag = .false.

        real(sp), dimension(:), allocatable :: iter_cost
        logical :: iter_cost_flag = .false.

        real(sp), dimension(:), allocatable :: iter_projg
        logical :: iter_projg_flag = .false.

        real(sp), dimension(:), allocatable :: control_vector
        logical :: control_vector_flag = .false.

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

        real(sp), dimension(:, :), allocatable :: serr_mu
        logical :: serr_mu_flag = .false.

        real(sp), dimension(:, :), allocatable :: serr_sigma
        logical :: serr_sigma_flag = .false.

        real(sp), dimension(:, :, :), allocatable :: qt
        logical :: qt_flag = .false.
        
        ! internal fluxes
!~         real(sp), dimension(:, :, :), allocatable :: internal_fluxes
!~         logical :: internal_fluxes_flag = .false.

        real(sp), dimension(:, :), allocatable :: pn
        logical :: pn_flag = .false.

        real(sp), dimension(:, :), allocatable :: en
        logical :: en_flag = .false.

        real(sp), dimension(:, :), allocatable :: pr
        logical :: pr_flag = .false.

        real(sp), dimension(:, :), allocatable :: perc
        logical :: perc_flag = .false.

        real(sp), dimension(:, :), allocatable :: lexc
        logical :: lexc_flag = .false.

        real(sp), dimension(:, :), allocatable :: prr
        logical :: prr_flag = .false.

        real(sp), dimension(:, :), allocatable :: prd
        logical :: prd_flag = .false.

        real(sp), dimension(:, :), allocatable :: qr
        logical :: qr_flag = .false.
        
        real(sp), dimension(:, :), allocatable :: qd
        logical :: qd_flag = .false.

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

        ! Variable inside forward run are pre allocated
        ! Variable inside optimize will be allocated on the fly
        do i = 1, size(wkeys)

            select case (wkeys(i))

            case ("rr_states")
                this%rr_states_flag = .true.
                allocate (this%rr_states(this%nmts))

            case ("q_domain")
                this%q_domain_flag = .true.
                allocate (this%q_domain(mesh%nrow, mesh%ncol, this%nmts))

            case ("iter_cost")
                this%iter_cost_flag = .true.

            case ("iter_projg")
                this%iter_projg_flag = .true.

            case ("control_vector")
                this%control_vector_flag = .true.

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

            case ("serr_mu")
                this%serr_mu_flag = .true.
                allocate (this%serr_mu(mesh%ng, setup%ntime_step))

            case ("serr_sigma")
                this%serr_sigma_flag = .true.
                allocate (this%serr_sigma(mesh%ng, setup%ntime_step))

            case ("qt")
                this%qt_flag = .true.
                allocate (this%qt(mesh%nrow, mesh%ncol, this%nmts))
            
            ! internal fluxes
!~             case ("internal_fluxes")
!~                 this%internal_fluxes_flag = .true.
!~                 this%internal_fluxes%pn(mesh%nrow, mesh%ncol)
            end select
            
            if (setup%hydrological_module == "gr4") then
                ! GR4
                select case (wkeys(i))
                case ("pn")
                    this%pn_flag = .true.
                    allocate (this%pn(mesh%nrow, mesh%ncol))
                    
                case ("en")
                    this%en_flag = .true.
                    allocate (this%en(mesh%nrow, mesh%ncol))
                    
                case ("pr")
                    this%pr_flag = .true.
                    allocate (this%pr(mesh%nrow, mesh%ncol))

                case ("perc")
                    this%perc_flag = .true.
                    allocate (this%perc(mesh%nrow, mesh%ncol))
                    
                case ("lexc")
                    this%lexc_flag = .true.
                    allocate (this%lexc(mesh%nrow, mesh%ncol))
                    
                case ("prr")
                    this%prr_flag = .true.
                    allocate (this%prr(mesh%nrow, mesh%ncol))
                    
                case ("prd")
                    this%prd_flag = .true.
                    allocate (this%prd(mesh%nrow, mesh%ncol))
                
                case ("qr")
                    this%qr_flag = .true.
                    allocate (this%qr(mesh%nrow, mesh%ncol))
                    
                case ("qd")
                    this%qd_flag = .true.
                    allocate (this%qd(mesh%nrow, mesh%ncol))
                
                end select
            end if    
        end do

    end subroutine ReturnsDT_initialise

    subroutine ReturnsDT_copy(this, this_copy)

        implicit none

        type(ReturnsDT), intent(in) :: this
        type(ReturnsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine ReturnsDT_copy

end module mwd_returns
