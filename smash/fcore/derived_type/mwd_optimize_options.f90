!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Optimize_OptionsDT
!%          Optimization options passed by user to define the 'parameters-to-control' mapping,
!%          parameters to optimize and optimizer options (factr, pgtol, bounds)
!%
!%          ================================== =======================================
!%          `Variables`                        Description
!%          ================================== =======================================
!%          ``mapping``                       Control mapping name
!%          ``optimizer``                     Optimizer name
!%          ``control_tfm``                   Type of transformation applied to control
!%          ``rr_parameters``                 Rr parameters to optimize
!%          ``l_rr_parameters``               Rr parameters lower bound
!%          ``u_rr_parameters``               Rr parameters upper bound
!%          ``rr_parameters_descriptor``      Rr parameters descriptor to use
!%          ``rr_initial_states``             Rr initial states to optimize
!%          ``l_rr_initial_states``           Rr initial states lower bound
!%          ``u_rr_initial_states``           Rr initial states upper bound
!%          ``rr_initial_states_descriptor``  Rr initial states descriptor use
!%          ``serr_mu_parameters``            Serr mu parameters to optimize
!%          ``l_serr_mu_parameters``          Serr mu parameters lower bound
!%          ``u_serr_mu_parameters``          Serr mu parameters upper bound
!%          ``serr_sigma_parameters``         Serr sigma parameters to optimize
!%          ``l_serr_sigma_parameters``       Serr sigma parameters lower bound
!%          ``u_serr_sigma_parameters``       Serr sigma parameters upper bound
!%          ``maxiter``                       Maximum number of iterations
!%          ``factr``                         LBFGSB cost function criterion
!%          ``pgtol``                         LBFGSB gradient criterion
!%          ================================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Optimize_OptionsDT_initialise
!%      - Optimize_OptionsDT_copy

module mwd_optimize_options

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT

    implicit none

    type Optimize_OptionsDT

        character(lchar) :: mapping = "..." !$F90W char
        character(lchar) :: optimizer = "..." !$F90W char
        character(lchar) :: control_tfm = "..." !$F90W char

        integer, dimension(:), allocatable :: rr_parameters
        real(sp), dimension(:), allocatable :: l_rr_parameters
        real(sp), dimension(:), allocatable :: u_rr_parameters
        integer, dimension(:, :), allocatable :: rr_parameters_descriptor

        integer, dimension(:), allocatable :: rr_initial_states
        real(sp), dimension(:), allocatable :: l_rr_initial_states
        real(sp), dimension(:), allocatable :: u_rr_initial_states
        integer, dimension(:, :), allocatable :: rr_initial_states_descriptor

        integer, dimension(:), allocatable :: serr_mu_parameters
        real(sp), dimension(:), allocatable :: l_serr_mu_parameters
        real(sp), dimension(:), allocatable :: u_serr_mu_parameters

        integer, dimension(:), allocatable :: serr_sigma_parameters
        real(sp), dimension(:), allocatable :: l_serr_sigma_parameters
        real(sp), dimension(:), allocatable :: u_serr_sigma_parameters

        integer :: maxiter = -99
        real(sp) :: factr = -99._sp
        real(sp) :: pgtol = -99._sp

    end type Optimize_OptionsDT

contains

    subroutine Optimize_OptionsDT_initialise(this, setup)

        implicit none

        type(Optimize_OptionsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup

        allocate (this%rr_parameters(setup%nop))
        this%rr_parameters = -99

        allocate (this%l_rr_parameters(setup%nop))
        this%l_rr_parameters = -99._sp

        allocate (this%u_rr_parameters(setup%nop))
        this%u_rr_parameters = -99._sp

        allocate (this%rr_parameters_descriptor(setup%nd, setup%nop))
        this%rr_parameters_descriptor = -99

        allocate (this%rr_initial_states(setup%nos))
        this%rr_initial_states = -99

        allocate (this%l_rr_initial_states(setup%nos))
        this%l_rr_initial_states = -99._sp

        allocate (this%u_rr_initial_states(setup%nos))
        this%u_rr_initial_states = -99._sp

        allocate (this%rr_initial_states_descriptor(setup%nd, setup%nos))
        this%rr_initial_states_descriptor = -99

        allocate (this%serr_mu_parameters(setup%nsep_mu))
        this%serr_mu_parameters = -99

        allocate (this%l_serr_mu_parameters(setup%nsep_mu))
        this%l_serr_mu_parameters = -99._sp

        allocate (this%u_serr_mu_parameters(setup%nsep_mu))
        this%u_serr_mu_parameters = -99._sp

        allocate (this%serr_sigma_parameters(setup%nsep_sigma))
        this%serr_sigma_parameters = -99

        allocate (this%l_serr_sigma_parameters(setup%nsep_sigma))
        this%l_serr_sigma_parameters = -99._sp

        allocate (this%u_serr_sigma_parameters(setup%nsep_sigma))
        this%u_serr_sigma_parameters = -99._sp

    end subroutine Optimize_OptionsDT_initialise

    subroutine Optimize_OptionsDT_copy(this, this_copy)

        implicit none

        type(Optimize_OptionsDT), intent(in) :: this
        type(Optimize_OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Optimize_OptionsDT_copy

end module mwd_optimize_options
