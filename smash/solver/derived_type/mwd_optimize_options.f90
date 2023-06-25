!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Optimize_OptionsDT
!%
!%          ================================== =======================================
!%          `Variables`                        Description
!%          ================================== =======================================
!%          ``mapping``                       Control mapping name                       (default: "...")
!%          ``optimizer``                     Optimizer name                             (default: "...")
!%          ``control_tfm``                   Type of transformation applied to control  (default: "...")
!%          ``maxiter``                       Maximum number of iterations               (default: 100)
!%          ``opr_parameters``                Opr parameters to optimize
!%          ``opr_initial_states``            Opr initial states to optimize
!%          ``l_opr_parameters``              Opr parameters lower bound
!%          ``u_opr_parameters``              Opr parameters upper bound
!%          ``l_opr_initial_states``          Opr initial states lower bound
!%          ``u_opr_initial_states``          Opr initial states upper bound
!%          ``opr_parameters_descriptor``     Opr parameters descriptor to use
!%          ``opr_initial_states_descriptor`` Opr initial states descriptor use
!%          ``opr_parameters_tfunc``          Opr parameters transformation function
!%          ``opr_initial_states_tfunc``      Opr initial states transformation function
!%          ================================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Optimize_OptionsDT_initialise
!%      - Optimize_OptionsDT_copy

module mwd_optimize_options

    use md_constant !% only: sp, lchar, nopr_parameters, nopr_states
    use mwd_setup !% only: SetupDT

    implicit none

    type Optimize_OptionsDT
            
        character(lchar) :: mapping = "..." !$F90W char
        character(lchar) :: optimizer = "..." !$F90W char
        character(lchar) :: control_tfm = "..." !$F90W char
        
        integer :: maxiter = 100
        
        integer, dimension(nopr_parameters) :: opr_parameters = 1
        integer, dimension(nopr_states) :: opr_initial_states = 0
        
        real(sp), dimension(nopr_parameters) :: l_opr_parameters = 0._sp
        real(sp), dimension(nopr_parameters) :: u_opr_parameters = 1._sp
        
        real(sp), dimension(nopr_states) :: l_opr_initial_states = 0._sp
        real(sp), dimension(nopr_states) :: u_opr_initial_states = 1._sp
        
        integer, dimension(:, :), allocatable :: opr_parameters_descriptor
        integer, dimension(:, :), allocatable :: opr_initial_states_descriptor

    end type Optimize_OptionsDT

contains

    subroutine Optimize_OptionsDT_initialise(this, setup)

        implicit none

        type(Optimize_OptionsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        
        allocate (this%opr_parameters_descriptor(setup%nd, nopr_parameters))
        this%opr_parameters_descriptor = 1
        
        allocate (this%opr_initial_states_descriptor(setup%nd, nopr_states))
        this%opr_initial_states_descriptor = 1
        
    end subroutine Optimize_OptionsDT_initialise

    subroutine Optimize_OptionsDT_copy(this, this_copy)

        implicit none

        type(Optimize_OptionsDT), intent(in) :: this
        type(Optimize_OptionsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Optimize_OptionsDT_copy

end module mwd_optimize_options
