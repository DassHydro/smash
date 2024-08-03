!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - NN_ParametersDT
!%          Contain weights and biases of the neural network
!%
!%          ======================== ===========================================================
!%          `Variables`              Description
!%          ======================== ===========================================================
!%          ``weight_1``             Transposed weight at the first layer of the neural network
!%          ``bias_1``               Bias at the first layer of the neural network
!%          ``weight_2``             Transposed weight at the second layer of the neural network
!%          ``bias_2``               Bias at the second layer of the neural network
!%          ``weight_3``             Transposed weight at the third layer of the neural network
!%          ``bias_3``               Bias at the third layer of the neural network
!%          ======================== ===========================================================
!%
!%      Subroutine
!%      ----------
!%
!%      - NN_ParametersDT_initialise
!%      - NN_ParametersDT_copy

module mwd_nn_parameters

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT

    implicit none

    type NN_ParametersDT

        real(sp), dimension(:, :), allocatable :: weight_1
        real(sp), dimension(:), allocatable :: bias_1

        real(sp), dimension(:, :), allocatable :: weight_2
        real(sp), dimension(:), allocatable :: bias_2

        real(sp), dimension(:, :), allocatable :: weight_3
        real(sp), dimension(:), allocatable :: bias_3

    end type NN_ParametersDT

contains

    subroutine NN_ParametersDT_initialise(this, setup)

        implicit none

        type(NN_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup

        !% First layer
        allocate (this%weight_1(setup%neurons(2), setup%neurons(1)))
        this%weight_1 = -99._sp

        allocate (this%bias_1(setup%neurons(2)))
        this%bias_1 = -99._sp

        !% Second layer
        allocate (this%weight_2(setup%neurons(3), setup%neurons(2)))
        this%weight_2 = -99._sp

        allocate (this%bias_2(setup%neurons(3)))
        this%bias_2 = -99._sp

        !% Third layer
        allocate (this%weight_3(setup%neurons(4), setup%neurons(3)))
        this%weight_3 = -99._sp

        allocate (this%bias_3(setup%neurons(4)))
        this%bias_3 = -99._sp

    end subroutine NN_ParametersDT_initialise

    subroutine NN_ParametersDT_copy(this, this_copy)

        implicit none

        type(NN_ParametersDT), intent(in) :: this
        type(NN_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine NN_ParametersDT_copy

end module mwd_nn_parameters
