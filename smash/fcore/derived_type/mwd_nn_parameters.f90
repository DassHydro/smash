!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - NN_Parameters_LayerDT
!%          Layer containing weight and bias of the neural network
!%
!%          ======================== ========================================================
!%          `Variables`              Description
!%          ======================== ========================================================
!%          ``weight``               Transposed weight at current layer of the neural network
!%          ``bias``                 Bias at current layer of the neural network
!%          ======================== ========================================================
!%
!%      - NN_ParametersDT
!%          Contain multiple layers of the neural network
!%
!%          ======================== ========================================================
!%          `Variables`              Description
!%          ======================== ========================================================
!%          ``layers``               Layers containing weights and biases
!%          ``neurons``              Number of neurons of the neural network
!%          ``n_layers``             Number of layers of the neural network
!%          ======================== ========================================================
!%
!%      Subroutine
!%      ----------
!%
!%      - NN_Parameters_LayerDT_initialise
!%      - NN_Parameters_LayerDT_copy
!%      - NN_ParametersDT_initialise
!%      - NN_ParametersDT_copy

module mwd_nn_parameters

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT

    implicit none

    type NN_Parameters_LayerDT

        real(sp), dimension(:, :), allocatable :: weight
        real(sp), dimension(:), allocatable :: bias

    end type NN_Parameters_LayerDT

    type NN_ParametersDT

        type(NN_Parameters_LayerDT), dimension(:), allocatable :: layers
        integer, dimension(:), allocatable :: neurons
        integer :: n_layers

    end type NN_ParametersDT

contains

    subroutine NN_Parameters_LayerDT_initialise(this, n_neuron, n_in)

        implicit none

        type(NN_Parameters_LayerDT), intent(inout) :: this
        integer, intent(in) :: n_neuron
        integer, intent(in) :: n_in

        allocate (this%weight(n_neuron, n_in))
        this%weight = -99._sp

        allocate (this%bias(n_neuron))
        this%bias = -99._sp

    end subroutine NN_Parameters_LayerDT_initialise

    subroutine NN_Parameters_LayerDT_copy(this, this_copy)

        implicit none

        type(NN_Parameters_LayerDT), intent(in) :: this
        type(NN_Parameters_LayerDT), intent(out) :: this_copy

        this_copy = this

    end subroutine NN_Parameters_LayerDT_copy

    subroutine NN_ParametersDT_initialise(this, setup)

        implicit none

        type(NN_ParametersDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup

        integer :: i, n_in_layer
        integer :: n_in = 4  ! fixed NN input size
        integer :: n_out = 5  ! fixed NN output size

        this%n_layers = setup%nhl + 1

        allocate (this%layers(this%n_layers))
        allocate (this%neurons(this%n_layers + 1))

        this%neurons(1) = n_in

        n_in_layer = n_in

        do i = 1, setup%nhl

            call NN_Parameters_LayerDT_initialise(this%layers(i), setup%hidden_neuron(i), n_in_layer)
            n_in_layer = setup%hidden_neuron(i)

            this%neurons(i + 1) = setup%hidden_neuron(i)

        end do

        call NN_Parameters_LayerDT_initialise(this%layers(this%n_layers), n_out, n_in_layer)

        this%neurons(this%n_layers + 1) = n_out

    end subroutine NN_ParametersDT_initialise

    subroutine NN_ParametersDT_copy(this, this_copy)

        implicit none

        type(NN_ParametersDT), intent(in) :: this
        type(NN_ParametersDT), intent(out) :: this_copy

        this_copy = this

    end subroutine NN_ParametersDT_copy

end module mwd_nn_parameters
