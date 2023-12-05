!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - feedforward_mlp

module md_neural_network

    use md_constant !% only : sp
    use md_algebra !% only: dot_product_2d_1d
    use mwd_nn_parameters !% only: NN_Parameters_LayerDT

    implicit none

contains

    subroutine feedforward_mlp(layers, neurons, input_layer, output_layer)
        !% The forward pass of the multilayer perceptron used in hydrological model structure

        implicit none

        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), dimension(:), intent(in) :: input_layer
        real(sp), dimension(:), intent(out) :: output_layer

        real(sp), dimension(maxval(neurons)) :: f_in, f_out
        integer :: i, j

        do i = 1, neurons(1)
            f_in(i) = input_layer(i)
        end do

        do i = 1, size(layers)

            call dot_product_2d_1d(layers(i)%weight, f_in, f_out)

            do j = 1, neurons(i + 1)
                f_out(j) = f_out(j) + layers(i)%bias(j)

                if (i .lt. size(layers)) then
                    f_out(j) = max(0.01_sp*f_out(j), f_out(j)) ! Leaky ReLU
                else
                    f_out(j) = 2._sp/(1._sp + exp(-f_out(j))) ! Softmax*2
                end if

            end do

            do j = 1, neurons(i)
                f_in(j) = f_out(j)
            end do

        end do

        do i = 1, neurons(size(neurons))
            output_layer(i) = f_out(i)
        end do

    end subroutine feedforward_mlp

end module md_neural_network
