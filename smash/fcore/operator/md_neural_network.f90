!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - forward_mlp

module md_neural_network

    use md_constant !% only : sp
    use md_algebra !% only: dot_product_2d_1d

    implicit none

contains

    subroutine forward_mlp(weight_1, bias_1, weight_2, bias_2, input_layer, output_layer)
        !% The forward pass of the multilayer perceptron used in hydrological model structure

        implicit none

        real(sp), dimension(:, :), intent(in) :: weight_1
        real(sp), dimension(:), intent(in) :: bias_1

        real(sp), dimension(:, :), intent(in) :: weight_2
        real(sp), dimension(:), intent(in) :: bias_2

        real(sp), dimension(:), intent(in) :: input_layer
        real(sp), dimension(:), intent(out) :: output_layer

        real(sp), dimension(size(bias_1)) :: inter_layer
        integer :: i

        call dot_product_2d_1d(weight_1, input_layer, inter_layer)
        do i = 1, size(inter_layer)
            inter_layer(i) = inter_layer(i) + bias_1(i)
            inter_layer(i) = max(0.01_sp*inter_layer(i), inter_layer(i)) ! Leaky ReLU
        end do

        call dot_product_2d_1d(weight_2, inter_layer, output_layer)
        do i = 1, size(output_layer)
            output_layer(i) = output_layer(i) + bias_2(i)
            output_layer(i) = 2._sp/(1._sp + exp(-output_layer(i))) ! Softmax*2
        end do

    end subroutine forward_mlp

end module md_neural_network
