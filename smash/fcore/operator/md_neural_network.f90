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

    subroutine forward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, input_layer, output_layer)
        !% The forward pass of the multilayer perceptron used in hydrological model structure

        implicit none

        real(sp), dimension(:, :), intent(in) :: weight_1
        real(sp), dimension(:), intent(in) :: bias_1

        real(sp), dimension(:, :), intent(in) :: weight_2
        real(sp), dimension(:), intent(in) :: bias_2

        real(sp), dimension(:, :), intent(in) :: weight_3
        real(sp), dimension(:), intent(in) :: bias_3

        real(sp), dimension(:), intent(in) :: input_layer
        real(sp), dimension(:), intent(out) :: output_layer

        real(sp), dimension(size(bias_1)) :: inter_layer_1
        real(sp), dimension(size(bias_2)) :: inter_layer_2
        integer :: i

        call dot_product_2d_1d(weight_1, input_layer, inter_layer_1)
        do i = 1, size(inter_layer_1)
            inter_layer_1(i) = inter_layer_1(i) + bias_1(i)
            inter_layer_1(i) = max(0.01_sp*inter_layer_1(i), inter_layer_1(i)) ! Leaky ReLU
        end do

        if (size(bias_3) .gt. 0) then  ! in case of having 3 layers

            call dot_product_2d_1d(weight_2, inter_layer_1, inter_layer_2)
            do i = 1, size(inter_layer_2)
                inter_layer_2(i) = inter_layer_2(i) + bias_2(i)
                inter_layer_2(i) = max(0.01_sp*inter_layer_2(i), inter_layer_2(i)) ! Leaky ReLU
            end do

            call dot_product_2d_1d(weight_3, inter_layer_2, output_layer)
            do i = 1, size(output_layer)
                output_layer(i) = tanh(output_layer(i) + bias_3(i)) ! TanH
            end do

        else  ! in case of having 2 layers

            call dot_product_2d_1d(weight_2, inter_layer_1, output_layer)
            do i = 1, size(output_layer)
                output_layer(i) = tanh(output_layer(i) + bias_2(i)) ! TanH
            end do

        end if

    end subroutine forward_mlp

end module md_neural_network
