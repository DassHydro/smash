!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - forward_mlp
!%      - forward_and_backward_mlp

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

        call dot_product_2d_1d(weight_1, input_layer, inter_layer_1)
        inter_layer_1 = inter_layer_1 + bias_1
        inter_layer_1 = inter_layer_1*(1._sp/(1._sp + exp(-inter_layer_1))) ! SiLU

        if (size(bias_3) .gt. 0) then  ! Case with 3 layers

            call dot_product_2d_1d(weight_2, inter_layer_1, inter_layer_2)
            inter_layer_2 = inter_layer_2 + bias_2
            inter_layer_2 = inter_layer_2*(1._sp/(1._sp + exp(-inter_layer_2))) ! SiLU

            call dot_product_2d_1d(weight_3, inter_layer_2, output_layer)
            output_layer = tanh(output_layer + bias_3) ! TanH

        else  ! Case with 2 layers

            call dot_product_2d_1d(weight_2, inter_layer_1, output_layer)
            output_layer = tanh(output_layer + bias_2) ! TanH

        end if

    end subroutine forward_mlp

    subroutine forward_and_backward_mlp(weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, &
    & input_layer, output_layer, output_jacobian_1, output_jacobian_2)
        !% The forward pass and backward pass of the MLP used in hydrological model structure
        !% also get the jacobian of outputs wrt the first two inputs

        implicit none

        real(sp), dimension(:, :), intent(in) :: weight_1
        real(sp), dimension(:), intent(in)    :: bias_1

        real(sp), dimension(:, :), intent(in) :: weight_2
        real(sp), dimension(:), intent(in)    :: bias_2

        real(sp), dimension(:, :), intent(in) :: weight_3
        real(sp), dimension(:), intent(in)    :: bias_3

        real(sp), dimension(:), intent(in)    :: input_layer

        real(sp), dimension(:), intent(out)   :: output_layer
        real(sp), dimension(:), intent(out) :: output_jacobian_1
        real(sp), dimension(:), intent(out) :: output_jacobian_2

        real(sp), dimension(size(bias_1)) :: inter_layer_1, inter_layer_1_tf, inter_layer_1_grad, layer_1_grad
        real(sp), dimension(size(bias_2)) :: inter_layer_2, inter_layer_2_tf, inter_layer_2_grad, layer_2_grad
        integer :: i, j, k

        output_jacobian_1 = 0._sp
        output_jacobian_2 = 0._sp

        call dot_product_2d_1d(weight_1, input_layer, inter_layer_1)
        inter_layer_1 = inter_layer_1 + bias_1
        inter_layer_1_tf = inter_layer_1*(1._sp/(1._sp + exp(-inter_layer_1))) ! SiLU
        inter_layer_1_grad = inter_layer_1_tf + &
        & (1._sp - inter_layer_1_tf)/(1._sp + exp(-inter_layer_1))  ! Derivative of SiLU

        if (size(bias_3) .gt. 0) then  ! Case with 3 layers
            call dot_product_2d_1d(weight_2, inter_layer_1_tf, inter_layer_2)
            inter_layer_2 = inter_layer_2 + bias_2
            inter_layer_2_tf = inter_layer_2*(1._sp/(1._sp + exp(-inter_layer_2))) ! SiLU
            inter_layer_2_grad = inter_layer_2_tf + &
            & (1._sp - inter_layer_2_tf)/(1._sp + exp(-inter_layer_2))  ! Derivative of SiLU

            call dot_product_2d_1d(weight_3, inter_layer_2_tf, output_layer)
            output_layer = tanh(output_layer + bias_3)  ! TanH

            ! Compute Jacobian matrix of output wrt input MLP
            do i = 1, size(output_layer)
                do j = 1, size(inter_layer_2)
                    layer_2_grad(j) = (1._sp - output_layer(i)**2)*weight_3(i, j)  ! Derivative of TanH
                    layer_2_grad(j) = layer_2_grad(j)*inter_layer_2_grad(j)
                end do

                ! Gradient of second layer wrt first layer
                layer_1_grad = 0._sp
                do j = 1, size(inter_layer_1)
                    do k = 1, size(inter_layer_2)
                        layer_1_grad(j) = layer_1_grad(j) + layer_2_grad(k)*weight_2(k, j)
                    end do
                    layer_1_grad(j) = layer_1_grad(j)*inter_layer_1_grad(j)
                end do

                ! Gradient of first layer wrt input layer
                do k = 1, size(inter_layer_1)
                    output_jacobian_1(i) = output_jacobian_1(i) + layer_1_grad(k)*weight_1(k, 1)
                    output_jacobian_2(i) = output_jacobian_2(i) + layer_1_grad(k)*weight_1(k, 2)
                end do
            end do

        else  ! Case with 2 layers
            call dot_product_2d_1d(weight_2, inter_layer_1_tf, output_layer)
            output_layer = tanh(output_layer + bias_2)

            ! Compute Jacobian matrix of output wrt input MLP
            do i = 1, size(output_layer)
                do j = 1, size(inter_layer_1)
                    layer_1_grad(j) = (1._sp - output_layer(i)**2)*weight_2(i, j)  ! Derivative of TanH
                    layer_1_grad(j) = layer_1_grad(j)*inter_layer_1_grad(j)
                end do

                ! Gradient of first layer wrt input layer
                do k = 1, size(inter_layer_1)
                    output_jacobian_1(i) = output_jacobian_1(i) + layer_1_grad(k)*weight_1(k, 1)
                    output_jacobian_2(i) = output_jacobian_2(i) + layer_1_grad(k)*weight_1(k, 2)
                end do
            end do

        end if

    end subroutine forward_and_backward_mlp

end module md_neural_network
