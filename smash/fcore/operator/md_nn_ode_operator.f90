!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr_ode
!%      - feedforward_nn
!%      - gr_neural_ode

module md_nn_ode_operator

    use md_constant !% only : sp
    use md_algebra !% only: solve_linear_system_2vars, dot_product_2d_1d
    use mwd_nn_parameters !% only: NN_ParametersDT

    implicit none

contains

    ! subroutine gr_explicit_ode(pn, en, cp, ct, kexc, hp, ht, qti)
    !     ! % Solve state-space ODE system with explicit Euler

    !     implicit none

    !     real(sp), intent(in) :: pn, en, cp, ct, kexc
    !     real(sp), intent(inout) :: hp, ht, qti

    !     hp = hp + ((1._sp - hp**2)*pn - hp*(2._sp - hp)*en)/cp
    !     ht = ht + (0.9_sp*pn*hp**2 - ct*ht**5 + kexc*ht**3.5_sp)/ct

    !     qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    ! end subroutine gr_explicit_ode

    subroutine gr_ode(pn, en, cp, ct, kexc, hp, ht, qti)
        !% Solve state-space ODE system with implicit Euler

        implicit none

        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, qti

        real(sp), dimension(2, 2) :: jacob
        real(sp), dimension(2) :: dh, delta_h
        real(sp) :: hp0, ht0
        integer :: i
        integer :: maxiter = 20

        hp0 = hp
        ht0 = ht

        do i = 1, maxiter

            dh(1) = hp - hp0 - ((1._sp - hp**2)*pn - hp*(2._sp - hp)*en)/cp
            dh(2) = ht - ht0 - (0.9_sp*pn*hp**2 - ct*ht**5 + kexc*ht**3.5_sp)/ct

            jacob(1, 1) = 1._sp + 2._sp*(hp*(pn - en) + en)/cp
            jacob(1, 2) = 0._sp
            jacob(2, 1) = 1.8_sp*pn*hp/ct
            jacob(2, 2) = 1._sp + (5._sp*ht**4 - 3.5_sp*kexc*(ht**2.5_sp)/ct)

            call solve_linear_system_2vars(jacob, delta_h, dh)

            hp = hp + delta_h(1)
            ht = ht + delta_h(2)

            if (sqrt((delta_h(1)/hp)**2 + (delta_h(2)/ht)**2) .lt. 1.e-6_sp) exit

        end do

        qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_ode

    subroutine feedforward_nn(layers, neurons, input_layer, output_layer)
        !% The forward pass of the neural network used in hydrological model structure

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
                    f_out(j) = max(0._sp, f_out(j)) ! ReLU
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

    end subroutine feedforward_nn

    subroutine gr_neural_ode(layers, neurons, pn, en, cp, ct, kexc, hp, ht, qti)
        !% Solve state-space ODE system with explicit Euler and neural network

        implicit none

        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, qti

        real(sp), dimension(4) :: input_layer  ! fixed NN input size
        real(sp), dimension(5) :: output_layer  ! fixed NN output size
        real(sp) :: dhp, dht

        input_layer(1) = hp
        input_layer(2) = ht
        input_layer(3) = pn
        input_layer(4) = en
        ! input_layer(5) = cp
        ! input_layer(6) = ct
        ! input_layer(7) = kexc

        call feedforward_nn(layers, neurons, input_layer, output_layer)

        dhp = (output_layer(1)*pn*(1._sp - hp**2) - &
        & output_layer(2)*en*hp*(2._sp - hp))/cp

        if (dhp .lt. 0) then
            dhp = max(dhp, -hp) + 1.e-6_sp
        else
            dhp = min(dhp, 1._sp - hp) - 1.e-6_sp
        end if

        hp = hp + dhp

        dht = (output_layer(3)*0.9_sp*pn*hp**2 - &
        & output_layer(4)*ct*ht**5 + &
        & output_layer(5)*kexc*ht**3.5_sp)/ct

        if (dht .lt. 0) then
            dht = max(dht, -ht) + 1.e-8_sp
        else
            dht = min(dht, 1._sp - ht) - 1.e-8_sp
        end if
        
        ht = ht + dht

        qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_neural_ode

end module md_nn_ode_operator
