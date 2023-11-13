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

    subroutine gr_ode(pn, en, cp, ct, kexc, hp, ht, qti)
        !% Solve state-space ODE system with implicit Euler

        implicit none

        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, qti

        real(sp), dimension(2, 2) :: jacob
        real(sp), dimension(2) :: dh, delta_h
        real(sp) :: hp0, ht0, dt
        integer :: i, j
        integer :: n_subtimesteps = 2
        integer :: maxiter = 10
        
        dt = 1._sp/real(n_subtimesteps, sp)

        do i = 1, n_subtimesteps

            hp0 = hp
            ht0 = ht

            do j = 1, maxiter

                dh(1) = hp - hp0 - dt*((1._sp - hp**2)*pn - hp*(2._sp - hp)*en)/cp
                dh(2) = ht - ht0 - dt*(0.9_sp*pn*hp**2 - ct*ht**5 + kexc*ht**3.5_sp)/ct

                jacob(1, 1) = 1._sp + dt*2._sp*(hp*(pn - en) + en)/cp
                jacob(1, 2) = 0._sp
                jacob(2, 1) = dt*1.8_sp*pn*hp/ct
                jacob(2, 2) = 1._sp + dt*(5._sp*ht**4 - 3.5_sp*kexc*(ht**2.5_sp)/ct)

                call solve_linear_system_2vars(jacob, delta_h, dh)

                hp = hp + delta_h(1)
                if (hp .le. 0._sp) hp = 1.e-6_sp
                if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp
                
                ht = ht + delta_h(2)
                if (ht .le. 0._sp) ht = 1.e-6_sp
                if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

                if (sqrt((delta_h(1)/hp)**2 + (delta_h(2)/ht)**2) .lt. 1.e-6_sp) exit

            end do

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
        real(sp) :: dt
        integer :: i
        integer :: n_subtimesteps = 1

        input_layer(1) = hp
        input_layer(2) = ht
        input_layer(3) = pn
        input_layer(4) = en

        call feedforward_nn(layers, neurons, input_layer, output_layer)
        
        dt = 1._sp/real(n_subtimesteps, sp)

        do i = 1, n_subtimesteps

            hp = hp + dt*(output_layer(1)*pn*(1._sp - hp**2) - &
            & output_layer(2)*en*hp*(2._sp - hp))/cp
            if (hp .le. 0._sp) hp = 1.e-6_sp
            if (hp .ge. 1._sp) hp = 1._sp - 1.e-6_sp

            ht = ht + dt*(output_layer(3)*0.9_sp*pn*hp**2 - &
            & output_layer(4)*ct*ht**5 + &
            & output_layer(5)*kexc*ht**3.5_sp)/ct
            if (ht .le. 0._sp) ht = 1.e-6_sp
            if (ht .ge. 1._sp) ht = 1._sp - 1.e-6_sp

        end do

        qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_neural_ode

    subroutine gr_nn_alg(layers, neurons, pn, en, cp, beta, ct, kexc, n, prcp, hp, ht, qti)
        !% Integrate neural networks in stepwise aprroximation method

        implicit none

        type(NN_Parameters_LayerDT), dimension(:), intent(inout) :: layers
        integer, dimension(:), intent(in) :: neurons
        real(sp), intent(in) :: pn, en, cp, beta, ct, kexc, n, prcp
        real(sp), intent(inout) :: hp, ht, qti

        real(sp), dimension(4) :: input_layer  ! fixed NN input size
        real(sp), dimension(5) :: output_layer  ! fixed NN output size
        real(sp) :: pr, perc, ps, es, hp_imd, pr_imd, ht_imd, nm1, d1pnm1, qd

        input_layer(1) = hp
        input_layer(2) = ht
        input_layer(3) = pn
        input_layer(4) = en

        call feedforward_nn(layers, neurons, input_layer, output_layer)

        pr = 0._sp

        ps = cp*(1._sp - hp*hp)*tanh(pn/cp)/ &
        & (1._sp + hp*tanh(pn/cp))

        es = (hp*cp)*(2._sp - hp)*tanh(en/cp)/ &
        & (1._sp + (1._sp - hp)*tanh(en/cp))

        hp_imd = hp + (output_layer(1)*ps - output_layer(2)*es)/cp

        if (pn .gt. 0) then

            pr = pn - (hp_imd - hp)*cp

        end if

        perc = (hp_imd*cp)*(1._sp - (1._sp + (hp_imd/beta)**4)**(-0.25_sp))

        hp = hp_imd - perc/cp

        nm1 = n - 1._sp
        d1pnm1 = 1._sp/nm1

        if (prcp .lt. 0._sp) then

            pr_imd = ((ht*ct)**(-nm1) - ct**(-nm1))**(-d1pnm1) - (ht*ct)

        else

            pr_imd = 0.9_sp*(output_layer(3)*pr + output_layer(4)*perc) &
            & + output_layer(5)*kexc*(ht**3.5_sp)

        end if

        ht_imd = max(1.e-6_sp, ht + pr_imd/ct)

        ht = (((ht_imd*ct)**(-nm1) + ct**(-nm1))**(-d1pnm1))/ct

        qd = 0.1_sp*(output_layer(3)*pr + output_layer(4)*perc) &
        & + output_layer(5)*kexc*(ht**3.5_sp)

        qti = (ht_imd - ht)*ct + max(0._sp, qd)

    end subroutine gr_nn_alg

end module md_nn_ode_operator
