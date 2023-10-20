!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr_ode_implicit_euler
!%      - feedforward_nn

module md_neural_ode_operator

    use md_constant !% only : sp
    use md_algebra !% only: solve_linear_system_2vars, multiply_matrix_2d_1d
    use mwd_nn_parameters !% only: NN_ParametersDT

    implicit none

contains

    !% TODO comment
    subroutine gr_ode_implicit_euler(pn, en, cp, ct, kexc, hp, ht, qti)

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
                ht = ht + delta_h(2)

                if (sqrt((delta_h(1)/hp)**2 + (delta_h(2)/ht)**2) .lt. 1.e-6_sp) exit

            end do

        end do

        qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_ode_implicit_euler

    ! subroutine gr_ode_explicit_euler(pn, en, cp, ct, kexc, hp, ht, qti)

    !     implicit none

    !     real(sp), intent(in) :: pn, en, cp, ct, kexc
    !     real(sp), intent(inout) :: hp, ht, qti

    !     real(sp) :: hp_dot, ht_dot, dt
    !     integer :: i
    !     integer :: n_subtimesteps = 20

    !     dt = 1._sp/real(n_subtimesteps, sp)

    !     do i = 1, n_subtimesteps

    !         hp_dot = ((1._sp - hp**2)*pn - hp*(2._sp - hp)*en)/cp
    !         ht_dot = (0.9_sp*pn*hp**2 - ct*ht**5 + kexc*ht**3.5_sp)/ct

    !         hp = hp + dt*hp_dot
    !         ht = ht + dt*ht_dot

    !     end do

    !     qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    ! end subroutine gr_ode_explicit_euler

    subroutine feedforward_nn(nn, x, y)

        implicit none

        type(NN_ParametersDT), intent(inout) :: nn
        real(sp), dimension(:), intent(in) :: x
        real(sp), dimension(:), intent(out) :: y

        integer :: i, j

        do i = 1, size(x)

            nn%layers(1)%x(i) = x(i)

        end do

        do i = 1, nn%n_layers

            call multiply_matrix_2d_1d(nn%layers(i)%weight, nn%layers(i)%x, nn%layers(i)%y)

            do j = 1, size(nn%layers(i)%bias)
                nn%layers(i)%y(j) = nn%layers(i)%y(j) + nn%layers(i)%bias(j)

                if (i .lt. nn%n_layers) then
                    !% TODO: add acivation function hidden nn%layers
                    nn%layers(i + 1)%x(j) = nn%layers(i)%y(j)
                else
                    !% TODO: add acivation function last layer
                    y(j) = nn%layers(i)%y(j)
                end if

            end do

        end do

    end subroutine feedforward_nn

end module md_neural_ode_operator
