!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - gr_ode_explicit_euler
!%      - solve_linear_system_2vars
!%      - gr_ode_implicit_euler

module md_node_operator

    use md_constant !% only : sp

    implicit none

contains

    !% TODO comment
    subroutine gr_ode_explicit_euler(pn, en, cp, ct, kexc, hp, ht, qti)

        implicit none

        real(sp), intent(in) :: pn, en, cp, ct, kexc
        real(sp), intent(inout) :: hp, ht, qti

        real(sp) :: hp_dot, ht_dot, dt
        integer :: i
        integer :: n_subtimesteps = 20

        dt = 1._sp/real(n_subtimesteps, sp)

        do i = 1, n_subtimesteps

            hp_dot = ((1._sp - hp**2)*pn - hp*(2._sp - hp)*en)/cp
            ht_dot = (0.9_sp*pn*hp**2 - ct*ht**5 + kexc*ht**3.5_sp)/ct

            hp = hp + dt*hp_dot
            ht = ht + dt*ht_dot

        end do

        qti = ct*ht**5 + 0.1_sp*pn*hp**2 + kexc*ht**3.5_sp

    end subroutine gr_ode_explicit_euler

    subroutine solve_linear_system_2vars(a, x, b)
        !% Solve linear system ax+b=0 with 2 variables

        implicit none

        real(sp), dimension(2, 2), intent(in) :: a
        real(sp), dimension(2), intent(in) :: b
        real(sp), dimension(2), intent(out) :: x

        real(sp) :: det_a

        det_a = a(1, 1)*a(2, 2) - a(1, 2)*a(2, 1)

        if (abs(det_a) .gt. 0._sp) then

            x(1) = (b(2)*a(1, 2) - b(1)*a(2, 2))/det_a
            x(2) = (b(1)*a(2, 1) - b(2)*a(1, 1))/det_a

        else
            x = 0._sp

        end if

    end subroutine solve_linear_system_2vars

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

end module md_node_operator
