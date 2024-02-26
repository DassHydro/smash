!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - solve_linear_system_2vars
!%      - dot_product_2d_1d

module md_algebra

    use md_constant !% only : sp

    implicit none

contains

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

    subroutine dot_product_2d_1d(a, x, b)
        !% Multiply 2D matrix (m, n) with 1D vector (n,) producing a 1D vector of (m,)
        !% This routine accepts all arguments x and b with size(x)>=n and size(b)>=m

        implicit none

        real(sp), dimension(:, :), intent(in) :: a
        real(sp), dimension(:), intent(in) :: x
        real(sp), dimension(:), intent(inout) :: b

        integer :: i, j

        b = 0._sp

        do j = 1, size(a, 2)
            do i = 1, size(a, 1)
                b(i) = b(i) + a(i, j)*x(j)
            end do
        end do

    end subroutine dot_product_2d_1d

end module md_algebra
