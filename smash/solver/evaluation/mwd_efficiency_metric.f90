!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - nse
!%      - kge_components
!%      - kge
!%      - se
!%      - rmse
!%      - logarithmic

module mwd_efficiency_metric

    use md_constant !% only: sp

    implicit none

contains

    function nse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% NSE computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of NSE computation
        !% num = sum(x**2) - 2 * sum(x*y) + sum(y**2)
        !% den = sum(x**2) - n * mean(x) ** 2
        !% NSE = num / den

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        real(sp) :: sum_x, sum_xx, sum_yy, sum_xy, mean_x, num, den
        integer :: i, n

        !% Metric computation
        n = 0
        sum_x = 0._sp
        sum_xx = 0._sp
        sum_yy = 0._sp
        sum_xy = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1
                sum_x = sum_x + x(i)
                sum_xx = sum_xx + (x(i)*x(i))
                sum_yy = sum_yy + (y(i)*y(i))
                sum_xy = sum_xy + (x(i)*y(i))

            end if

        end do

        mean_x = sum_x/n

        !% NSE numerator / denominator
        num = sum_xx - 2*sum_xy + sum_yy
        den = sum_xx - n*mean_x*mean_x

        !% NSE criterion
        res = num/den

    end function nse

    subroutine kge_components(x, y, r, a, b)

        !% Notes
        !% -----
        !%
        !% KGE components computation subroutine
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns KGE components r, a, b
        !% r = cov(x,y) / std(y) / std(x)
        !% a = mean(y) / mean(x)
        !% b = std(y) / std(x)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp), intent(inout) :: r, a, b

        real(sp) :: sum_x, sum_y, sum_xx, sum_yy, sum_xy, mean_x, mean_y, &
        & var_x, var_y, cov
        integer :: n, i

        ! Metric computation
        n = 0
        sum_x = 0._sp
        sum_y = 0._sp
        sum_xx = 0._sp
        sum_yy = 0._sp
        sum_xy = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1
                sum_x = sum_x + x(i)
                sum_y = sum_y + y(i)
                sum_xx = sum_xx + (x(i)*x(i))
                sum_yy = sum_yy + (y(i)*y(i))
                sum_xy = sum_xy + (x(i)*y(i))

            end if

        end do

        mean_x = sum_x/n
        mean_y = sum_y/n
        var_x = (sum_xx/n) - (mean_x*mean_x)
        var_y = (sum_yy/n) - (mean_y*mean_y)
        cov = (sum_xy/n) - (mean_x*mean_y)

        ! KGE components (r, alpha, beta)
        r = (cov/sqrt(var_x))/sqrt(var_y)
        a = sqrt(var_y)/sqrt(var_x)
        b = mean_y/mean_x

    end subroutine kge_components

    function kge(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% KGE computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of KGE computation
        !% KGE = sqrt((1 - r) ** 2 + (1 - a) ** 2 + (1 - b) ** 2)
        !%
        !% See Also
        !% --------
        !% kge_components

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        real(sp) :: r, a, b

        call kge_components(x, y, r, a, b)

        ! KGE criterion
        res = sqrt(&
        & (r - 1)*(r - 1) + (b - 1)*(b - 1) + (a - 1)*(a - 1) &
        & )

    end function kge

    function se(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Square Error (SE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SE computation
        !% SE = sum((x - y) ** 2)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i

        res = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                res = res + (x(i) - y(i))*(x(i) - y(i))

            end if

        end do

    end function se

    function rmse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Root Mean Square Error (RMSE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SE computation
        !% RMSE = sqrt(SE / n)
        !%
        !% See Also
        !% --------
        !% se

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i, n

        n = 0

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1

            end if

        end do

        res = sqrt(se(x, y)/n)

    end function rmse

    function logarithmic(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Logarithmic (LGRM) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of LGRM computation
        !% LGRM = sum(x * log(y/x) ** 2)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i

        res = 0._sp

        do i = 1, size(x)

            if (x(i) .gt. 0._sp .and. y(i) .gt. 0._sp) then

                res = res + x(i)*log(y(i)/x(i))*log(y(i)/x(i))

            end if

        end do

    end function logarithmic

end module mwd_efficiency_metric
