!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - kge_components
!%
!%      Function
!%      --------
!%
!%      - nse
!%      - nnse
!%      - kge
!%      - mae
!%      - mape
!%      - se
!%      - mse
!%      - rmse
!%      - lgrm

module mwd_metrics

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
        !% NSE = 1 - num / den

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

            if (x(i) .lt. 0._sp) cycle

            n = n + 1
            sum_x = sum_x + x(i)
            sum_xx = sum_xx + (x(i)*x(i))
            sum_yy = sum_yy + (y(i)*y(i))
            sum_xy = sum_xy + (x(i)*y(i))

        end do

        mean_x = sum_x/n

        !% NSE numerator / denominator
        num = sum_xx - 2*sum_xy + sum_yy
        den = sum_xx - n*mean_x*mean_x

        !% NSE criterion
        res = 1._sp - num/den

    end function nse

    function nnse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Normalized NSE (NNSE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of NSE computation
        !% NSE = 1 / (2 - NSE)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        res = 1._sp/(2._sp - nse(x, y))

    end function nnse

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

            if (x(i) .lt. 0._sp) cycle

            n = n + 1
            sum_x = sum_x + x(i)
            sum_y = sum_y + y(i)
            sum_xx = sum_xx + (x(i)*x(i))
            sum_yy = sum_yy + (y(i)*y(i))
            sum_xy = sum_xy + (x(i)*y(i))

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
        !% KGE = 1 - sqrt((r - 1) ** 2 + (b - 1) ** 2 + (a - 1) ** 2)
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
        res = 1._sp - sqrt(&
        & (r - 1._sp)*(r - 1._sp) + &
        & (b - 1._sp)*(b - 1._sp) + &
        & (a - 1._sp)*(a - 1._sp) &
        & )

    end function kge

    function mae(x, y) result(res)
        !% Notes
        !% -----
        !%
        !% Mean Absolute Error (MAE) computation function
        !%
        !% Given two single precision arrays (x, y) of size n,
        !% it returns the result of MAE computation
        !% MAE = sum(abs(x - y)) / n

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i, n

        n = 0
        res = 0._sp

        do i = 1, size(x)
            if (x(i) .lt. 0._sp) cycle

            n = n + 1
            res = res + abs(x(i) - y(i))
        end do

        res = res/n

    end function mae

    function mape(x, y) result(res)
        !% Notes
        !% -----
        !%
        !% Mean Absolute Percentage Error (MAPE) computation function
        !%
        !% Given two single precision arrays (x, y) of size n,
        !% it returns the result of MAPE computation
        !% MAPE = sum(abs((x - y) / x)) / n

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i, n

        n = 0
        res = 0._sp

        do i = 1, size(x)

            if (x(i) .lt. 0._sp) cycle

            n = n + 1
            res = res + abs((x(i) - y(i))/x(i))

        end do

        res = res/n

    end function mape

    function se(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Squared Error (SE) computation function
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

            if (x(i) .lt. 0._sp) cycle

            res = res + (x(i) - y(i))*(x(i) - y(i))

        end do

    end function se

    function mse(x, y) result(res)
        !% Notes
        !% -----
        !%
        !% Mean Squared Error (MSE) computation function
        !%
        !% Given two single precision arrays (x, y) of size n,
        !% it returns the result of MSE computation
        !% MSE = SE / n
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

            if (x(i) .lt. 0._sp) cycle

            n = n + 1

        end do

        res = se(x, y)/n

    end function mse

    function mse_2d(x, y, mask) result(res)
    
        implicit none
        real(sp), dimension(:, :, :), intent(in) :: x, y
        integer, dimension(size(x, 1), size(x, 2)), intent(in) :: mask
        real(sp) :: res, temp_res
        
        integer :: counter
        integer :: i, j, k
        
        res = 0._sp
        counter = 0

        do k=1, size(x, 3)

            do i=1, size(x, 1)

                do j=1, size(x, 2)
                    
                    if (y(i,j,k) .ge. 0._sp) then
        
                        res = res + (x(i,j,k) - y(i,j,k))*(x(i,j,k) - y(i,j,k))
                        counter = counter + 1

                    end if  
                                                
                end do
            end do


            ! read(*,*)

        end do

        res = sqrt(res/counter)
    
    end function mse_2d

    subroutine spatial_efficiency(x, y, sig, alpha, active_cell)
    
        real(sp), dimension(:,:), intent(in) :: x, y
        real(sp) :: sum_x, sum_y, mean_x, mean_y, std_x, std_y, rms, sum_xx, sum_yy
        real(sp), intent(inout) :: sig, alpha
        integer :: i,j,counter
        integer, dimension(:, :), intent(in) :: active_cell

        sum_x = 0._sp
        sum_y = 0._sp
        counter = 0

        do i=1, size(x, 1)
            do j=1, size(x,2)

                if (active_cell(i,j)==1) then

                    sum_x = sum_x + x(i,j)
                    sum_y = sum_y + y(i,j)
                    counter = counter + 1

                end if
            end do
        end do

        mean_x = sum_x / counter
        mean_y = sum_y / counter

        ! print *, '-------------------------------------------------'
        ! print *, 'mean x'
        ! print*, mean_x

        ! print *, 'mean y'
        ! print*, mean_y

        ! compute the standard deviations

        sum_xx = 0._sp
        sum_yy = 0._sp
        counter = 0

        do i=1, size(x, 1)
            do j=1, size(x,2)

                if (active_cell(i,j)==1) then

                    sum_xx = sum_xx + (x(i,j) - mean_x)*(x(i,j) - mean_x)
                    sum_yy = sum_yy + (y(i,j) - mean_y)*(y(i,j) - mean_y)
                    counter = counter + 1

                end if
            end do
        end do

        std_x = sqrt(sum_xx / counter)
        std_y = sqrt(sum_yy / counter)

        ! print *, 'std_x'
        ! print*, std_x

        ! print *, 'std_y'
        ! print*, std_y

        sig = (std_x/mean_x)/(std_y/mean_y)
        ! print *, 'sig'
        ! print *, sig

        ! print *, '-------------------------------------------------'

        rms = 0._sp
        counter = 0
        do i = 1, size(x,1)
            do j = 1, size(x,2)

                if (active_cell(i,j)==1) then

                    rms = rms + &
                    & ((x(i,j) - mean_x)/std_x - (y(i,j) - mean_y)/std_y)*((x(i,j) - mean_x)/std_x - (y(i,j) - mean_y)/std_y)
                    counter = counter + 1
                end if

            end do
        end do

        alpha = 1 - sqrt(rms/counter)
        
    end subroutine spatial_efficiency

    ! function compute_trace(x,y) result(res)

    !     implicit none
    !     real(sp), dimension(:,:), intent(in) :: x, y
    !     real(sp), dimension(size(y,2),size(y,1)) :: y_transpose
    !     real(sp), dimension(size(x,1), size(x,1)) :: z
    !     real(sp) :: res
    !     integer :: i, j, k

    !     ! Compute the transpose of Y
    !     do i = 1, size(y,1)
    !         do j = 1, size(y,2)
    !             y_transpose(j, i) = y(i, j)
    !         end do
    !     end do

    !     ! Multiply X by the transpose of Y
    !     z = 0.0
    !     do i = 1, size(x, 1)
    !         do j = 1, size(x, 1)
    !             do k = 1, size(x, 2)
    !                 z(i, j) = z(i, j) + x(i, k) * y_transpose(k, j)
    !             end do
    !         end do
    !     end do

    !     ! Compute the trace of Z
    !     res = 0
    !     do i=1, size(z,1)
    !         if (z(i,i) .ge. 0._sp) then
    !             res = res + z(i,i)
    !         end if
    !     end do

    ! end function compute_trace

    function check_matrix_exists(x) result(res)
        implicit none
        real(sp), dimension(:,:), intent(in) :: x
        integer :: res
        integer :: i,j

        res = 0

        do i=1, size(x,1)
            do j=1, size(x,2)
        
                if(x(i,j) .ge. 0) then
                    res=1
                    exit
                end if

            end do
            
            if(res == 1) then 
                exit   
            end if

        end do
    end function check_matrix_exists

    subroutine flatten_and_mask(input, output, active_cell)
        real(sp), dimension(:, :), intent(in) :: input
        real(sp), dimension(:), intent(inout) :: output
        integer, dimension(:, :), intent(in) :: active_cell


        integer :: i, j, count

        count = 0

        do i = 1, size(input, 1)
            do j = 1, size(input, 2)
                if (active_cell(i,j)==1) then
                    count = count + 1
                    output(count) = input(i, j)
                end if
            end do
        end do
        
    end subroutine flatten_and_mask


    subroutine spearman_rank_correlation(x, y, spearman_coeff)
        real(sp), dimension(:), intent(in) :: x
        real(sp), dimension(:), intent(in) :: y
        real(sp), intent(out) :: spearman_coeff
        real(sp), dimension(size(x)) :: rank_x, rank_y
        real(sp) :: d
        integer :: i, n

        n = size(x)

        ! Calculate ranks for x and y
        call rankdata(x, rank_x)
        call rankdata(y, rank_y)

        ! Calculate the differences between ranks
        d = 0._sp
        do i=1, n
            d = d + (rank_x(i) - rank_y(i))*(rank_x(i) - rank_y(i))
        end do
        
        ! Calculate Spearman correlation coefficient
        spearman_coeff = 1.0 - ((6.0 * d)/(n * (n*n - 1.0)))
    end subroutine spearman_rank_correlation

    subroutine rankdata(data, ranked)
        real(sp), intent(in) :: data(:)
        real(sp), intent(inout) :: ranked(:)
        integer :: i, j, count, n
        real(sp) :: temp

        n = size(data)
        ranked = 0.0

        do i = 1, n
            count = 0
            temp = data(i)

            do j = 1, n
                if (data(j) <= temp) then
                    count = count + 1
                end if
            end do

            ranked(i) = count
        end do
    end subroutine rankdata



    subroutine spatial_bias_insensitive(x, y, res, active_cell, nac)
    
        real(sp), dimension(:, :, :), intent(in) :: x, y
        integer, dimension(:, :), intent(in) :: active_cell
        real(sp), intent(inout) :: res
        real(sp) :: sig, alpha, rs, trace, trace_x, trace_y
        integer :: k, n, counter, flag_exists, nac
        real(sp), dimension(nac) :: flattened_x, flattened_y
        
        res = 0._sp
        n = size(x, 3)
        counter = 0

        do k=1, n

            flag_exists = check_matrix_exists(y(:,:,k))
        
            if(flag_exists==1) then

                call flatten_and_mask(x(:,:,k), flattened_x, active_cell)
                call flatten_and_mask(y(:,:,k), flattened_y, active_cell)
                rs = 0._sp
                call spearman_rank_correlation(flattened_x, flattened_y, rs)

                sig = 0._sp
                alpha = 0._sp
                ! trace = compute_trace(x(:,:,k), y(:,:,k))
                ! trace_x = compute_trace(x(:,:,k), x(:,:,k))
                ! trace_y = compute_trace(y(:,:,k), y(:,:,k))
                ! rs = trace/(trace_x * trace_y)
                call spatial_efficiency(x(:,:,k), y(:,:,k), sig, alpha, active_cell)

                ! print *,'----------------------------------------------------------'
                ! print *, alpha
                ! print *, sig
                ! print *, rs
                ! print *, (1 - sqrt((rs-1)*(rs-1) + (sig-1)*(sig-1) + (alpha-1)*(alpha-1)))
                res = res + (1 - sqrt((rs-1)*(rs-1) + (sig-1)*(sig-1) + (alpha-1)*(alpha-1)))
                ! res = res + (1 - sqrt((sig-1)*(sig-1) + (alpha-1)*(alpha-1)))

                counter = counter + 1
                ! print *,'----------------------------------------------------------'

            end if
        end do
        ! print *,'##################################################################'
        ! print *, res
        ! print *, counter
        ! print *,'##################################################################'

        res =  1 - res/counter  
        ! res = res/counter 

    end subroutine spatial_bias_insensitive


    function rmse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Root Mean Squared Error (RMSE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SE computation
        !% RMSE = sqrt(MSE)
        !%
        !% See Also
        !% --------
        !% mse

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        res = sqrt(mse(x, y))

    end function rmse

    function lgrm(x, y) result(res)

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

            if (x(i) .le. 0._sp .or. y(i) .le. 0._sp) cycle

            res = res + x(i)*log(y(i)/x(i))*log(y(i)/x(i))

        end do

    end function lgrm

end module mwd_metrics
