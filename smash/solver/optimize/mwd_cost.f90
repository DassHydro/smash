!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - compute_cost
!%      - nse

module mwd_cost

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT

    implicit none

contains

    subroutine compute_cost(setup, mesh, input_data, parameters, output, options, returns)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(inout) :: output
        type(OptionsDT), intent(in) :: options
        type(ReturnsDT), intent(inout) :: returns

        call nse(input_data%obs_response%q(1, :), output%sim_response%q(1, :), output%cost)

    end subroutine compute_cost

    subroutine nse(x, y, res)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp), intent(inout) :: res

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

    end subroutine nse

end module mwd_cost
