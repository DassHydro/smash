!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - baseflow_separation
!%
!%      Function
!%      --------
!%
!%      - rc
!%      - rchf
!%      - rclf
!%      - rch2r
!%      - cfp
!%      - eff
!%      - ebf
!%      - epf
!%      - elt

module mwd_signatures

    use md_constant !% only: sp
    use md_stats !% only: quantile1d_r

    implicit none

contains

    subroutine baseflow_separation(streamflow, bt, qft, filter_parameter, passes)

        !% Notes
        !% -----
        !%
        !% Baseflow separation routine
        !%
        !% Dafult parameters:
        !% filter_parameter = 0.925, passes = 3

        implicit none

        real(sp), dimension(:), intent(in) :: streamflow
        real(sp), dimension(size(streamflow)), intent(inout) :: bt, qft
        real(sp), intent(in) :: filter_parameter
        integer, intent(in) :: passes

        real(sp), dimension(size(streamflow)) :: btp
        integer, dimension(passes + 1) :: ends
        integer, dimension(passes) :: addtostart
        integer :: i, j
        logical :: odd

        odd = .true.

        ! Start and end values for the filter function
        do j = 1, passes
            if (odd) then
                ends(j) = 1
                addtostart(j) = 1

            else
                ends(j) = size(streamflow)
                addtostart(j) = -1
            end if

            odd = .not. odd

        end do

        ends(passes + 1) = ends(passes - 1)

        btp = streamflow

        bt = 0._sp
        qft = 0._sp

        ! Guess baseflow value in the first time step
        if (streamflow(1) .lt. quantile1d_r(streamflow, 0.25_sp)) then
            bt(1) = streamflow(1)

        else
            bt(1) = (sum(streamflow)/size(streamflow))/1.5_sp

        end if

        ! Perform baseflow separation
        do j = 1, passes
            do i = ends(j) + addtostart(j), ends(j + 1), addtostart(j)
                if ((filter_parameter*bt(i - addtostart(j)) + &
                     ((1._sp - filter_parameter)/2._sp)*(btp(i) + btp(i - addtostart(j)))) .gt. btp(i)) then
                    bt(i) = btp(i)

                else
                    bt(i) = filter_parameter*bt(i - addtostart(j)) + &
                            ((1._sp - filter_parameter)/2._sp)*(btp(i) + btp(i - addtostart(j)))

                end if

                qft(i) = streamflow(i) - bt(i)

            end do

            if (j .lt. passes) then
                btp = bt

                if (streamflow(ends(j + 1)) .lt. sum(btp)/size(btp)) then
                    bt(ends(j + 1)) = streamflow(ends(j + 1))/1.2_sp

                else
                    bt(ends(j + 1)) = sum(btp)/size(btp)

                end if

            end if

        end do

    end subroutine baseflow_separation

    function rc(p, q) result(res)

        !% Notes
        !% -----
        !%
        !% Runoff Cofficient (Crc or Erc)
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of RC computation

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        real(sp) :: res

        integer :: n, i
        real(sp) :: numer, denom

        n = size(p)

        res = -99._sp

        numer = 0._sp
        denom = 0._sp

        do i = 1, n

            if (p(i) .lt. 0._sp .or. q(i) .lt. 0._sp) cycle

            numer = numer + q(i)
            denom = denom + p(i)

        end do

        if (denom .gt. 0._sp) then

            res = numer/denom

        end if

    end function rc

    function rchf(p, q) result(res)

        !% Notes
        !% -----
        !%
        !% Runoff Cofficient on Highflow (Crchf or Erchf)
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of RCHF computation

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        real(sp) :: res

        integer :: n, i, j
        real(sp) :: numer, denom
        real(sp), dimension(size(p)) :: nonnegative_p, nonnegative_q, bf, qf

        n = size(p)

        res = -99._sp

        nonnegative_p = 0._sp
        nonnegative_q = 0._sp

        j = 0

        do i = 1, n

            if (p(i) .lt. 0._sp .or. q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_p(j) = p(i)
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925_sp, 3)

            numer = 0._sp
            denom = 0._sp

            do i = 1, j

                numer = numer + qf(i)
                denom = denom + nonnegative_p(i)

            end do

            if (denom .gt. 0._sp) then

                res = numer/denom

            end if

        end if

    end function rchf

    function rclf(p, q) result(res)

        !% Notes
        !% -----
        !%
        !% Runoff Cofficient on Lowflow (Crclf or Erclf)
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of RCLF computation

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        real(sp) :: res

        integer :: n, i, j
        real(sp) :: numer, denom
        real(sp), dimension(size(p)) :: nonnegative_p, nonnegative_q, bf, qf

        n = size(p)

        res = -99._sp

        nonnegative_p = 0._sp
        nonnegative_q = 0._sp

        j = 0

        do i = 1, n

            if (p(i) .lt. 0._sp .or. q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_p(j) = p(i)
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925_sp, 3)

            numer = 0._sp
            denom = 0._sp

            do i = 1, j

                numer = numer + bf(i)
                denom = denom + nonnegative_p(i)

            end do

            if (denom .gt. 0._sp) then

                res = numer/denom

            end if

        end if

    end function rclf

    function rch2r(p, q) result(res)

        !% Notes
        !% -----
        !%
        !% Runoff Cofficient on Highflow and discharge (Crch2r or Erch2r)
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of RCLF computation

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        real(sp) :: res

        integer :: n, i, j
        real(sp) :: numer, denom
        real(sp), dimension(size(p)) :: nonnegative_p, nonnegative_q, bf, qf

        n = size(p)

        res = -99._sp

        nonnegative_p = 0._sp
        nonnegative_q = 0._sp

        j = 0

        do i = 1, n

            if (p(i) .lt. 0._sp .or. q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_p(j) = p(i)
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925_sp, 3)

            numer = 0._sp
            denom = 0._sp

            do i = 1, j

                numer = numer + qf(i)
                denom = denom + nonnegative_q(i)

            end do

            if (denom .gt. 0._sp) then

                res = numer/denom

            end if

        end if

    end function rch2r

    function cfp(q, quant) result(res)

        !% Notes
        !% -----
        !%
        !% Flow percentiles (Cfp2, Cfp10, Cfp50, or Cfp90)
        !% Given one single precision array q of dim(1) and size(n),
        !% it returns the result of FP computation

        implicit none

        real(sp), dimension(:), intent(in) :: q
        real(sp), intent(in) :: quant
        real(sp) :: res

        integer :: n, i, j
        real(sp), dimension(size(q)) :: nonnegative_q

        n = size(q)

        res = -99._sp

        j = 0

        do i = 1, n

            if (q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            res = quantile1d_r(nonnegative_q(1:j), quant)

        end if

    end function cfp

    function eff(q) result(res)

        !% Notes
        !% -----
        !%
        !% Flood flow (Eff)
        !% Given one single precision array q of dim(1) and size(n),
        !% it returns the result of FF computation

        implicit none

        real(sp), dimension(:), intent(in) :: q
        real(sp) :: res

        integer :: n, i, j
        real(sp), dimension(size(q)) :: nonnegative_q, bf, qf

        n = size(q)

        res = -99._sp

        nonnegative_q = 0._sp

        j = 0

        do i = 1, n

            if (q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925_sp, 3)

            res = sum(qf(1:j))/j

        end if

    end function eff

    function ebf(q) result(res)

        !% Notes
        !% -----
        !%
        !% Base flow (Ebf)
        !% Given one single precision array q of dim(1) and size(n),
        !% it returns the result of BF computation

        implicit none

        real(sp), dimension(:), intent(in) :: q
        real(sp) :: res

        integer :: n, i, j
        real(sp), dimension(size(q)) :: nonnegative_q, bf, qf

        n = size(q)

        res = -99._sp

        nonnegative_q = 0._sp

        j = 0

        do i = 1, n

            if (q(i) .lt. 0._sp) cycle

            j = j + 1
            nonnegative_q(j) = q(i)

        end do

        if (j .gt. 1) then

            call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925_sp, 3)

            res = sum(bf(1:j))/j

        end if

    end function ebf

    function epf(q) result(res)

        !% Notes
        !% -----
        !%
        !% Peak flow (Epf)
        !% Given one single precision array q of dim(1) and size(n),
        !% it returns the result of PF computation

        implicit none

        real(sp), dimension(:), intent(in) :: q
        real(sp) :: res

        integer :: n, i

        n = size(q)

        res = -99._sp

        do i = 1, n

            if (q(i) .le. res) cycle

            res = q(i)

        end do

    end function epf

    function elt(p, q) result(res)

        !% Notes
        !% -----
        !%
        !% Lag time (Elt)
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of LT computation

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        real(sp) :: res

        integer :: n, i, imax_p, imax_q
        real(sp) :: max_p, max_q

        n = size(q)

        res = -99._sp

        max_p = -99._sp
        max_q = -99._sp

        imax_p = 0
        imax_q = 0

        do i = 1, n
            if (p(i) .gt. max_p) then

                max_p = p(i)
                imax_p = i

            end if

            if (q(i) .gt. max_q) then

                max_q = q(i)
                imax_q = i

            end if

        end do

        if (imax_p .gt. 0 .and. imax_q .gt. 0) then

            res = imax_q - imax_p

        end if

    end function elt

end module mwd_signatures
