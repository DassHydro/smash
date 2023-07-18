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
!%      - signature_computation

module mwd_signatures

    use md_constant !% only: sp
    use md_stats !% any type

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
            bt(1) = mean1d_r(streamflow)/1.5_sp

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

                if (streamflow(ends(j + 1)) .lt. mean1d_r(btp)) then
                    bt(ends(j + 1)) = streamflow(ends(j + 1))/1.2_sp

                else
                    bt(ends(j + 1)) = mean1d_r(btp)

                end if

            end if

        end do

    end subroutine baseflow_separation

    function signature_computation(p, q, stype) result(res)

        !% Notes
        !% -----
        !%
        !% Signature computation (S)
        !%
        !% Given two single precision array (p, q) of dim(1) and size(n),
        !% it returns the result of S computation
        !% S_i = f_i(p, q)
        !% where i is a signature i and f_i is its associated signature computation function

        implicit none

        real(sp), dimension(:), intent(in) :: p, q
        character(len=*), intent(in) :: stype

        real(sp), dimension(size(p)) :: nonnegative_p, nonnegative_q
        real(sp), dimension(size(p)) :: bf, qf

        real(sp) :: res

        integer :: i, j, n, imax_p, imax_q
        real(sp) :: quant, numer, denom, max_p, max_q

        n = size(p)

        res = -99._sp

        select case (stype)

            ! runoff coefs: only based on p and q
        case ("Crc", "Erc")

            numer = 0._sp
            denom = 0._sp

            do i = 1, n

                if (p(i) .ge. 0._sp .and. q(i) .ge. 0._sp) then

                    numer = numer + q(i)
                    denom = denom + p(i)

                end if

            end do

            if (denom .gt. 0._sp) then

                res = numer/denom

            end if

            ! runoff coefs based on baseflow and/or quickflow
        case ("Crchf", "Crclf", "Crch2r", "Erchf", "Erclf", "Erch2r", "Eff", "Ebf")
            nonnegative_p = 0._sp
            nonnegative_q = 0._sp

            j = 0

            do i = 1, n
                if (p(i) .ge. 0._sp .and. q(i) .ge. 0._sp) then
                    j = j + 1

                    nonnegative_p(j) = p(i)
                    nonnegative_q(j) = q(i)

                end if

            end do

            if (j .gt. 1) then

                call baseflow_separation(nonnegative_q(1:j), bf(1:j), qf(1:j), 0.925, 3)

                numer = 0._sp
                denom = 0._sp

                do i = 1, j
                    select case (stype)

                    case ("Crchf", "Erchf")
                        numer = numer + qf(i)
                        denom = denom + nonnegative_p(i)

                    case ("Crclf", "Erclf")
                        numer = numer + bf(i)
                        denom = denom + nonnegative_p(i)

                    case ("Crch2r", "Erch2r")
                        numer = numer + qf(i)
                        denom = denom + nonnegative_q(i)

                    end select

                end do

                if (denom .gt. 0._sp) then

                    res = numer/denom

                end if

                select case (stype)

                case ("Eff")
                    res = mean1d_r(qf(1:j))

                case ("Ebf")
                    res = mean1d_r(bf(1:j))

                end select

            end if

            ! flow percentiles
        case ("Cfp2", "Cfp10", "Cfp50", "Cfp90")

            j = 0

            do i = 1, n
                if (q(i) .ge. 0._sp) then

                    j = j + 1
                    nonnegative_q(j) = q(i)

                end if

            end do

            if (j .gt. 1) then

                quant = 0._sp

                select case (stype)

                case ("Cfp2")
                    quant = 0.02_sp

                case ("Cfp10")
                    quant = 0.1_sp

                case ("Cfp50")
                    quant = 0.5_sp

                case ("Cfp90")
                    quant = 0.9_sp

                end select

                res = quantile1d_r(nonnegative_q(1:j), quant)

            end if

            ! peak flow
        case ("Epf")
            max_q = -99._sp

            do i = 1, n
                if (q(i) .gt. max_q) then

                    max_q = q(i)

                end if

            end do

            res = max_q

            ! lag time
        case ("Elt")
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

        end select

    end function signature_computation

end module mwd_signatures
