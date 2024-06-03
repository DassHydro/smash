!%      (MD) Module Differentiated.
!%
!%      Interface
!%      ---------
!%
!%      - quantile1d_r
!%          . quantile1d_r_scl
!%          . quantile1d_r_1d
!%
!%      Subroutine
!%      ----------
!%
!%      - heap_sort
!%
!%      Function
!%      --------
!%
!%      - quantile1d_r_scl
!%      - quantile1d_r_1d

module md_stats

    use md_constant !% only: sp
    use mwd_mesh !% only: MeshDT
    use mwd_output !% only: OutputDT
    use mwd_returns !% only: ReturnsDT
    
    implicit none

    interface quantile1d_r

        module procedure quantile1d_r_scl
        module procedure quantile1d_r_1d

    end interface quantile1d_r

contains

    subroutine heap_sort(n, arr)

        !% Notes
        !% -----
        !%
        !% Implement heap sort algorithm
        !%
        !% Computational complexity is O(n log n)

        implicit none

        integer, intent(in) :: n
        real(sp), dimension(n), intent(inout) :: arr

        integer :: l, ir, i, j
        real(sp) :: arr_l

        l = n/2 + 1

        ir = n

10      continue

        if (l .gt. 1) then

            l = l - 1

            arr_l = arr(l)

        else

            arr_l = arr(ir)

            arr(ir) = arr(1)

            ir = ir - 1

            if (ir .eq. 1) then

                arr(1) = arr_l

                return

            end if

        end if

        i = l

        j = l + l

20      if (j .le. ir) then

            if (j .lt. ir) then

                if (arr(j) .lt. arr(j + 1)) j = j + 1

            end if

            if (arr_l .lt. arr(j)) then

                arr(i) = arr(j)

                i = j; j = j + j

            else

                j = ir + 1

            end if

            goto 20

        end if

        arr(i) = arr_l

        goto 10

    end subroutine heap_sort

    recursive subroutine quicksort(arr)

        implicit none

        real(sp), intent(inout) :: arr(:)

        real pivot, temp
        integer :: first = 1, last
        integer i, j

        last = size(arr, 1)
        pivot = arr((first + last)/2)
        i = first
        j = last

        do
            do while (arr(i) .lt. pivot)
                i = i + 1
            end do

            do while (pivot .lt. arr(j))
                j = j - 1
            end do

            if (i .ge. j) exit

            temp = arr(i); arr(i) = arr(j); arr(j) = temp

            i = i + 1
            j = j - 1
        end do

        if (first < i - 1) call quicksort(arr(first:i - 1))
        if (j + 1 < last) call quicksort(arr(j + 1:last))

    end subroutine quicksort

    subroutine compute_fluxes_stats(mesh, t, idx, returns)
        implicit none

        type(MeshDT), intent(in) :: mesh
        integer, intent(in) :: t, idx
        type(ReturnsDT), intent(inout) :: returns
        real(sp), dimension(mesh%nrow, mesh%ncol) :: fx
        real(sp), dimension(:), allocatable :: fx_flat
        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        integer :: j, npos_val
        real(sp) :: m

        fx = returns%stats%internal_fluxes(:, :, idx)
        !$AD start-exclude
        do j = 1, mesh%ng

            if (returns%stats%fluxes_keys(idx) .eq. 'kexc') then
                mask = mesh%mask_gauge(:, :, j)
            else
                mask = (fx .ge. 0._sp .and. mesh%mask_gauge(:, :, j))
            end if

            npos_val = count(mask)
            m = sum(fx, mask=mask)/npos_val
            returns%stats%fluxes_values(j, t, 1, idx) = m
            returns%stats%fluxes_values(j, t, 2, idx) = sum((fx - m)*(fx - m), mask=mask)/npos_val
            returns%stats%fluxes_values(j, t, 3, idx) = minval(fx, mask=mask)
            returns%stats%fluxes_values(j, t, 4, idx) = maxval(fx, mask=mask)

            if (.not. allocated(fx_flat)) allocate (fx_flat(npos_val))
            fx_flat = pack(fx, mask .eqv. .True.)

            call quicksort(fx_flat)

            if (mod(npos_val, 2) .ne. 0) then
                returns%stats%fluxes_values(j, t, 5, idx) = fx_flat(npos_val/2 + 1)
            else
                returns%stats%fluxes_values(j, t, 5, idx) = (fx_flat(npos_val/2) + fx_flat(npos_val/2 + 1))/2
            end if

        end do
        !$AD end-exclude
    end subroutine

    subroutine compute_states_stats(mesh, output, t, idx, returns)
        implicit none

        type(MeshDT), intent(in) :: mesh
        type(OutputDT), intent(in) :: output
        integer, intent(in) :: t, idx
        type(ReturnsDT), intent(inout) :: returns
        real(sp), dimension(mesh%nrow, mesh%ncol) :: h
        real(sp), dimension(:), allocatable :: h_flat
        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        integer :: j, npos_val
        real(sp) :: m

        h = output%rr_final_states%values(:, :, idx)
        !$AD start-exclude
        do j = 1, mesh%ng

            mask = (h .ge. 0._sp .and. mesh%mask_gauge(:, :, j))

            npos_val = count(mask)
            m = sum(h, mask=mask)/npos_val
            returns%stats%rr_states_values(j, t, 1, idx) = m
            returns%stats%rr_states_values(j, t, 2, idx) = sum((h - m)*(h - m), mask=mask)/npos_val
            returns%stats%rr_states_values(j, t, 3, idx) = minval(h, mask=mask)
            returns%stats%rr_states_values(j, t, 4, idx) = maxval(h, mask=mask)

            if (.not. allocated(h_flat)) allocate (h_flat(npos_val))
            h_flat = pack(h, mask .eqv. .True.)

            call quicksort(h_flat)

            if (mod(npos_val, 2) .ne. 0) then
                returns%stats%rr_states_values(j, t, 5, idx) = h_flat(npos_val/2 + 1)
            else
                returns%stats%rr_states_values(j, t, 5, idx) = (h_flat(npos_val/2) + h_flat(npos_val/2 + 1))/2
            end if

        end do
        !$AD end-exclude
    end subroutine

    function quantile1d_r_scl(dat, p) result(res)

        !% Notes
        !% -----
        !%
        !% Quantile function for real 1d array and real scalar quantile value using linear interpolation
        !%
        !% Similar to numpy.quantile

        implicit none

        real(sp), dimension(:), intent(in) :: dat
        real(sp), intent(in) :: p
        real(sp) :: res

        real(sp), dimension(size(dat)) :: sorted_dat
        integer :: n
        real(sp) :: q1, q2, frac

        res = dat(1)

        n = size(dat)

        if (n .gt. 1) then

            sorted_dat = dat

            call heap_sort(n, sorted_dat)

            frac = (n - 1)*p + 1

            if (frac .le. 1) then

                res = sorted_dat(1)

            else if (frac .ge. n) then

                res = sorted_dat(n)

            else
                q1 = sorted_dat(int(frac))

                q2 = sorted_dat(int(frac) + 1)

                res = q1 + (q2 - q1)*(frac - int(frac)) ! linear interpolation

            end if

        end if

    end function quantile1d_r_scl

    function quantile1d_r_1d(dat, p) result(res)

        !% Notes
        !% -----
        !%
        !% Quantile function for real 1d array and real 1d array quantile using linear interpolation
        !%
        !% Similar to numpy.quantile

        implicit none

        real(sp), dimension(:), intent(in) :: dat
        real(sp), dimension(:), intent(in) :: p
        real(sp), dimension(size(p)) :: res

        real(sp), dimension(size(dat)) :: sorted_dat
        integer :: n, i
        real(sp) :: q1, q2, frac

        res = dat(1)

        n = size(dat)

        if (n .gt. 1) then

            sorted_dat = dat

            call heap_sort(n, sorted_dat)

            do i = 1, size(p)

                frac = (n - 1)*p(i) + 1

                if (frac .le. 1) then

                    res(i) = sorted_dat(1)

                else if (frac .ge. n) then

                    res(i) = sorted_dat(n)

                else
                    q1 = sorted_dat(int(frac))

                    q2 = sorted_dat(int(frac) + 1)

                    res(i) = q1 + (q2 - q1)*(frac - int(frac)) ! linear interpolation

                end if

            end do

        end if

    end function quantile1d_r_1d

end module md_stats
