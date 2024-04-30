!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - get_flwdst_cls
!%      - get_width_function_cdf
!%      - get_rainfall_weighted_width_function_cdf
!%      - precipitation_indices_computation

module mw_prcp_indices

    use md_constant, only: sp
    use md_stats, only: quantile1d_r
    use mwd_setup, only: SetupDT
    use mwd_mesh, only: MeshDT
    use mwd_input_data, only: Input_DataDT
    use mw_mask, only: mask_upstream_cells
    use mwd_sparse_matrix_manipulation, only: sparse_matrix_to_matrix

    implicit none

contains

    subroutine get_flwdst_cls(flwdst, flwdst_cls)

        implicit none

        real(sp), dimension(:, :), intent(in) :: flwdst
        real(sp), dimension(:), intent(inout) :: flwdst_cls

        integer :: i, n
        real(sp) :: step

        n = size(flwdst_cls)
        step = maxval(flwdst)/(n - 1)

        flwdst_cls(1) = 0._sp

        do i = 2, n

            flwdst_cls(i) = flwdst_cls(i - 1) + step

        end do

    end subroutine get_flwdst_cls

    subroutine get_width_function_cdf(flwdst, flwdst_cls, w_cdf)

        implicit none

        real(sp), dimension(:, :), intent(in) :: flwdst
        real(sp), dimension(:), intent(in) :: flwdst_cls
        real(sp), dimension(:), intent(inout) :: w_cdf

        integer :: i

        w_cdf(1) = 1._sp

        do i = 2, size(flwdst_cls)

            w_cdf(i) = w_cdf(i - 1) + count(flwdst .gt. flwdst_cls(i - 1) .and. flwdst .le. flwdst_cls(i))

        end do

        w_cdf = w_cdf/w_cdf(size(w_cdf))

    end subroutine get_width_function_cdf

    subroutine get_rainfall_weighted_width_function_cdf(flwdst, flwdst_cls, prcp_matrix, wp_cdf)

        implicit none

        real(sp), dimension(:, :), intent(in) :: flwdst, prcp_matrix
        real(sp), dimension(:), intent(in) :: flwdst_cls
        real(sp), dimension(:), intent(inout) :: wp_cdf

        integer :: i
        logical, dimension(size(flwdst, 1), size(flwdst, 2)) :: mask

        mask = (prcp_matrix .ge. 0._sp .and. flwdst .ge. 0._sp .and. flwdst .le. flwdst_cls(1))

        wp_cdf(1) = sum(prcp_matrix, mask=mask)

        do i = 2, size(flwdst_cls)

            mask = (prcp_matrix .ge. 0._sp .and. flwdst .gt. flwdst_cls(i - 1) .and. flwdst .le. flwdst_cls(i))
            wp_cdf(i) = wp_cdf(i - 1) + sum(prcp_matrix, mask=mask)

        end do

        wp_cdf = wp_cdf/wp_cdf(size(wp_cdf))

    end subroutine get_rainfall_weighted_width_function_cdf

    ! TODO FC: Parallelize the time loop
    subroutine precipitation_indices_computation(setup, mesh, input_data, prcp_indices)

        implicit none

        integer, parameter :: nprcp_indices = 5
        integer, parameter :: ncls = 10

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        real(sp), dimension(nprcp_indices, mesh%ng, setup%ntime_step), intent(inout) :: prcp_indices

        integer :: i, j, row, col, ipvg, ivg
        real(sp) :: minv_n, sum_p, sum_p2, sum_d, sum_d2, sum_pd, sum_pd2, &
        & p0, p1, p2, g1, g2, fpvg, dpvg, dvg, dmax, std, d1, d2, vg, hg
        logical, dimension(mesh%nrow, mesh%ncol) :: mask
        real(sp), dimension(mesh%nrow, mesh%ncol) :: prcp_matrix
        logical, dimension(mesh%nrow, mesh%ncol, mesh%ng) :: mask_gauge
        real(sp), dimension(mesh%nrow, mesh%ncol, mesh%ng) :: flwdst_gauge
        real(sp), dimension(ncls + 1, mesh%ng) :: flwdst_cls_gauge, w_cdf_gauge
        real(sp), dimension(ncls + 1) :: wp_cdf, absd_cdf

        prcp_indices = -99._sp

        ! Get upstream cells mask and flow distances for each gauge
        mask_gauge = .false.

        do i = 1, mesh%ng

            row = mesh%gauge_pos(i, 1)
            col = mesh%gauge_pos(i, 2)

            call mask_upstream_cells(mesh, row, col, mask_gauge(:, :, i))
            flwdst_gauge(:, :, i) = mesh%flwdst - mesh%flwdst(row, col)
            call get_flwdst_cls(flwdst_gauge(:, :, i), flwdst_cls_gauge(:, i))
            call get_width_function_cdf(flwdst_gauge(:, :, i), flwdst_cls_gauge(:, i), w_cdf_gauge(:, i))

        end do

        ! Get precpitation indices for each time step
        do i = 1, setup%ntime_step

            ! Get precipitation matrix
            if (setup%sparse_storage) then

                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(i), prcp_matrix)

            else

                prcp_matrix = input_data%atmos_data%prcp(:, :, i)

            end if

            ! Get precipitation indices for each gauge
            do j = 1, mesh%ng

                ! Get intersect mask between mask_gauge and prcp_matrix .ge. 0
                mask = (prcp_matrix .ge. 0._sp .and. mask_gauge(:, :, j))

                minv_n = 1._sp/count(mask)

                sum_p = sum(prcp_matrix, mask=mask)

                ! Cycle if there is no precipitation
                if (sum_p .le. 0._sp) cycle

                ! Get intermediate values for std, d1 and d2
                sum_p2 = sum(prcp_matrix*prcp_matrix, mask=mask)
                sum_d = sum(flwdst_gauge(:, :, j), mask=mask)
                sum_d2 = sum(flwdst_gauge(:, :, j)*flwdst_gauge(:, :, j), mask=mask)
                sum_pd = sum(prcp_matrix*flwdst_gauge(:, :, j), mask=mask)
                sum_pd2 = sum(prcp_matrix*flwdst_gauge(:, :, j)*flwdst_gauge(:, :, j), mask=mask)
                p0 = minv_n*sum_p
                p1 = minv_n*sum_pd
                p2 = minv_n*sum_pd2
                g1 = minv_n*sum_d
                g2 = minv_n*sum_d2

                ! Get intermediate values for vg and hg
                call get_rainfall_weighted_width_function_cdf(flwdst_gauge(:, :, j), flwdst_cls_gauge(:, j), prcp_matrix, wp_cdf)
                absd_cdf = abs(w_cdf_gauge(:, j) - wp_cdf)
                ipvg = maxloc(absd_cdf, dim=1)
                fpvg = wp_cdf(ipvg)
                ivg = minloc(abs(w_cdf_gauge(:, j) - fpvg), dim=1)
                dpvg = flwdst_cls_gauge(ipvg, j)
                dvg = flwdst_cls_gauge(ivg, j)
                dmax = flwdst_cls_gauge(ncls + 1, j)

                ! Get precipitation indices
                std = sqrt((minv_n*sum_p2) - (p0*p0))
                d1 = p1/(p0*g1)
                d2 = 1._sp/(g2 - g1*g1)*((p2/p0) - (p1/p0)*(p1/p0))
                vg = absd_cdf(ipvg)
                hg = abs(dvg - dpvg)/dmax

                prcp_indices(:, j, i) = [std, d1, d2, vg, hg]

            end do

        end do

    end subroutine precipitation_indices_computation

end module mw_prcp_indices
