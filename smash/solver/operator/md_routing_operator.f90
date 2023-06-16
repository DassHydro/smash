!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - upstream_discharge
!%      - linear_routing

module md_routing_operator

    use md_constant !% only : sp

    implicit none

contains

    subroutine upstream_discharge(nrow, ncol, dt, dx, dy, row, col, &
    & flwdir, flwacc, q, qup)
    
        implicit none
        
        integer, intent(in) :: nrow, ncol
        real(sp), intent(in) :: dt, dx, dy
        integer, intent(in) :: row, col
        integer, dimension(nrow, ncol), intent(in) :: flwdir
        real(sp), dimension(nrow, ncol), intent(in) :: flwacc
        real(sp), dimension(nrow, ncol), intent(in) :: q
        real(sp), intent(out) :: qup
        
        integer :: i, row_imd, col_imd
        integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
        integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
        
        qup = 0._sp
        
        if (flwacc(row, col) .le. dx * dy) return
    
        do i=1, 8
            
            col_imd = col + dcol(i)
            row_imd = row + drow(i)
            
            if (row_imd .lt. 1 .or. row_imd .gt. nrow .or. col_imd .lt. 1 .or. col_imd .gt. ncol) cycle
            
            if (flwdir(row_imd, col_imd) .eq. i) qup = qup + q(row_imd, col_imd)

        end do
            
        qup = (qup * dt) / (1e-3_sp * (flwacc(row, col) - dx * dy))

    end subroutine upstream_discharge

    subroutine linear_routing(dt, qup, lr, hlr, qrout)

        implicit none

        real(sp), intent(in) :: dt
        real(sp), intent(in) :: qup, lr
        real(sp), intent(inout) :: hlr
        real(sp), intent(out) :: qrout

        real(sp) :: hlr_imd

        hlr_imd = hlr + qup

        hlr = hlr_imd*exp(-dt/(lr*60._sp))

        qrout = hlr_imd - hlr

    end subroutine linear_routing

end module md_routing_operator
