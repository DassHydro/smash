!%      This module `md_routing_operator` encapsulates all SMASH routing operator.
!%      This module is differentiated.
!%
!%      contains
!%
!%      [1] upstream_discharge
!%      [2] sparse_upstream_discharge
!%      [3] linear_routing

module md_routing_operator
    
    use md_constant !% only : sp

    implicit none
    
    contains
        
        subroutine upstream_discharge(dt, dx, nrow, ncol, &
        & flwdir, flwacc, row, col, q, qup)
        
            implicit none

            real(sp), intent(in) :: dt, dx
            integer, intent(in) :: nrow, ncol, row, col
            integer, dimension(nrow, ncol), intent(in) :: flwdir, flwacc
            real(sp), dimension(nrow, ncol), intent(in) :: q
            real(sp), intent(out) :: qup
            
            integer :: i, row_imd, col_imd
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            qup = 0._sp
            
            if (flwacc(row, col) .gt. 1) then
            
                do i=1, 8
                    
                    col_imd = col + dcol(i)
                    row_imd = row + drow(i)
                    
                    if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
                    &   row_imd .gt. 0 .and. row_imd .le. nrow) then
                    
                        if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
                        
                            qup = qup + q(row_imd, col_imd)
                            
                        end if
                        
                    end if
                
                end do
                
                qup = (qup * dt) / &
                & (0.001_sp * dx * dx * real(flwacc(row, col) - 1))
            
            end if
        
        end subroutine upstream_discharge
        
        
        subroutine sparse_upstream_discharge(dt, dx, nrow, ncol, nac, &
        & flwdir, flwacc, ind_sparse, row, col, q, qup)
            
            implicit none

            real(sp), intent(in) :: dt, dx
            integer, intent(in) :: nrow, ncol, nac, row, col
            integer, dimension(nrow, ncol), intent(in) :: flwdir, flwacc, ind_sparse
            real(sp), dimension(nac), intent(in) :: q
            real(sp), intent(out) :: qup
            
            integer :: i, row_imd, col_imd, k
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            qup = 0._sp
            
            if (flwacc(row, col) .gt. 1) then
            
                do i=1, 8
                    
                    col_imd = col + dcol(i)
                    row_imd = row + drow(i)
                    
                    if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
                    &   row_imd .gt. 0 .and. row_imd .le. nrow) then
                    
                        if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
                            
                            k = ind_sparse(row_imd, col_imd)
                            qup = qup + q(k)
                            
                        end if
                        
                    end if
                
                end do
                
                qup = (qup * dt) / &
                & (0.001_sp * dx * dx * real(flwacc(row, col) - 1))
            
            end if
        
        end subroutine sparse_upstream_discharge


        subroutine linear_routing(dt, qup, lr, hr, qrout)
            
            implicit none
            
            real(sp), intent(in) :: dt
            real(sp), intent(in) :: qup, lr
            real(sp), intent(inout) :: hr
            real(sp), intent(out) :: qrout
            
            real(sp) :: hr_imd
            
            hr_imd = hr + qup
            
            hr = hr_imd * exp(- dt / (lr * 60._sp))
            
            qrout = hr_imd - hr
        
        end subroutine linear_routing
        
        
end module md_routing_operator
