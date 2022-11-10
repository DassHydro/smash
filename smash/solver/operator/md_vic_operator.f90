!%      This module `md_vic_operator` encapsulates all SMASH VIC operators.
!%      This module is differentiated.
!%
!%      contains
!%
!%      [1] vic_infiltration
!%      [2] vic_vertical_transfer
!%      [3] vic_interflow
!%      [4] vic_baseflow
!%      [5] brooks_and_corey_flow
!%      [6] linear_evapotranspiration

module md_vic_operator
    
    use md_constant !% only : sp

    implicit none
    
    contains
        
        subroutine vic_infiltration(prcp, cusl1, cusl2, b, husl1, husl2, runoff)
        
            implicit none
            
            real(sp), intent(in) :: prcp, cusl1, cusl2, b
            real(sp), intent(inout) :: husl1, husl2
            real(sp), intent(out) :: runoff
            
            real(sp) :: bp1, ifl, cusl, wusl, iflm, iflc, ifl_usl1, ifl_usl2
            
            bp1 = b + 1._sp
            
            if (prcp .le. 0._sp) then
            
                ifl = 0._sp
                
            else
                
                cusl = cusl1 + cusl2
                
                wusl = husl1 * cusl1 + husl2 * cusl2
                
                wusl = max(1.e-6, wusl)
                wusl = min(cusl - 1e-6, wusl)
                
                iflm = cusl * bp1
                
                iflc = iflm * (1._sp - (1._sp - (wusl / cusl)) ** (1._sp / bp1))
                
                if (iflc + prcp .ge. iflm) then
                
                    ifl = cusl - wusl
                    
                else
                
                    ifl = (cusl - wusl) - cusl * (1._sp - ((iflc + prcp) / iflm)) ** bp1 
                
                end if
                
                ifl = min(prcp, ifl)
            
            end if
            
            ifl_usl1 = min((1._sp - husl1) * cusl1, ifl)
            
            ifl = ifl - ifl_usl1
            
            ifl_usl2 = min((1._sp - husl2) * cusl2, ifl)
            
            ifl = ifl - ifl_usl2
            
            husl1 = husl1 + ifl_usl1 / cusl1
            husl2 = husl2 + ifl_usl2 / cusl2
            
            runoff = prcp - (ifl_usl1 + ifl_usl2)
        
        end subroutine vic_infiltration
        
        
        subroutine vic_vertical_transfer(pet, cusl1, cusl2, clsl, ks, husl1, husl2, hlsl)
        
            implicit none
            
            real(sp), intent(in) :: pet, cusl1, cusl2, clsl, ks
            real(sp), intent(inout) :: husl1, husl2, hlsl
            
            real(sp) :: fbc, fe, pet_remain
            
            call brooks_and_corey_flow(ks, 0._sp, 1._sp, 1._sp, cusl1, cusl2, husl1, husl2, fbc)
            
            husl1 = husl1 - fbc / cusl1
            husl2 = husl2 + fbc / cusl2
            
            call brooks_and_corey_flow(ks, 0._sp, 1._sp, 1._sp, cusl2, clsl, husl2, hlsl, fbc)
            
            husl2 = husl2 - fbc / cusl2
            hlsl = hlsl + fbc / clsl
            
            call linear_evapotranspiration(pet, cusl1, husl1, fe)
            
            husl1 = husl1 - fe / cusl1
            
            pet_remain = max(0._sp, pet - fe)
            
            call linear_evapotranspiration(pet_remain, cusl2, husl2, fe)
            
            husl2 = husl2 - fe / cusl2
            
            pet_remain = max(0._sp, pet_remain - fe)
            
            call linear_evapotranspiration(pet_remain, clsl, hlsl, fe)
            
            hlsl = hlsl - fe / clsl
        
        end subroutine vic_vertical_transfer
        
        
        subroutine vic_interflow(n, cusl2, husl2, qi)
        
            implicit none
            
            real(sp), intent(in) :: n, cusl2
            real(sp), intent(inout) :: husl2
            real(sp), intent(out) :: qi
            
            real(sp) :: nm1, d1pnm1, husl2_imd
            
            nm1 = n - 1._sp
            d1pnm1 = 1._sp / nm1
            
            husl2_imd = husl2
            
            husl2 = (((husl2_imd * cusl2) ** (- nm1) + cusl2 ** (- nm1)) ** (- d1pnm1)) / cusl2
            
            qi = (husl2_imd - husl2) * cusl2
        
        end subroutine vic_interflow
        
        
        subroutine vic_baseflow(clsl, ds, dsm, ws, hlsl, qb)
        
            implicit none
            
            real(sp), intent(in) :: clsl, ds, dsm, ws
            real(sp), intent(inout) :: hlsl
            real(sp), intent(out) :: qb
            
            real(sp) :: wlsl
            
            if (hlsl .le. ws) then
            
                qb = (ds * dsm) / ws * hlsl
                
            else
                
                qb = dsm * (1._sp - ds / ws) * (hlsl - ws) / (1._sp - ws)
                
            end if
            
            wlsl = clsl * hlsl
            
            qb = min(wlsl, qb)
            
            hlsl = hlsl - qb / clsl
        
        end subroutine vic_baseflow
        
        
        subroutine brooks_and_corey_flow(ks, residual, porosity, lambda, c_upper, c_lower, h_upper, h_lower, flow)
        
            implicit none
            
            real(sp), intent(in) :: ks, residual, porosity, lambda, c_upper, c_lower, h_upper, h_lower
            real(sp), intent(out) :: flow
            
            real(sp) :: w_upper, w_lower, max_flow
            
            flow = ks * ((h_upper - residual) / (porosity - residual)) ** lambda
            
            w_upper = h_upper * c_upper * porosity
            w_lower = h_lower * c_lower * porosity
            
            max_flow = min(w_upper, c_lower - w_lower)
            
            flow = min(max_flow, flow)
        
        end subroutine brooks_and_corey_flow
        
        
        subroutine linear_evapotranspiration(e, c, h, flow)
        
            implicit none
            
            real(sp), intent(in) :: e, c, h
            real(sp), intent(out) :: flow
            
            real(sp) :: w
            
            flow = e * h
            
            w = c * h
            
            flow = min(w, flow)
        
        end subroutine linear_evapotranspiration
        
end module md_vic_operator
