!%    This module `m_operator` encapsulates all SMASH operator (type, subroutines, functions)
module m_operator
    
    use m_common, only: sp, dp, lchar, np, ns

    implicit none
    
    contains
    
        subroutine GR_production(p, e, cp, beta, hp, pr, perc)
        
            implicit none
            
            real(sp), intent(in) :: p, e, cp, beta
            real(sp), intent(inout) :: hp
            real(sp), intent(out) :: pr, perc
            
            real(sp) :: inv_cp, ps, es, hp_imd
            
            inv_cp = 1._sp / cp
            pr = 0._sp
            
            ps = cp * (1._sp - hp * hp) * tanh(p * inv_cp) / &
            & (1._sp + hp * tanh(p * inv_cp))
            
            es = hp * cp + (2._sp - hp) * tanh(e * inv_cp) / &
            & (1._sp + (1._sp - hp) * tanh(e * inv_cp))
            
            hp_imd = hp + (ps - es) * inv_cp
            
            if (p .gt. 0) then
            
                pr = p - (hp_imd - hp) * cp
            
            end if
            
            perc = hp_imd * cp * (1._sp - (1._sp + ((hp_imd / beta) ** 4) ** - 0.25_sp))
            
            hp = hp_imd - perc * inv_cp

        end subroutine GR_production
        

end module m_operator
