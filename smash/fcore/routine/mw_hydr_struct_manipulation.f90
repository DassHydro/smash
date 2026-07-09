!%      (MW) Module Wrapped.
!%
!%      Subroutine
!%      ----------
!%
!%      - reallocate_dam_struct_data

module mw_hydr_struct_manipulation

    use md_constant, only: sp, lchar
    use mwd_mesh, only: MeshDT
    use mwd_hydraulic_structure

    implicit none

contains

    subroutine reallocate_dam_struct_data(this, key, ndam, nmax_val)

        implicit none

        type(DamDT), intent(inout) :: this
        character(lchar), intent(in) :: key
        integer, intent(in) :: ndam, nmax_val

        select case (key)

        case ("dam_hv")
            if (allocated(this%dam_hv)) then
                deallocate (this%dam_hv)
                allocate (this%dam_hv(ndam, 2, nmax_val))
            end if
            this%dam_hv = -99.

        case ("dam_hq")
            if (allocated(this%dam_hq)) then
                deallocate (this%dam_hq)
                allocate (this%dam_hq(ndam, 2, nmax_val))
            end if
            this%dam_hq = -99.

        end select

    end subroutine reallocate_dam_struct_data

end module mw_hydr_struct_manipulation
