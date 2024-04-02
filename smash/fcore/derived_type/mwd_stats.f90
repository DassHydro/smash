!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - StatsDT
!%        ...
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``keys``                   keys internal fluxes
!%          ``values ``                stats arrays
!%
!§      Subroutine
!%      ----------
!%
!%      - StatsDT_initialise
!%      - StatsDT_copy

module mwd_stats

    use md_constant !% only: sp, lchar
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type StatsDT

        character(lchar), dimension(:), allocatable :: keys !$F90W char-array
        real(sp), dimension(:, :, :, :), allocatable :: values 
        real(sp), dimension(:, :, :), allocatable :: internal_fluxes
        
    end type StatsDT

contains

    subroutine StatsDT_initialise(this, setup, mesh)
        !% Default states value will be handled in Python

        implicit none

        type(StatsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%keys(setup%nfx))
        this%keys = "..."

        allocate (this%values(mesh%ng, setup%ntime_step, 5, setup%nfx))
        
        allocate (this%internal_fluxes(mesh%nrow, mesh%ncol, setup%nfx))
        
    end subroutine StatsDT_initialise

    subroutine StatsDT_copy(this, this_copy)

        implicit none

        type(StatsDT), intent(in) :: this
        type(StatsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine StatsDT_copy
end module
