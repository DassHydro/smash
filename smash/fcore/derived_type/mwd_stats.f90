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

        real(sp), dimension(:, :, :), allocatable :: internal_fluxes
        character(lchar), dimension(:), allocatable :: fluxes_keys !$F90W char-array
        real(sp), dimension(:, :, :, :), allocatable :: fluxes_values
        character(lchar), dimension(:), allocatable :: rr_states_keys !$F90W char-array
        real(sp), dimension(:, :, :, :), allocatable :: rr_states_values ! rr_states_keys in Rr_StatesDT class

    end type StatsDT

contains

    subroutine StatsDT_initialise(this, setup, mesh)
        !% Default states value will be handled in Python

        implicit none

        type(StatsDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%internal_fluxes(mesh%nrow, mesh%ncol, setup%nfx))

        allocate (this%fluxes_keys(setup%nfx))
        this%fluxes_keys = "..."

        allocate (this%fluxes_values(mesh%ng, setup%ntime_step, 5, setup%nfx))

        allocate (this%rr_states_keys(setup%nrrs))
        this%rr_states_keys = "..."

        allocate (this%rr_states_values(mesh%ng, setup%ntime_step, 5, setup%nrrs))

    end subroutine StatsDT_initialise

    subroutine StatsDT_copy(this, this_copy)

        implicit none

        type(StatsDT), intent(in) :: this
        type(StatsDT), intent(out) :: this_copy

        this_copy = this

    end subroutine StatsDT_copy
end module
