!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Physio_DataDT
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``descriptor``           Descriptor maps field                       [(descriptor dependent)]
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Physio_DataDT_initialise
!%      - Physio_DataDT_copy

module mwd_physio_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    implicit none

    type Physio_DataDT

        real(sp), dimension(:, :, :), allocatable :: descriptor

    end type Physio_DataDT

contains

    subroutine Physio_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Physio_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%descriptor(mesh%nrow, mesh%ncol, setup%nd))
        this%descriptor = -99._sp

    end subroutine Physio_DataDT_initialise

    subroutine Physio_DataDT_copy(this, this_copy)

        implicit none

        type(Physio_DataDT), intent(in) :: this
        type(Physio_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Physio_DataDT_copy

end module mwd_physio_data
