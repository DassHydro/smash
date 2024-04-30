!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Physio_DataDT
!%           Physiographic data used to force the regionalization, among other things.
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``descriptor``           Descriptor maps field                       [(descriptor dependent)]
!%          ``l_descriptor``         Descriptor maps field min value             [(descriptor dependent)]
!%          ``u_descriptor``         Descriptor maps field max value             [(descriptor dependent)]
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
        real(sp), dimension(:), allocatable :: l_descriptor
        real(sp), dimension(:), allocatable :: u_descriptor

    end type Physio_DataDT

contains

    subroutine Physio_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Physio_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%descriptor(mesh%nrow, mesh%ncol, setup%nd))
        this%descriptor = -99._sp

        allocate (this%l_descriptor(setup%nd))
        this%l_descriptor = -99._sp

        allocate (this%u_descriptor(setup%nd))
        this%u_descriptor = -99._sp

    end subroutine Physio_DataDT_initialise

    subroutine Physio_DataDT_copy(this, this_copy)

        implicit none

        type(Physio_DataDT), intent(in) :: this
        type(Physio_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Physio_DataDT_copy

end module mwd_physio_data
