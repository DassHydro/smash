!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - SegmentDT
!%
!%      Subroutine
!%      ----------
!%
!%      - SegmentDT_initialise
!%      - SegmentDT_copy

module mwd_segment

    use md_constant !%only: sp

    implicit none

    type SegmentDT

        integer :: first_cross_section !$F90W index
        integer :: last_cross_section !$F90W index

        integer :: nds_seg
        integer, dimension(:), allocatable :: ds_segment !$F90W index-array
        integer :: nus_seg
        integer, dimension(:), allocatable :: us_segment !$F90W index-array

    end type SegmentDT

contains

    subroutine SegmentDT_initialise(this, nds_seg, nus_seg)

        implicit none

        type(SegmentDT) :: this
        integer, intent(in) :: nds_seg
        integer, intent(in) :: nus_seg

        this%nds_seg = nds_seg
        this%nus_seg = nus_seg

        this%first_cross_section = -99
        this%last_cross_section = -99

        allocate (this%ds_segment(this%nds_seg))
        this%ds_segment = -99
        allocate (this%us_segment(this%nus_seg))
        this%us_segment = -99

    end subroutine SegmentDT_initialise

    subroutine SegmentDT_copy(this, this_copy)

        implicit none

        type(SegmentDT), intent(in) :: this
        type(SegmentDT), intent(out) :: this_copy

        this_copy = this

    end subroutine SegmentDT_copy

end module mwd_segment
