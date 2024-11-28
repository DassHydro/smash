!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Cross_SectionDT
!%
!%      Subroutine
!%      ----------
!%
!%      - Cross_SectionDT_initialise
!%      - Cross_SectionDT_copy

module mwd_cross_section

    use md_constant !%only: sp

    implicit none

    type Cross_SectionDT

        real(sp), dimension(2) :: coord
        integer, dimension(2) :: rowcol !$F90W index-array
        real(sp) :: x
        real(sp) :: bathy
        integer :: nlevels
        real(sp), dimension(:), allocatable :: strickler
        real(sp), dimension(:), allocatable :: level_heights
        real(sp), dimension(:), allocatable :: level_widths

        integer :: nlat
        integer, dimension(:, :), allocatable :: lat_rowcols !$F90W index-array
        integer :: nup
        integer, dimension(:, :), allocatable :: up_rowcols !$F90W index-array

    end type Cross_SectionDT

contains

    subroutine Cross_SectionDT_initialise(this, nlevels, nlat, nup)

        implicit none

        type(Cross_SectionDT) :: this
        integer, intent(in) :: nlevels
        integer, intent(in) :: nlat
        integer, intent(in) :: nup

        this%nlevels = nlevels
        this%nlat = nlat
        this%nup = nup

        this%coord = -99._sp
        this%rowcol = -99
        this%x = -99._sp
        this%bathy = -99._sp

        allocate (this%strickler(this%nlevels))
        this%strickler = -99._sp
        allocate (this%level_heights(this%nlevels))
        this%level_heights = -99._sp
        allocate (this%level_widths(this%nlevels))
        this%level_widths = -99._sp

        allocate (this%lat_rowcols(this%nlat, 2))
        this%lat_rowcols = -99
        allocate (this%up_rowcols(this%nup, 2))
        this%up_rowcols = -99

    end subroutine Cross_SectionDT_initialise

    subroutine Cross_SectionDT_copy(this, this_copy)

        implicit none

        type(Cross_SectionDT), intent(in) :: this
        type(Cross_SectionDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Cross_SectionDT_copy

end module mwd_cross_section
