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
        real(sp), dimension(:), allocatable :: manning
        real(sp), dimension(:), allocatable :: level_heights
        real(sp), dimension(:), allocatable :: level_widths

        integer :: nlat
        integer, dimension(:, :), allocatable :: lat_rowcols !$F90W index-array
        integer :: nup
        integer, dimension(:, :), allocatable :: up_rowcols !$F90W index-array

        integer :: ids_cs                                    ! Downstream CS index
        integer :: nus_cs                                    ! Number of upstream CS
        integer, dimension(:), allocatable :: ius_cs         ! Upstream CS indices
        real(sp) :: dx                                       ! Distance to downstream
        logical :: is_outlet                                 ! Outlet flag
        real(sp) :: bathy_bc                                 ! Boundary condition bathymetry (outlet only)



    end type Cross_SectionDT

contains

    subroutine Cross_SectionDT_initialise(this, nlevels, nlat, nup, nus_cs)

        implicit none

        type(Cross_SectionDT) :: this
        integer, intent(in) :: nlevels
        integer, intent(in) :: nlat
        integer, intent(in) :: nup
        integer, intent(in) :: nus_cs

        this%nlevels = nlevels
        this%nlat = nlat
        this%nup = nup
        this%nus_cs = nus_cs

        this%coord = -99._sp
        this%rowcol = -99
        this%x = -99._sp
        this%bathy = -99._sp

        allocate (this%manning(this%nlevels))
        this%manning = -99._sp
        allocate (this%level_heights(this%nlevels))
        this%level_heights = -99._sp
        allocate (this%level_widths(this%nlevels))
        this%level_widths = -99._sp

        allocate (this%lat_rowcols(this%nlat, 2))
        this%lat_rowcols = -99
        allocate (this%up_rowcols(this%nup, 2))
        this%up_rowcols = -99

        this%ids_cs = -99
        allocate (this%ius_cs(this%nus_cs))
        this%ius_cs = -99
        this%dx = -99._sp
        this%is_outlet = .false.
        this%bathy_bc = -999._sp

    end subroutine Cross_SectionDT_initialise

    subroutine Cross_SectionDT_copy(this, this_copy)

        implicit none

        type(Cross_SectionDT), intent(in) :: this
        type(Cross_SectionDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Cross_SectionDT_copy

end module mwd_cross_section
