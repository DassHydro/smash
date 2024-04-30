!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Sparse_MatrixDT
!%          Sparse matrices handling atmospheric data (prcp, pet, snow ...)
!%          See COO matrices (google, scipy)
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``n``                    Number of data stored
!%          ``coo_fmt``              Sparse Matrix in COO format       (default: .true.)
!%          ``zvalue``               Non stored value                  (default: 0)
!%          ``indices``              Indices of the sparse matrix
!%          ``values``               Values of the sparse matrix
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Sparse_MatrixDT_initialise
!%      - Sparse_MatrixDT_copy

module mwd_sparse_matrix

    use md_constant !% only: sp

    implicit none

    type Sparse_MatrixDT

        integer :: n
        logical :: coo_fmt = .true.
        real(sp) :: zvalue = 0._sp
        integer, dimension(:), allocatable :: indices
        real(sp), dimension(:), allocatable :: values

    end type Sparse_MatrixDT

contains

    subroutine Sparse_MatrixDT_initialise(this, n, coo_fmt, zvalue)

        implicit none

        type(Sparse_MatrixDT), intent(inout) :: this
        integer, intent(in) :: n
        logical, intent(in) :: coo_fmt
        real(sp), intent(in) :: zvalue

        call Sparse_MatrixDT_finalise(this)

        this%n = n
        this%coo_fmt = coo_fmt
        this%zvalue = zvalue

        allocate (this%values(this%n))
        this%values = 0._sp

        if (coo_fmt) then

            allocate (this%indices(this%n))
            this%indices = 0

        end if

    end subroutine Sparse_MatrixDT_initialise

    subroutine Sparse_MatrixDT_finalise(this)

        implicit none

        type(Sparse_MatrixDT), intent(inout) :: this

        if (allocated(this%values)) deallocate (this%values)
        if (allocated(this%indices)) deallocate (this%indices)

    end subroutine Sparse_MatrixDT_finalise

    subroutine Sparse_MatrixDT_initialise_array(this, n, coo_fmt, zvalue)

        implicit none

        type(Sparse_MatrixDT), dimension(:), intent(inout) :: this
        integer, intent(in) :: n
        logical, intent(in) :: coo_fmt
        real(sp), intent(in) :: zvalue

        integer :: i

        do i = 1, size(this)

            call Sparse_MatrixDT_initialise(this(i), n, coo_fmt, zvalue)

        end do

    end subroutine Sparse_MatrixDT_initialise_array

    ! To manually alloc from Python in place. ControlDT_initialise is used as
    ! __init__ method (implemented by f90wrap automatically)
    subroutine Sparse_MatrixDT_alloc(this, n, coo_fmt, zvalue)

        implicit none

        type(Sparse_MatrixDT), intent(inout) :: this
        integer, intent(in) :: n
        logical, intent(in) :: coo_fmt
        real(sp), intent(in) :: zvalue

        call Sparse_MatrixDT_initialise(this, n, coo_fmt, zvalue)

    end subroutine Sparse_MatrixDT_alloc

    subroutine Sparse_MatrixDT_copy(this, this_copy)

        implicit none

        type(Sparse_MatrixDT), intent(in) :: this
        type(Sparse_MatrixDT), intent(inout) :: this_copy

        this_copy = this

    end subroutine Sparse_MatrixDT_copy

end module mwd_sparse_matrix
