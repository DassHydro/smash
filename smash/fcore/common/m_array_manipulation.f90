!%      (M) Module.
!%

module m_array_manipulation

    use md_constant, only: sp

    implicit none

    ! Can be fill with 2d, 3d etc
    interface reallocate

        module procedure reallocate1d_i
        module procedure reallocate1d_r

    end interface reallocate

contains

    subroutine reallocate1d_i(arr, nsz)

        implicit none

        integer, dimension(:), allocatable, intent(inout) :: arr
        integer, intent(in) :: nsz

        integer :: osz
        integer, dimension(:), allocatable :: warr

        intrinsic move_alloc

        osz = size(arr)

        if (osz .eq. nsz) then
            return

        else if (osz .lt. nsz) then
            allocate (warr(nsz))
            warr(1:nsz) = arr(1:nsz)
            call move_alloc(warr, arr)

        else if (osz .gt. nsz) then
            allocate (warr(nsz))
            warr(1:osz) = arr(1:osz)
            call move_alloc(warr, arr)

        end if
        ! warr automatically deallocated with move_alloc
    end subroutine reallocate1d_i

    subroutine reallocate1d_r(arr, nsz)

        implicit none

        real(sp), dimension(:), allocatable, intent(inout) :: arr
        integer, intent(in) :: nsz

        integer :: osz
        real(sp), dimension(:), allocatable :: warr

        intrinsic move_alloc

        osz = size(arr)

        if (osz .eq. nsz) then
            return

        else if (nsz .lt. osz) then
            allocate (warr(nsz))
            warr(1:nsz) = arr(1:nsz)
            call move_alloc(warr, arr)

        else
            allocate (warr(nsz))
            warr(1:osz) = arr(1:osz)
            call move_alloc(warr, arr)

        end if
        ! warr automatically deallocated with move_alloc
    end subroutine reallocate1d_r

end module m_array_manipulation
