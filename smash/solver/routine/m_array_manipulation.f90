!%      This module `m_array_manipulation` encapsulates all SMASH array_manipulation.
!%
!%      ma_flatten interface:
!%
!%      module procedure ma_flatten2d_i
!%      module procedure ma_flatten2d_r
!%      module procedure ma_flatten3d_i
!%      module procedure ma_flatten3d_r
!%
!%      flatten interface:
!%
!%      module procedure flatten2d_i
!%      module procedure flatten2d_r
!%      module procedure flatten3d_i
!%      module procedure flatten3d_r
!%
!%      contains
!%
!%      [1]  ma_flatten2d_i
!%      [2]  ma_flatten2d_r
!%      [3]  ma_flatten3d_i
!%      [4]  ma_flatten3d_r
!%      [5]  flatten2d_i
!%      [6]  flatten2d_r
!%      [7]  flatten3d_i
!%      [8]  flatten3d_r

module m_array_manipulation

   use md_constant, only: sp

   implicit none

   interface ma_flatten

      module procedure ma_flatten2d_i
      module procedure ma_flatten2d_r
      module procedure ma_flatten3d_i
      module procedure ma_flatten3d_r

   end interface ma_flatten

   interface flatten

      module procedure flatten2d_i
      module procedure flatten2d_r
      module procedure flatten3d_i
      module procedure flatten3d_r

   end interface flatten

contains

   subroutine ma_flatten2d_i(a, mask, res)

      !% Notes
      !% -----
      !%
      !% Masked flatten subroutine
      !%
      !% Given an integer matrix and logical mask of dim(2),
      !% it returns an integer array of dim(1) and size(count(mask))

      implicit none

      integer, dimension(:, :), intent(in) :: a
      logical, dimension(size(a, 1), size(a, 2)), intent(in) :: mask
      integer, dimension(:), allocatable, intent(inout) :: res

      integer :: i, j, n

      if (allocated(res)) deallocate (res)

      allocate (res(count(mask)))

      n = 1

      do i = 1, size(a, 2)

         do j = 1, size(a, 1)

            if (mask(j, i)) then

               res(n) = a(j, i)
               n = n + 1

            end if

         end do

      end do

   end subroutine ma_flatten2d_i

   subroutine ma_flatten2d_r(a, mask, res)

      !% Notes
      !% -----
      !%
      !% Masked flatten subroutine
      !%
      !% Given a single precision matrix and logical mask of dim(2),
      !% it returns a single precision array of dim(1) and size(count(mask))

      implicit none

      real(sp), dimension(:, :), intent(in) :: a
      logical, dimension(size(a, 1), size(a, 2)), intent(in) :: mask
      real(sp), dimension(:), allocatable, intent(inout) :: res

      integer :: i, j, n

      if (allocated(res)) deallocate (res)

      allocate (res(count(mask)))

      n = 1

      do i = 1, size(a, 2)

         do j = 1, size(a, 1)

            if (mask(j, i)) then

               res(n) = a(j, i)
               n = n + 1

            end if

         end do

      end do

   end subroutine ma_flatten2d_r

   subroutine ma_flatten3d_i(a, mask, res)

      !% Notes
      !% -----
      !%
      !% Masked flatten subroutine
      !%
      !% Given an integer matrix and logical mask of dim(3),
      !% it returns an integer array of dim(1) and size(count(mask))

      implicit none

      integer, dimension(:, :, :), intent(in) :: a
      logical, dimension(size(a, 1), size(a, 2), size(a, 3)), intent(in) :: mask
      integer, dimension(:), allocatable, intent(inout) :: res

      integer :: i, j, k, n

      if (allocated(res)) deallocate (res)

      allocate (res(count(mask)))

      n = 1

      do i = 1, size(a, 3)

         do j = 1, size(a, 2)

            do k = 1, size(a, 1)

               if (mask(k, j, i)) then

                  res(n) = a(k, j, i)
                  n = n + 1

               end if

            end do

         end do

      end do

   end subroutine ma_flatten3d_i

   subroutine ma_flatten3d_r(a, mask, res)

      !% Notes
      !% -----
      !%
      !% Masked flatten subroutine
      !%
      !% Given a single precision matrix and logical mask of dim(3),
      !% it returns a single precision array of dim(1) and size(count(mask))

      implicit none

      real(sp), dimension(:, :, :), intent(in) :: a
      logical, dimension(size(a, 1), size(a, 2), size(a, 3)), intent(in) :: mask
      real(sp), dimension(:), allocatable, intent(inout) :: res

      integer :: i, j, k, n

      if (allocated(res)) deallocate (res)

      allocate (res(count(mask)))

      n = 1

      do i = 1, size(a, 3)

         do j = 1, size(a, 2)

            do k = 1, size(a, 1)

               if (mask(k, j, i)) then

                  res(n) = a(k, j, i)
                  n = n + 1

               end if

            end do

         end do

      end do

   end subroutine ma_flatten3d_r

   subroutine flatten2d_i(a, res)

      !% Notes
      !% -----
      !%
      !% Flatten subroutine
      !%
      !% Given an integer matrix of dim(2),
      !% it returns an integer array of dim(1) and size(size(matrix))

      implicit none

      integer, dimension(:, :), intent(in) :: a
      integer, dimension(size(a)), intent(inout) :: res

      integer :: i, j, n

      n = 1

      do i = 1, size(a, 2)

         do j = 1, size(a, 1)

            res(n) = a(j, i)
            n = n + 1

         end do

      end do

   end subroutine flatten2d_i

   subroutine flatten2d_r(a, res)

      !% Notes
      !% -----
      !%
      !% Flatten subroutine
      !%
      !% Given a single precision matrix of dim(2),
      !% it returns a single precision array of dim(1) and size(size(matrix))

      implicit none

      real(sp), dimension(:, :), intent(in) :: a
      real(sp), dimension(size(a)), intent(inout) :: res

      integer :: i, j, n

      n = 1

      do i = 1, size(a, 2)

         do j = 1, size(a, 1)

            res(n) = a(j, i)
            n = n + 1

         end do

      end do

   end subroutine flatten2d_r

   subroutine flatten3d_i(a, res)

      !% Notes
      !% -----
      !%
      !% Flatten subroutine
      !%
      !% Given an integer matrix of dim(3),
      !% it returns an integer array of dim(1) and size(size(matrix))

      implicit none

      integer, dimension(:, :, :), intent(in) :: a
      integer, dimension(size(a)), intent(inout) :: res

      integer :: i, j, k, n

      n = 1

      do i = 1, size(a, 3)

         do j = 1, size(a, 2)

            do k = 1, size(a, 1)

               res(n) = a(k, j, i)
               n = n + 1

            end do

         end do

      end do

   end subroutine flatten3d_i

   subroutine flatten3d_r(a, res)

      !% Notes
      !% -----
      !%
      !% Flatten subroutine
      !%
      !% Given a single precision matrix of dim(2),
      !% it returns a single precision array of dim(1) and size(size(matrix))

      implicit none

      real(sp), dimension(:, :, :), intent(in) :: a
      real(sp), dimension(size(a)), intent(inout) :: res

      integer :: i, j, k, n

      n = 1

      do i = 1, size(a, 3)

         do j = 1, size(a, 2)

            do k = 1, size(a, 1)

               res(n) = a(k, j, i)
               n = n + 1

            end do

         end do

      end do

   end subroutine flatten3d_r

end module m_array_manipulation
