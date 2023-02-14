!%      This module `mw_interception_store` encapsulates all SMASH interception store routine.
!%      This module is wrapped

module mw_interception_store

   use md_constant, only: sp
   use m_array_creation, only: arange, linspace
   use mwd_setup, only: SetupDT
   use mwd_mesh, only: MeshDT
   use mwd_input_data, only: Input_DataDT
   use mwd_parameters, only: ParametersDT
   use md_gr_operator, only: gr_interception
   use mw_sparse_storage, only: sparse_vector_to_matrix_r

   implicit none

contains

   subroutine adjust_interception_store(setup, mesh, input_data, parameters, nday, day_index)

      implicit none

      type(SetupDT), intent(in) :: setup
      type(MeshDT), intent(in) :: mesh
      type(Input_DataDT), intent(in) :: input_data
      type(ParametersDT), intent(inout) :: parameters
      integer, intent(in) :: nday
      integer, dimension(setup%ntime_step), intent(in) :: day_index

      real(sp), dimension(mesh%nrow, mesh%ncol, nday) :: daily_prcp, daily_pet
      real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix_prcp, matrix_pet, h, daily_cumulated, sub_daily_cumulated
      real(sp), parameter :: stt = 0.1_sp, stp = 5._sp, step = 0.1_sp
      real(sp), dimension(ceiling((stp - stt)/step)) :: cmax
      real(sp), dimension(mesh%nrow, mesh%ncol, size(cmax)) :: diff
      real(sp) :: ec, pth, prcp, pet
      integer :: n, i, j, k, l, ind

      !% =========================================================================================================== %!
      !%   Daily aggregation of precipitation and evapotranspiration
      !% =========================================================================================================== %!

      daily_prcp = 0._sp
      daily_pet = 0._sp

      if (setup%sparse_storage) then

         call sparse_vector_to_matrix_r(mesh, input_data%sparse_prcp(:, 1), daily_prcp(:, :, 1))
         call sparse_vector_to_matrix_r(mesh, input_data%sparse_pet(:, 1), daily_pet(:, :, 1))

      else

         daily_prcp(:, :, 1) = input_data%prcp(:, :, 1)
         daily_pet(:, :, 1) = input_data%pet(:, :, 1)

      end if

      n = 1

      do i = 2, setup%ntime_step

         if (day_index(i) .ne. day_index(i - 1)) n = n + 1

         if (setup%sparse_storage) then

            call sparse_vector_to_matrix_r(mesh, input_data%sparse_prcp(:, i), matrix_prcp)
            call sparse_vector_to_matrix_r(mesh, input_data%sparse_pet(:, i), matrix_pet)

         else

            matrix_prcp = input_data%prcp(:, :, i)
            matrix_pet = input_data%pet(:, :, i)

         end if

         daily_prcp(:, :, n) = daily_prcp(:, :, n) + matrix_prcp
         daily_pet(:, :, n) = daily_pet(:, :, n) + matrix_pet

      end do

      daily_cumulated = 0._sp

      do i = 1, nday

         do j = 1, mesh%ncol

            do k = 1, mesh%nrow

               daily_cumulated(k, j) = daily_cumulated(k, j) + min(daily_prcp(k, j, i), daily_pet(k, j, i))

            end do

         end do

      end do

      !% =========================================================================================================== %!
      !%   Calculate interception storage
      !% =========================================================================================================== %!

      call arange(stt, stp, step, cmax)

      do i = 1, size(cmax)

         h = 0._sp
         sub_daily_cumulated = 0._sp

         do j = 1, setup%ntime_step

            do k = 1, mesh%ncol

               do l = 1, mesh%nrow

                  if (mesh%active_cell(l, k) .eq. 1 .and. mesh%local_active_cell(l, k) .eq. 1) then

                     if (setup%sparse_storage) then

                        ind = mesh%rowcol_to_ind_sparse(l, k)
                        prcp = input_data%sparse_prcp(ind, j)
                        pet = input_data%sparse_pet(ind, j)

                     else

                        prcp = input_data%prcp(l, k, j)
                        pet = input_data%pet(l, k, j)

                     end if

                     call gr_interception(prcp, pet, cmax(i), h(l, k), pth, ec)
                     sub_daily_cumulated(l, k) = sub_daily_cumulated(l, k) + ec

                  end if

               end do

            end do

         end do

         diff(:, :, i) = abs(sub_daily_cumulated - daily_cumulated)

      end do

      do i = 1, mesh%ncol

         do j = 1, mesh%nrow

            if (mesh%active_cell(j, i) .eq. 1 .and. mesh%local_active_cell(j, i) .eq. 1) then

               ! Specify dim=1 to return an integer
               ind = minloc(diff(j, i, :), dim=1)

               parameters%ci(j, i) = cmax(ind)

            end if

         end do

      end do

   end subroutine adjust_interception_store

end module mw_interception_store
