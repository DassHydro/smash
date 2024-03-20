!%      (MW) Module Wrapped and Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      - get_atmos_data_timestep
!%      - set_atmos_data_timestep
!%      - get_ac_atmos_data_timestep
!%      - set_ac_atmos_data_timestep

module mwd_atmos_manipulation

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_sparse_matrix_manipulation !% only: sparse_matrix_to_matrix, matrix_to_sparse_matrix, &
    !% ac_vector_to_matrix, matrix_to_ac_vector

    implicit none

contains

    subroutine get_atmos_data_time_step(setup, mesh, input_data, time_step, key, vle)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, intent(in) :: time_step
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: vle

        select case (trim(key))

        case ("prcp")

            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(time_step), vle)

            else
                vle = input_data%atmos_data%prcp(:, :, time_step)

            end if

        case ("pet")

            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(time_step), vle)

            else
                vle = input_data%atmos_data%pet(:, :, time_step)

            end if

        case ("snow")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_snow(time_step), vle)

            else
                vle = input_data%atmos_data%snow(:, :, time_step)

            end if

        case ("temp")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_temp(time_step), vle)

            else
                vle = input_data%atmos_data%temp(:, :, time_step)

            end if

        end select

    end subroutine get_atmos_data_time_step

    subroutine set_atmos_data_time_step(setup, mesh, input_data, time_step, key, vle)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        integer, intent(in) :: time_step
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: vle

        select case (trim(key))

        case ("prcp")

            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_prcp(time_step))

            else
                input_data%atmos_data%prcp(:, :, time_step) = vle

            end if

        case ("pet")

            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_pet(time_step))

            else
                input_data%atmos_data%pet(:, :, time_step) = vle

            end if

        case ("snow")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_snow(time_step))

            else
                input_data%atmos_data%snow(:, :, time_step) = vle

            end if

        case ("temp")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_temp(time_step))

            else
                input_data%atmos_data%temp(:, :, time_step) = vle

            end if

        end select

    end subroutine set_atmos_data_time_step

    subroutine get_ac_atmos_data_time_step(setup, mesh, input_data, time_step, key, ac_vector)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, intent(in) :: time_step
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nac), intent(inout) :: ac_vector

        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix

        call get_atmos_data_time_step(setup, mesh, input_data, time_step, key, matrix)

        call matrix_to_ac_vector(mesh, matrix, ac_vector)

    end subroutine get_ac_atmos_data_time_step

    subroutine set_ac_atmos_data_time_step(setup, mesh, input_data, time_step, key, ac_vector)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        integer, intent(in) :: time_step
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nac), intent(in) :: ac_vector

        real(sp), dimension(mesh%nrow, mesh%ncol) :: matrix

        call ac_vector_to_matrix(mesh, ac_vector, matrix)

        call set_atmos_data_time_step(setup, mesh, input_data, time_step, key, matrix)

    end subroutine set_ac_atmos_data_time_step

end module mwd_atmos_manipulation
