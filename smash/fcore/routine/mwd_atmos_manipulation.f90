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

    use md_constant
    use mwd_setup
    use mwd_mesh
    use mwd_input_data
    use mwd_sparse_matrix_manipulation

    implicit none

contains

    subroutine get_atmos_data_timestep(setup, mesh, input_data, timestep, key, vle)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        integer, intent(in) :: timestep
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(inout) :: vle

        select case (trim(key))

        case ("prcp")

            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_prcp(timestep), vle)

            else
                vle = input_data%atmos_data%prcp(:, :, timestep)

            end if

        case ("pet")

            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_pet(timestep), vle)

            else
                vle = input_data%atmos_data%pet(:, :, timestep)

            end if

        case ("snow")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_snow(timestep), vle)

            else
                vle = input_data%atmos_data%snow(:, :, timestep)

            end if

        case ("temp")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call sparse_matrix_to_matrix(mesh, input_data%atmos_data%sparse_temp(timestep), vle)

            else
                vle = input_data%atmos_data%temp(:, :, timestep)

            end if

        end select

    end subroutine get_atmos_data_timestep

    subroutine set_atmos_data_timestep(setup, mesh, input_data, timestep, key, vle)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(inout) :: input_data
        integer, intent(in) :: timestep
        character(*), intent(in) :: key
        real(sp), dimension(mesh%nrow, mesh%ncol), intent(in) :: vle

        select case (trim(key))

        case ("prcp")

            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_prcp(timestep))

            else
                input_data%atmos_data%prcp(:, :, timestep) = vle

            end if

        case ("pet")

            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_pet(timestep))

            else
                input_data%atmos_data%pet(:, :, timestep) = vle

            end if

        case ("snow")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_snow(timestep))

            else
                input_data%atmos_data%snow(:, :, timestep) = vle

            end if

        case ("temp")

            !% assert (setup%snow_module_present)
            if (setup%sparse_storage) then
                call matrix_to_sparse_matrix(mesh, vle, 0._sp, input_data%atmos_data%sparse_temp(timestep))

            else
                input_data%atmos_data%temp(:, :, timestep) = vle

            end if

        end select

    end subroutine set_atmos_data_timestep

end module mwd_atmos_manipulation
