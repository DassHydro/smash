!%      (MWD) Module Wrapped and Differentiated
!%
!%      Subroutine
!%      ----------
!%
!%      - map_control_to_parameters

module mwd_parameters_manipulation

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_parameters !% only: ParametersDT
    use mwd_options !% only: OptionsDT

    implicit none

contains

    subroutine opr_parameters_to_matrix(setup, mesh, parameters, matrix)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_p), intent(inout) :: matrix

        select case (setup%structure)

        case ("gr_a")

            matrix(:, :, 1) = parameters%opr_parameters%gr_a%cp
            matrix(:, :, 2) = parameters%opr_parameters%gr_a%cft
            matrix(:, :, 3) = parameters%opr_parameters%gr_a%exc
            matrix(:, :, 4) = parameters%opr_parameters%gr_a%lr

        case ("gr_b")

            matrix(:, :, 1) = parameters%opr_parameters%gr_b%cp
            matrix(:, :, 2) = parameters%opr_parameters%gr_b%cft
            matrix(:, :, 3) = parameters%opr_parameters%gr_b%exc
            matrix(:, :, 4) = parameters%opr_parameters%gr_b%lr

        case ("gr_c")

            matrix(:, :, 1) = parameters%opr_parameters%gr_c%cp
            matrix(:, :, 2) = parameters%opr_parameters%gr_c%cft
            matrix(:, :, 3) = parameters%opr_parameters%gr_c%cst
            matrix(:, :, 4) = parameters%opr_parameters%gr_c%exc
            matrix(:, :, 5) = parameters%opr_parameters%gr_c%lr

        case ("gr_d")

            matrix(:, :, 1) = parameters%opr_parameters%gr_d%cp
            matrix(:, :, 2) = parameters%opr_parameters%gr_d%cft
            matrix(:, :, 3) = parameters%opr_parameters%gr_d%lr

        end select

    end subroutine opr_parameters_to_matrix

    subroutine opr_initial_states_to_matrix(setup, mesh, parameters, matrix)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_s), intent(inout) :: matrix

        select case (setup%structure)

        case ("gr_a")

            matrix(:, :, 1) = parameters%opr_initial_states%gr_a%hp
            matrix(:, :, 2) = parameters%opr_initial_states%gr_a%hft
            matrix(:, :, 3) = parameters%opr_initial_states%gr_a%hlr

        case ("gr_b")

            matrix(:, :, 1) = parameters%opr_initial_states%gr_b%hi
            matrix(:, :, 2) = parameters%opr_initial_states%gr_b%hp
            matrix(:, :, 3) = parameters%opr_initial_states%gr_b%hft
            matrix(:, :, 4) = parameters%opr_initial_states%gr_b%hlr

        case ("gr_c")

            matrix(:, :, 1) = parameters%opr_initial_states%gr_c%hi
            matrix(:, :, 2) = parameters%opr_initial_states%gr_c%hp
            matrix(:, :, 3) = parameters%opr_initial_states%gr_c%hft
            matrix(:, :, 4) = parameters%opr_initial_states%gr_c%hst
            matrix(:, :, 5) = parameters%opr_initial_states%gr_c%hlr

        case ("gr_d")

            matrix(:, :, 1) = parameters%opr_initial_states%gr_d%hp
            matrix(:, :, 2) = parameters%opr_initial_states%gr_d%hft
            matrix(:, :, 3) = parameters%opr_initial_states%gr_d%hlr

        end select

    end subroutine opr_initial_states_to_matrix

    subroutine matrix_to_opr_parameters(setup, mesh, matrix, parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_p), intent(in) :: matrix
        type(ParametersDT), intent(inout) :: parameters

        select case (setup%structure)

        case ("gr_a")

            parameters%opr_parameters%gr_a%cp = matrix(:, :, 1)
            parameters%opr_parameters%gr_a%cft = matrix(:, :, 2)
            parameters%opr_parameters%gr_a%exc = matrix(:, :, 3)
            parameters%opr_parameters%gr_a%lr = matrix(:, :, 4)

        case ("gr_b")

            parameters%opr_parameters%gr_b%cp = matrix(:, :, 1)
            parameters%opr_parameters%gr_b%cft = matrix(:, :, 2)
            parameters%opr_parameters%gr_b%exc = matrix(:, :, 3)
            parameters%opr_parameters%gr_b%lr = matrix(:, :, 4)

        case ("gr_c")

            parameters%opr_parameters%gr_c%cp = matrix(:, :, 1)
            parameters%opr_parameters%gr_c%cft = matrix(:, :, 2)
            parameters%opr_parameters%gr_c%cst = matrix(:, :, 3)
            parameters%opr_parameters%gr_c%exc = matrix(:, :, 4)
            parameters%opr_parameters%gr_c%lr = matrix(:, :, 5)

        case ("gr_d")

            parameters%opr_parameters%gr_d%cp = matrix(:, :, 1)
            parameters%opr_parameters%gr_d%cft = matrix(:, :, 2)
            parameters%opr_parameters%gr_d%lr = matrix(:, :, 3)

        end select

    end subroutine matrix_to_opr_parameters

    subroutine matrix_to_opr_initial_states(setup, mesh, matrix, parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_s), intent(in) :: matrix
        type(ParametersDT), intent(inout) :: parameters

        select case (setup%structure)

        case ("gr_a")

            parameters%opr_initial_states%gr_a%hp = matrix(:, :, 1)
            parameters%opr_initial_states%gr_a%hft = matrix(:, :, 2)
            parameters%opr_initial_states%gr_a%hlr = matrix(:, :, 3)

        case ("gr_b")

            parameters%opr_initial_states%gr_b%hi = matrix(:, :, 1)
            parameters%opr_initial_states%gr_b%hp = matrix(:, :, 2)
            parameters%opr_initial_states%gr_b%hft = matrix(:, :, 3)
            parameters%opr_initial_states%gr_b%hlr = matrix(:, :, 4)

        case ("gr_c")

            parameters%opr_initial_states%gr_c%hi = matrix(:, :, 1)
            parameters%opr_initial_states%gr_c%hp = matrix(:, :, 2)
            parameters%opr_initial_states%gr_c%hft = matrix(:, :, 3)
            parameters%opr_initial_states%gr_c%hst = matrix(:, :, 4)
            parameters%opr_initial_states%gr_c%hlr = matrix(:, :, 5)

        case ("gr_d")

            parameters%opr_initial_states%gr_d%hp = matrix(:, :, 1)
            parameters%opr_initial_states%gr_d%hft = matrix(:, :, 2)
            parameters%opr_initial_states%gr_d%hlr = matrix(:, :, 3)

        end select

    end subroutine matrix_to_opr_initial_states

    subroutine map_control_to_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_p) :: opr_parameters_matrix
        real(sp), dimension(mesh%nrow, mesh%ncol, setup%nopr_s) :: opr_initial_states_matrix

        call opr_parameters_to_matrix(setup, mesh, parameters, opr_parameters_matrix)
        call opr_initial_states_to_matrix(setup, mesh, parameters, opr_initial_states_matrix)

!~         select case (options%optimize%mapping)

!~         case ("uniform")

!~         case ("distributed")

!~         end select

        call matrix_to_opr_parameters(setup, mesh, opr_parameters_matrix, parameters)
        call matrix_to_opr_initial_states(setup, mesh, opr_initial_states_matrix, parameters)

    end subroutine map_control_to_parameters

end module mwd_parameters_manipulation
