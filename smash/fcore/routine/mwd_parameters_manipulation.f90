!%      (MWD) Module Wrapped and Differentiated
!%
!%      Subroutine
!%      ----------
!%
!%      - uniform_get_control_size
!%      - distributed_get_control_size
!%      - multi_linear_get_control_size
!%      - multi_polynomial_get_control_size
!%      - sigmoide
!%      - inv_sigmoide
!%      - scaled_sigmoide
!%      - inv_scaled_sigmoid
!%      - sigmoide2d
!%      - scaled_sigmoide2d
!%      - sbs_control_tfm
!%      - sbs_inv_control_tfm
!%      - normalize_control_tfm
!%      - normalize_inv_control_tfm
!%      - uniform_parameters_to_control
!%      - uniform_control_to_parameters
!%      - distributed_parameters_to_control
!%      - distributed_control_to_parameters
!%      - multi_linear_parameters_to_control
!%      - multi_linear_control_to_parameters
!%      - multi_polynomial_parameters_to_control
!%      - multi_polynomial_control_to_parameters
!%      - control_tfm
!%      - inv_control_tfm
!%      - parameters_to_control
!%      - control_to_parameters

module mwd_parameters_manipulation

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_options !% only: OptionsDT
    use mwd_control !% only: ControlDT_initialise, ControlDT_finalise

    implicit none

    public :: parameters_to_control

contains

    subroutine uniform_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%opr_parameters) + sum(options%optimize%opr_initial_states)

    end subroutine uniform_get_control_size

    subroutine distributed_get_control_size(setup, mesh, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = (sum(options%optimize%opr_parameters) + &
        & sum(options%optimize%opr_initial_states))*mesh%nac

    end subroutine distributed_get_control_size

    subroutine multi_linear_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            n = n + 1 + sum(options%optimize%opr_parameters_descriptor(:, i))

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            n = n + 1 + sum(options%optimize%opr_initial_states_descriptor(:, i))

        end do

    end subroutine multi_linear_get_control_size

    subroutine multi_polynomial_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            n = n + 1 + 2*sum(options%optimize%opr_parameters_descriptor(:, i))

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            n = n + 1 + 2*sum(options%optimize%opr_initial_states_descriptor(:, i))

        end do

    end subroutine multi_polynomial_get_control_size

    subroutine sigmoide(x, res)

        implicit none

        real(sp), intent(in) :: x
        real(sp), intent(inout) :: res

        res = 1._sp/(1._sp + exp(-x))

    end subroutine sigmoide

    subroutine inv_sigmoide(x, res)

        implicit none

        real(sp), intent(in) :: x
        real(sp), intent(inout) :: res

        res = log(x/(1._sp - x))

    end subroutine inv_sigmoide

    subroutine scaled_sigmoide(x, l, u, res)

        implicit none

        real(sp), intent(in) :: x, l, u
        real(sp), intent(inout) :: res

        call sigmoide(x, res)

        res = res*(u - l) + l

    end subroutine scaled_sigmoide

    subroutine inv_scaled_sigmoid(x, l, u, res)

        implicit none

        real(sp), intent(in) :: x, l, u
        real(sp), intent(inout) :: res

        real(sp) :: xw, eps = 1e-3_sp

        xw = max(x, l + eps)
        xw = min(x, u - eps)
        xw = (xw - l)/(u - l)

        call inv_sigmoide(xw, res)

    end subroutine inv_scaled_sigmoid

    subroutine sigmoide2d(x, res)

        implicit none

        real(sp), dimension(:, :), intent(in) :: x
        real(sp), dimension(:, :), intent(inout) :: res

        res = 1._sp/(1._sp + exp(-x))

    end subroutine sigmoide2d

    subroutine scaled_sigmoide2d(x, l, u, res)

        implicit none

        real(sp), dimension(:, :), intent(in) :: x
        real(sp), intent(in) :: l, u
        real(sp), dimension(:, :), intent(inout) :: res

        call sigmoide2d(x, res)

        res = res*(u - l) + l

    end subroutine scaled_sigmoide2d

    subroutine sbs_control_tfm(parameters)

        implicit none

        type(ParametersDT), intent(inout) :: parameters

        integer :: i
        logical, dimension(size(parameters%control%x)) :: nbd_mask

        !% Need lower and upper bound to sbs tfm
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        do i = 1, size(parameters%control%x)

            if (.not. nbd_mask(i)) cycle

            if (parameters%control%l_bkg(i) .lt. 0._sp) then

                parameters%control%x(i) = asinh(parameters%control%x(i))
                parameters%control%l(i) = asinh(parameters%control%l_bkg(i))
                parameters%control%u(i) = asinh(parameters%control%u_bkg(i))

            else if (parameters%control%l_bkg(i) .ge. 0._sp .and. parameters%control%u_bkg(i) .le. 1._sp) then

                parameters%control%x(i) = log(parameters%control%x(i)/(1._sp - parameters%control%x(i)))
                parameters%control%l(i) = log(parameters%control%l_bkg(i)/(1._sp - parameters%control%l_bkg(i)))
                parameters%control%u(i) = log(parameters%control%u_bkg(i)/(1._sp - parameters%control%u_bkg(i)))

            else

                parameters%control%x(i) = log(parameters%control%x(i))
                parameters%control%l(i) = log(parameters%control%l_bkg(i))
                parameters%control%u(i) = log(parameters%control%u_bkg(i))

            end if

        end do

    end subroutine sbs_control_tfm

    subroutine sbs_inv_control_tfm(parameters)

        implicit none

        type(ParametersDT), intent(inout) :: parameters

        integer :: i
        logical, dimension(size(parameters%control%x)) :: nbd_mask

        !% Need lower and upper bound to sbs tfm
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        do i = 1, size(parameters%control%x)

            if (.not. nbd_mask(i)) cycle

            if (parameters%control%l_bkg(i) .lt. 0._sp) then

                parameters%control%x(i) = sinh(parameters%control%x(i))

            else if (parameters%control%l_bkg(i) .ge. 0._sp .and. parameters%control%u_bkg(i) .le. 1._sp) then

                parameters%control%x(i) = exp(parameters%control%x(i))/(1._sp + exp(parameters%control%x(i)))

            else

                parameters%control%x(i) = exp(parameters%control%x(i))

            end if

        end do

        parameters%control%l = parameters%control%l_bkg
        parameters%control%u = parameters%control%u_bkg

    end subroutine sbs_inv_control_tfm

    subroutine normalize_control_tfm(parameters)

        implicit none

        type(ParametersDT), intent(inout) :: parameters

        logical, dimension(size(parameters%control%x)) :: nbd_mask

        !% Need lower and upper bound to normalize
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        where (nbd_mask)

            parameters%control%x = (parameters%control%x - parameters%control%l_bkg)/ &
                                   (parameters%control%u_bkg - parameters%control%l_bkg)
            parameters%control%l = 0._sp
            parameters%control%u = 1._sp

        end where

    end subroutine normalize_control_tfm

    subroutine normalize_inv_control_tfm(parameters)

        implicit none

        type(ParametersDT), intent(inout) :: parameters

        logical, dimension(size(parameters%control%x)) :: nbd_mask

        !% Need lower and upper bound to denormalize
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        where (nbd_mask)

            parameters%control%x = parameters%control%x* &
            & (parameters%control%u_bkg - parameters%control%l_bkg) + parameters%control%l_bkg
            parameters%control%l = parameters%control%l_bkg
            parameters%control%u = parameters%control%u_bkg

        end where

    end subroutine normalize_inv_control_tfm

    subroutine uniform_parameters_to_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: n, i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        call uniform_get_control_size(setup, options, n)

        call ControlDT_initialise(parameters%control, n)

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            parameters%control%x(j) = sum(parameters%opr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            parameters%control%l(j) = options%optimize%l_opr_parameters(i)
            parameters%control%u(j) = options%optimize%u_opr_parameters(i)

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            parameters%control%x(j) = sum(parameters%opr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            parameters%control%l(j) = options%optimize%l_opr_initial_states(i)
            parameters%control%u(j) = options%optimize%u_opr_initial_states(i)

        end do

!~         parameters%control%x_bkg = parameters%control%x
        parameters%control%l_bkg = parameters%control%l
        parameters%control%u_bkg = parameters%control%u
        parameters%control%nbd = 2

    end subroutine uniform_parameters_to_control

    subroutine uniform_control_to_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            where (ac_mask)

                parameters%opr_parameters%values(:, :, i) = parameters%control%x(j)

            end where

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            where (ac_mask)

                parameters%opr_initial_states%values(:, :, i) = parameters%control%x(j)

            end where

        end do

    end subroutine uniform_control_to_parameters

    subroutine distributed_parameters_to_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: n, i, j, row, col

        call distributed_get_control_size(setup, mesh, options, n)

        call ControlDT_initialise(parameters%control, n)

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%control%x(j) = parameters%opr_parameters%values(row, col, i)
                    parameters%control%l(j) = options%optimize%l_opr_parameters(i)
                    parameters%control%u(j) = options%optimize%u_opr_parameters(i)

                end do

            end do

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%control%x(j) = parameters%opr_initial_states%values(row, col, i)
                    parameters%control%l(j) = options%optimize%l_opr_initial_states(i)
                    parameters%control%u(j) = options%optimize%u_opr_initial_states(i)

                end do

            end do

        end do

        !~         parameters%control%x_bkg = parameters%control%x
        parameters%control%l_bkg = parameters%control%l
        parameters%control%u_bkg = parameters%control%u
        parameters%control%nbd = 2

    end subroutine distributed_parameters_to_control

    subroutine distributed_control_to_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, row, col

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%opr_parameters%values(row, col, i) = parameters%control%x(j)

                end do

            end do

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%opr_initial_states%values(row, col, i) = parameters%control%x(j)

                end do

            end do

        end do

    end subroutine distributed_control_to_parameters

    subroutine multi_linear_parameters_to_control(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: n, i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        call multi_linear_get_control_size(setup, options, n)
        call ControlDT_initialise(parameters%control, n)

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            y = sum(parameters%opr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_opr_parameters(i)
            u = options%optimize%u_opr_parameters(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))

            do k = 1, setup%nd

                if (options%optimize%opr_parameters_descriptor(k, i) .ne. 1) cycle

                j = j + 1

                parameters%control%x(j) = 0._sp

            end do

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            y = sum(parameters%opr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_opr_initial_states(i)
            u = options%optimize%u_opr_initial_states(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))

            do k = 1, setup%nd

                if (options%optimize%opr_initial_states_descriptor(k, i) .ne. 1) cycle

                j = j + 1

                parameters%control%x(j) = 0._sp

            end do

        end do

        !~         parameters%control%x_bkg = parameters%control%x
        parameters%control%l_bkg = parameters%control%l
        parameters%control%u_bkg = parameters%control%u
        parameters%control%nbd = 0

    end subroutine multi_linear_parameters_to_control

    subroutine multi_linear_control_to_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%opr_parameters_descriptor(k, i) .ne. 1) cycle

                j = j + 1

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                wa2d = wa2d + parameters%control%x(j)*norm_desc

            end do

            l = options%optimize%l_opr_parameters(i)
            u = options%optimize%u_opr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%opr_parameters%values(:, :, i))

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%opr_initial_states_descriptor(k, i) .ne. 1) cycle

                j = j + 1

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                wa2d = wa2d + parameters%control%x(j)*norm_desc

            end do

            l = options%optimize%l_opr_initial_states(i)
            u = options%optimize%u_opr_initial_states(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%opr_initial_states%values(:, :, i))

        end do

    end subroutine multi_linear_control_to_parameters

    subroutine multi_polynomial_parameters_to_control(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: n, i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        call multi_polynomial_get_control_size(setup, options, n)
        call ControlDT_initialise(parameters%control, n)

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            y = sum(parameters%opr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_opr_parameters(i)
            u = options%optimize%u_opr_parameters(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0

            do k = 1, setup%nd

                if (options%optimize%opr_parameters_descriptor(k, i) .ne. 1) cycle

                j = j + 2

                parameters%control%x(j - 1) = 0._sp
                parameters%control%nbd(j - 1) = 0

                parameters%control%x(j) = 1._sp
                parameters%control%l(j) = 0.5_sp
                parameters%control%u(j) = 2._sp
                parameters%control%nbd(j) = 2

            end do

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            y = sum(parameters%opr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_opr_initial_states(i)
            u = options%optimize%u_opr_initial_states(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0

            do k = 1, setup%nd

                if (options%optimize%opr_initial_states_descriptor(k, i) .ne. 1) cycle

                j = j + 2

                parameters%control%x(j - 1) = 0._sp
                parameters%control%nbd(j - 1) = 0

                parameters%control%x(j) = 1._sp
                parameters%control%l(j) = 0.5_sp
                parameters%control%u(j) = 2._sp
                parameters%control%nbd(j) = 2

            end do

        end do

        !~         parameters%control%x_bkg = parameters%control%x
        parameters%control%l_bkg = parameters%control%l
        parameters%control%u_bkg = parameters%control%u

    end subroutine multi_polynomial_parameters_to_control

    subroutine multi_polynomial_control_to_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        j = 0

        do i = 1, setup%nop

            if (options%optimize%opr_parameters(i) .ne. 1) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%opr_parameters_descriptor(k, i) .ne. 1) cycle

                j = j + 2

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                norm_desc = norm_desc**parameters%control%x(j)

                wa2d = wa2d + parameters%control%x(j - 1)*norm_desc

            end do

            l = options%optimize%l_opr_parameters(i)
            u = options%optimize%u_opr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%opr_parameters%values(:, :, i))

        end do

        do i = 1, setup%nos

            if (options%optimize%opr_initial_states(i) .ne. 1) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%opr_initial_states_descriptor(k, i) .ne. 1) cycle

                j = j + 2

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                norm_desc = norm_desc**parameters%control%x(j)

                wa2d = wa2d + parameters%control%x(j - 1)*norm_desc

            end do

            l = options%optimize%l_opr_parameters(i)
            u = options%optimize%u_opr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%opr_initial_states%values(:, :, i))

        end do

    end subroutine multi_polynomial_control_to_parameters

    subroutine control_tfm(parameters, options)

        implicit none

        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        select case (options%optimize%control_tfm)

        case ("sbs")

            call sbs_control_tfm(parameters)

        case ("normalize")

            call normalize_control_tfm(parameters)

        end select

    end subroutine control_tfm

    subroutine inv_control_tfm(parameters, options)

        implicit none

        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        select case (options%optimize%control_tfm)

        case ("sbs")

            call sbs_inv_control_tfm(parameters)

        case ("normalize")

            call normalize_inv_control_tfm(parameters)

        end select

    end subroutine inv_control_tfm

    subroutine parameters_to_control(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        select case (options%optimize%mapping)

        case ("uniform")

            call uniform_parameters_to_control(setup, mesh, parameters, options)

        case ("distributed")

            call distributed_parameters_to_control(setup, mesh, parameters, options)

        case ("multi-linear")

            call multi_linear_parameters_to_control(setup, mesh, input_data, parameters, options)

        case ("multi-polynomial")

            call multi_polynomial_parameters_to_control(setup, mesh, input_data, parameters, options)

        end select

        call control_tfm(parameters, options)

    end subroutine parameters_to_control

    subroutine control_to_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        if (.not. allocated(parameters%control%x)) return

        call inv_control_tfm(parameters, options)

        select case (options%optimize%mapping)

        case ("uniform")

            call uniform_control_to_parameters(setup, mesh, parameters, options)

        case ("distributed")

            call distributed_control_to_parameters(setup, mesh, parameters, options)

        case ("multi-linear")

            call multi_linear_control_to_parameters(setup, mesh, input_data, parameters, options)

        case ("multi-polynomial")

            call multi_polynomial_control_to_parameters(setup, mesh, input_data, parameters, options)

        end select

    end subroutine control_to_parameters

end module mwd_parameters_manipulation
