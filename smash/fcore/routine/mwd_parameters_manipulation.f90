!%      (MWD) Module Wrapped and Differentiated
!%
!%      Subroutine
!%      ----------
!%
!%      - get_serr_mu
!%      - get_serr_sigma
!%      - get_rr_parameters
!%      - get_rr_states
!%      - get_serr_mu_parameters
!%      - get_serr_sigma_parameters
!%      - set_rr_parameters
!%      - set_rr_states
!%      - set_serr_mu_parameters
!%      - set_serr_sigma_parameters
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
!%      - control_tfm
!%      - inv_control_tfm
!%      - uniform_rr_parameters_get_control_size
!%      - uniform_rr_initial_states_get_control_size
!%      - distributed_rr_parameters_get_control_size
!%      - distributed_rr_initial_states_get_control_size
!%      - multi_linear_rr_parameters_get_control_size
!%      - multi_linear_rr_initial_states_get_control_size
!%      - multi_polynomial_rr_parameters_get_control_size
!%      - multi_polynomial_rr_initial_states_get_control_size
!%      - serr_mu_parameters_get_control_size
!%      - get_control_sizes
!%      - uniform_rr_parameters_fill_control
!%      - uniform_rr_initial_states_fill_control
!%      - distributed_rr_parameters_fill_control
!%      - distributed_rr_initial_states_fill_control
!%      - multi_linear_rr_parameters_fill_control
!%      - multi_linear_rr_initial_states_fill_control
!%      - multi_polynomial_rr_parameters_fill_control
!%      - multi_polynomial_rr_initial_states_fill_control
!%      - serr_mu_parameters_fill_control
!%      - serr_sigma_parameters_fill_control
!%      - fill_control
!%      - uniform_rr_parameters_fill_parameters
!%      - uniform_rr_initial_states_fill_parameters
!%      - distributed_rr_parameters_fill_parameters
!%      - distributed_rr_initial_states_fill_parameters
!%      - multi_linear_rr_parameters_fill_parameters
!%      - multi_linear_rr_initial_states_fill_parameters
!%      - multi_polynomial_rr_parameters_fill_parameters
!%      - multi_polynomial_rr_initial_states_fill_parameters
!%      - serr_mu_parameters_fill_parameters
!%      - serr_sigma_parameters_fill_parameters
!%      - fill_parameters

module mwd_parameters_manipulation

    use mwd_bayesian_tools !% only: MuFunk_vect, SigmaFunk_vect
    use md_constant !% only: sp, dp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_rr_parameters !% only: RR_ParametersDT
    use mwd_rr_states !% only: RR_StatesDT
    use mwd_serr_mu_parameters !% only: SErr_Mu_ParametersDT
    use mwd_serr_sigma_parameters !% only: SErr_Sigma_ParametersDT
    use mwd_output !% only: OutputDT
    use mwd_options !% only: OptionsDT
    use mwd_returns !% only: ReturnsDT
    use mwd_control !% only: ControlDT_initialise, ControlDT_finalise

    implicit none

contains

    !$AD start-exclude
    subroutine get_serr_mu(setup, mesh, parameters, output, serr_mu)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(in) :: output
        real(sp), dimension(mesh%ng, setup%ntime_step), intent(inout) :: serr_mu

        character(lchar) :: funk
        real(dp), dimension(setup%nsep_mu, mesh%ng) :: par
        real(dp), dimension(setup%ntime_step, mesh%ng) :: y, mu

        funk = setup%serr_mu_mapping
        par = real(transpose(parameters%serr_mu_parameters%values), dp)
        y = real(transpose(output%response%q), dp)

        call MuFunk_vect(funk, par, y, mu)

        serr_mu = real(transpose(mu), sp)

    end subroutine get_serr_mu
    !$AD end-exclude

    !$AD start-exclude
    subroutine get_serr_sigma(setup, mesh, parameters, output, serr_sigma)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        type(OutputDT), intent(in) :: output
        real(sp), dimension(mesh%ng, setup%ntime_step), intent(inout) :: serr_sigma

        character(lchar) :: funk
        real(dp), dimension(setup%nsep_sigma, mesh%ng) :: par
        real(dp), dimension(setup%ntime_step, mesh%ng) :: y, sigma

        funk = setup%serr_sigma_mapping
        par = real(transpose(parameters%serr_sigma_parameters%values), dp)
        y = real(transpose(output%response%q), dp)

        call SigmaFunk_vect(funk, par, y, sigma)

        serr_sigma = real(transpose(sigma), sp)

    end subroutine get_serr_sigma
    !$AD end-exclude

    subroutine get_rr_parameters(rr_parameters, key, vle)

        implicit none

        type(RR_ParametersDT), intent(in) :: rr_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:, :), intent(inout) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(rr_parameters%keys)

            if (trim(rr_parameters%keys(i)) .eq. key) then

                vle = rr_parameters%values(:, :, i)
                return

            end if

        end do

        ! Should be unreachable

    end subroutine get_rr_parameters

    subroutine get_rr_states(rr_states, key, vle)

        implicit none

        type(RR_StatesDT), intent(in) :: rr_states
        character(*), intent(in) :: key
        real(sp), dimension(:, :), intent(inout) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(rr_states%keys)

            if (trim(rr_states%keys(i)) .eq. key) then

                vle = rr_states%values(:, :, i)
                return

            end if

        end do

        ! Should be unreachable

    end subroutine get_rr_states

    subroutine get_serr_mu_parameters(serr_mu_parameters, key, vle)

        implicit none

        type(SErr_Mu_ParametersDT), intent(in) :: serr_mu_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:), intent(inout) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(serr_mu_parameters%keys)

            if (trim(serr_mu_parameters%keys(i)) .eq. key) then

                vle = serr_mu_parameters%values(:, i)
                return

            end if

        end do

        ! Should be unreachable

    end subroutine get_serr_mu_parameters

    subroutine get_serr_sigma_parameters(serr_sigma_parameters, key, vle)

        implicit none

        type(SErr_Sigma_ParametersDT), intent(in) :: serr_sigma_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:), intent(inout) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(serr_sigma_parameters%keys)

            if (trim(serr_sigma_parameters%keys(i)) .eq. key) then

                vle = serr_sigma_parameters%values(:, i)
                return

            end if

        end do

        ! Should be unreachable

    end subroutine get_serr_sigma_parameters

    subroutine set_rr_parameters(rr_parameters, key, vle)

        implicit none

        type(RR_ParametersDT), intent(inout) :: rr_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:, :), intent(in) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(rr_parameters%keys)

            if (trim(rr_parameters%keys(i)) .eq. key) then

                rr_parameters%values(:, :, i) = vle
                return

            end if

        end do

        ! Should be unreachable

    end subroutine set_rr_parameters

    subroutine set_rr_states(rr_states, key, vle)

        implicit none

        type(RR_StatesDT), intent(inout) :: rr_states
        character(*), intent(in) :: key
        real(sp), dimension(:, :), intent(in) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(rr_states%keys)

            if (trim(rr_states%keys(i)) .eq. key) then

                rr_states%values(:, :, i) = vle
                return

            end if

        end do

        ! Should be unreachable

    end subroutine set_rr_states

    subroutine set_serr_mu_parameters(serr_mu_parameters, key, vle)

        implicit none

        type(SErr_Mu_ParametersDT), intent(inout) :: serr_mu_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:), intent(in) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(serr_mu_parameters%keys)

            if (trim(serr_mu_parameters%keys(i)) .eq. key) then

                serr_mu_parameters%values(:, i) = vle
                return

            end if

        end do

        ! Should be unreachable

    end subroutine set_serr_mu_parameters

    subroutine set_serr_sigma_parameters(serr_sigma_parameters, key, vle)

        implicit none

        type(SErr_Sigma_ParametersDT), intent(inout) :: serr_sigma_parameters
        character(*), intent(in) :: key
        real(sp), dimension(:), intent(in) :: vle

        integer :: i

        ! Linear search on keys
        do i = 1, size(serr_sigma_parameters%keys)

            if (trim(serr_sigma_parameters%keys(i)) .eq. key) then

                serr_sigma_parameters%values(:, i) = vle
                return

            end if

        end do

        ! Should be unreachable

    end subroutine set_serr_sigma_parameters

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
        logical, dimension(parameters%control%n) :: nbd_mask

        !% Need lower and upper bound to sbs tfm
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        ! Only apply sbs transformation on RR parameters and RR initial states
        do i = 1, sum(parameters%control%nbk(1:2))

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
        logical, dimension(parameters%control%n) :: nbd_mask

        !% Need lower and upper bound to sbs tfm
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        ! Only apply sbs inv transformation on RR parameters et RR initial states
        do i = 1, sum(parameters%control%nbk(1:2))

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

        logical, dimension(parameters%control%n) :: nbd_mask

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

        logical, dimension(parameters%control%n) :: nbd_mask

        !% Need lower and upper bound to denormalize
        nbd_mask = (parameters%control%nbd(:) .eq. 2)

        where (nbd_mask)

            parameters%control%x = parameters%control%x* &
            & (parameters%control%u_bkg - parameters%control%l_bkg) + parameters%control%l_bkg
            parameters%control%l = parameters%control%l_bkg
            parameters%control%u = parameters%control%u_bkg

        end where

    end subroutine normalize_inv_control_tfm

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

    subroutine uniform_rr_parameters_get_control_size(options, n)

        implicit none

        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%rr_parameters)

    end subroutine uniform_rr_parameters_get_control_size

    subroutine uniform_rr_initial_states_get_control_size(options, n)

        implicit none

        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%rr_initial_states)

    end subroutine uniform_rr_initial_states_get_control_size

    subroutine distributed_rr_parameters_get_control_size(mesh, options, n)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%rr_parameters)*mesh%nac

    end subroutine distributed_rr_parameters_get_control_size

    subroutine distributed_rr_initial_states_get_control_size(mesh, options, n)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%rr_initial_states)*mesh%nac

    end subroutine distributed_rr_initial_states_get_control_size

    subroutine multi_linear_rr_parameters_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            n = n + 1 + sum(options%optimize%rr_parameters_descriptor(:, i))

        end do

    end subroutine multi_linear_rr_parameters_get_control_size

    subroutine multi_linear_rr_initial_states_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            n = n + 1 + sum(options%optimize%rr_initial_states_descriptor(:, i))

        end do

    end subroutine multi_linear_rr_initial_states_get_control_size

    subroutine multi_polynomial_rr_parameters_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            n = n + 1 + 2*sum(options%optimize%rr_parameters_descriptor(:, i))

        end do

    end subroutine multi_polynomial_rr_parameters_get_control_size

    subroutine multi_polynomial_rr_initial_states_get_control_size(setup, options, n)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        integer :: i

        n = 0

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            n = n + 1 + 2*sum(options%optimize%rr_initial_states_descriptor(:, i))

        end do

    end subroutine multi_polynomial_rr_initial_states_get_control_size

    subroutine serr_mu_parameters_get_control_size(options, n)

        implicit none

        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%serr_mu_parameters)*options%cost%nog

    end subroutine serr_mu_parameters_get_control_size

    subroutine serr_sigma_parameters_get_control_size(options, n)

        implicit none

        type(OptionsDT), intent(in) :: options
        integer, intent(inout) :: n

        n = sum(options%optimize%serr_sigma_parameters)*options%cost%nog

    end subroutine serr_sigma_parameters_get_control_size

    subroutine get_control_sizes(setup, mesh, options, nbk)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(OptionsDT), intent(in) :: options
        integer, dimension(:), intent(out) :: nbk

        select case (options%optimize%mapping)

        case ("uniform")

            call uniform_rr_parameters_get_control_size(options, nbk(1))
            call uniform_rr_initial_states_get_control_size(options, nbk(2))

        case ("distributed")

            call distributed_rr_parameters_get_control_size(mesh, options, nbk(1))
            call distributed_rr_initial_states_get_control_size(mesh, options, nbk(2))

        case ("multi-linear")

            call multi_linear_rr_parameters_get_control_size(setup, options, nbk(1))
            call multi_linear_rr_initial_states_get_control_size(setup, options, nbk(2))

        case ("multi-polynomial")

            call multi_polynomial_rr_parameters_get_control_size(setup, options, nbk(1))
            call multi_polynomial_rr_initial_states_get_control_size(setup, options, nbk(2))

        end select

        ! Directly working with hyper parameters
        call serr_mu_parameters_get_control_size(options, nbk(3))
        call serr_sigma_parameters_get_control_size(options, nbk(4))

    end subroutine get_control_sizes

    subroutine uniform_rr_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            parameters%control%x(j) = sum(parameters%rr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            parameters%control%l(j) = options%optimize%l_rr_parameters(i)
            parameters%control%u(j) = options%optimize%u_rr_parameters(i)
            parameters%control%nbd(j) = 2
            parameters%control%name(j) = trim(parameters%rr_parameters%keys(i))//"0"

        end do

    end subroutine uniform_rr_parameters_fill_control

    subroutine uniform_rr_initial_states_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            parameters%control%x(j) = sum(parameters%rr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            parameters%control%l(j) = options%optimize%l_rr_initial_states(i)
            parameters%control%u(j) = options%optimize%u_rr_initial_states(i)
            parameters%control%nbd(j) = 2
            parameters%control%name(j) = trim(parameters%rr_initial_states%keys(i))//"0"

        end do

    end subroutine uniform_rr_initial_states_fill_control

    subroutine distributed_rr_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        character(lchar) :: name
        integer :: i, j, row, col

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%control%x(j) = parameters%rr_parameters%values(row, col, i)
                    parameters%control%l(j) = options%optimize%l_rr_parameters(i)
                    parameters%control%u(j) = options%optimize%u_rr_parameters(i)
                    parameters%control%nbd(j) = 2
                    write (name, '(a,i0,a,i0)') trim(parameters%rr_parameters%keys(i)), row, "-", col
                    parameters%control%name(j) = name

                end do

            end do

        end do

    end subroutine distributed_rr_parameters_fill_control

    subroutine distributed_rr_initial_states_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        character(lchar) :: name
        integer :: i, j, row, col

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%control%x(j) = parameters%rr_initial_states%values(row, col, i)
                    parameters%control%l(j) = options%optimize%l_rr_initial_states(i)
                    parameters%control%u(j) = options%optimize%u_rr_initial_states(i)
                    parameters%control%nbd(j) = 2
                    write (name, '(a,i0,a,i0)') trim(parameters%rr_initial_states%keys(i)), row, "-", col
                    parameters%control%name(j) = name

                end do

            end do

        end do

    end subroutine distributed_rr_initial_states_fill_control

    subroutine multi_linear_rr_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            y = sum(parameters%rr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_rr_parameters(i)
            u = options%optimize%u_rr_parameters(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0
            parameters%control%name(j) = trim(parameters%rr_parameters%keys(i))//"0"

            do k = 1, setup%nd

                if (options%optimize%rr_parameters_descriptor(k, i) .eq. 0) cycle

                j = j + 1

                parameters%control%x(j) = 0._sp
                parameters%control%nbd(j) = 0
                parameters%control%name(j) = trim(parameters%rr_parameters%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-a"

            end do

        end do

    end subroutine multi_linear_rr_parameters_fill_control

    subroutine multi_linear_rr_initial_states_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            y = sum(parameters%rr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_rr_initial_states(i)
            u = options%optimize%u_rr_initial_states(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0
            parameters%control%name(j) = trim(parameters%rr_initial_states%keys(i))//"0"

            do k = 1, setup%nd

                if (options%optimize%rr_initial_states_descriptor(k, i) .eq. 0) cycle

                j = j + 1

                parameters%control%x(j) = 0._sp
                parameters%control%nbd(j) = 0
                parameters%control%name(j) = trim(parameters%rr_initial_states%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-a"

            end do

        end do

    end subroutine multi_linear_rr_initial_states_fill_control

    subroutine multi_polynomial_rr_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            y = sum(parameters%rr_parameters%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_rr_parameters(i)
            u = options%optimize%u_rr_parameters(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0
            parameters%control%name(j) = trim(parameters%rr_parameters%keys(i))//"0"

            do k = 1, setup%nd

                if (options%optimize%rr_parameters_descriptor(k, i) .eq. 0) cycle

                j = j + 2

                parameters%control%x(j - 1) = 0._sp
                parameters%control%nbd(j - 1) = 0
                parameters%control%name(j - 1) = trim(parameters%rr_parameters%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-a"

                parameters%control%x(j) = 1._sp
                parameters%control%l(j) = 0.5_sp
                parameters%control%u(j) = 2._sp
                parameters%control%nbd(j) = 2
                parameters%control%name(j) = trim(parameters%rr_parameters%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-b"

            end do

        end do

    end subroutine multi_polynomial_rr_parameters_fill_control

    subroutine multi_polynomial_rr_initial_states_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: y, l, u
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            y = sum(parameters%rr_initial_states%values(:, :, i), mask=ac_mask)/mesh%nac
            l = options%optimize%l_rr_initial_states(i)
            u = options%optimize%u_rr_initial_states(i)

            call inv_scaled_sigmoid(y, l, u, parameters%control%x(j))
            parameters%control%nbd(j) = 0
            parameters%control%name(j) = trim(parameters%rr_initial_states%keys(i))//"0"

            do k = 1, setup%nd

                if (options%optimize%rr_initial_states_descriptor(k, i) .eq. 0) cycle

                j = j + 2

                parameters%control%x(j - 1) = 0._sp
                parameters%control%nbd(j - 1) = 0
                parameters%control%name(j - 1) = trim(parameters%rr_initial_states%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-a"

                parameters%control%x(j) = 1._sp
                parameters%control%l(j) = 0.5_sp
                parameters%control%u(j) = 2._sp
                parameters%control%nbd(j) = 2
                parameters%control%name(j) = trim(parameters%rr_initial_states%keys(i))// &
                & "-"//trim(setup%descriptor_name(k))//"-b"

            end do

        end do

    end subroutine multi_polynomial_rr_initial_states_fill_control

    subroutine serr_mu_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k

        ! SErr mu parameters is third control kind
        j = sum(parameters%control%nbk(1:2))

        do i = 1, setup%nsep_mu

            if (options%optimize%serr_mu_parameters(i) .eq. 0) cycle

            do k = 1, mesh%ng

                if (options%cost%gauge(k) .eq. 0) cycle

                j = j + 1

                parameters%control%x(j) = parameters%serr_mu_parameters%values(k, i)
                parameters%control%l(j) = options%optimize%l_serr_mu_parameters(i)
                parameters%control%u(j) = options%optimize%u_serr_mu_parameters(i)
                parameters%control%nbd(j) = 2
                parameters%control%name(j) = trim(parameters%serr_mu_parameters%keys(i))//"-"//trim(mesh%code(k))

            end do

        end do

    end subroutine serr_mu_parameters_fill_control

    subroutine serr_sigma_parameters_fill_control(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k

        ! SErr sigma parameters is fourth control kind
        j = sum(parameters%control%nbk(1:3))

        do i = 1, setup%nsep_sigma

            if (options%optimize%serr_sigma_parameters(i) .eq. 0) cycle

            do k = 1, mesh%ng

                if (options%cost%gauge(k) .eq. 0) cycle

                j = j + 1

                parameters%control%x(j) = parameters%serr_sigma_parameters%values(k, i)
                parameters%control%l(j) = options%optimize%l_serr_sigma_parameters(i)
                parameters%control%u(j) = options%optimize%u_serr_sigma_parameters(i)
                parameters%control%nbd(j) = 2
                parameters%control%name(j) = trim(parameters%serr_sigma_parameters%keys(i))//"-"//trim(mesh%code(k))

            end do

        end do

    end subroutine serr_sigma_parameters_fill_control

    subroutine fill_control(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        select case (options%optimize%mapping)

        case ("uniform")

            call uniform_rr_parameters_fill_control(setup, mesh, parameters, options)
            call uniform_rr_initial_states_fill_control(setup, mesh, parameters, options)

        case ("distributed")

            call distributed_rr_parameters_fill_control(setup, mesh, parameters, options)
            call distributed_rr_initial_states_fill_control(setup, mesh, parameters, options)

        case ("multi-linear")

            call multi_linear_rr_parameters_fill_control(setup, mesh, parameters, options)
            call multi_linear_rr_initial_states_fill_control(setup, mesh, parameters, options)

        case ("multi-polynomial")

            call multi_polynomial_rr_parameters_fill_control(setup, mesh, parameters, options)
            call multi_polynomial_rr_initial_states_fill_control(setup, mesh, parameters, options)

        end select

        ! Directly working with hyper parameters
        call serr_mu_parameters_fill_control(setup, mesh, parameters, options)
        call serr_sigma_parameters_fill_control(setup, mesh, parameters, options)

        ! Store background
        parameters%control%x_bkg = parameters%control%x
        parameters%control%l_bkg = parameters%control%l
        parameters%control%u_bkg = parameters%control%u

    end subroutine fill_control

    subroutine uniform_rr_parameters_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            where (ac_mask)

                parameters%rr_parameters%values(:, :, i) = parameters%control%x(j)

            end where

        end do

    end subroutine uniform_rr_parameters_fill_parameters

    subroutine uniform_rr_initial_states_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j
        logical, dimension(mesh%nrow, mesh%ncol) :: ac_mask

        ac_mask = (mesh%active_cell(:, :) .eq. 1)

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            where (ac_mask)

                parameters%rr_initial_states%values(:, :, i) = parameters%control%x(j)

            end where

        end do

    end subroutine uniform_rr_initial_states_fill_parameters

    subroutine distributed_rr_parameters_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, row, col

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%rr_parameters%values(row, col, i) = parameters%control%x(j)

                end do

            end do

        end do

    end subroutine distributed_rr_parameters_fill_parameters

    subroutine distributed_rr_initial_states_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, row, col

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            do col = 1, mesh%ncol

                do row = 1, mesh%nrow

                    if (mesh%active_cell(row, col) .eq. 0) cycle

                    j = j + 1

                    parameters%rr_initial_states%values(row, col, i) = parameters%control%x(j)

                end do

            end do

        end do

    end subroutine distributed_rr_initial_states_fill_parameters

    subroutine multi_linear_rr_parameters_fill_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%rr_parameters_descriptor(k, i) .eq. 0) cycle

                j = j + 1

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                wa2d = wa2d + parameters%control%x(j)*norm_desc

            end do

            l = options%optimize%l_rr_parameters(i)
            u = options%optimize%u_rr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%rr_parameters%values(:, :, i))

        end do

    end subroutine multi_linear_rr_parameters_fill_parameters

    subroutine multi_linear_rr_initial_states_fill_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%rr_initial_states_descriptor(k, i) .eq. 0) cycle

                j = j + 1

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                wa2d = wa2d + parameters%control%x(j)*norm_desc

            end do

            l = options%optimize%l_rr_initial_states(i)
            u = options%optimize%u_rr_initial_states(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%rr_initial_states%values(:, :, i))

        end do

    end subroutine multi_linear_rr_initial_states_fill_parameters

    subroutine multi_polynomial_rr_parameters_fill_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        ! RR parameters is first control kind
        j = 0

        do i = 1, setup%nrrp

            if (options%optimize%rr_parameters(i) .eq. 0) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%rr_parameters_descriptor(k, i) .eq. 0) cycle

                j = j + 2

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                norm_desc = norm_desc**parameters%control%x(j)

                wa2d = wa2d + parameters%control%x(j - 1)*norm_desc

            end do

            l = options%optimize%l_rr_parameters(i)
            u = options%optimize%u_rr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%rr_parameters%values(:, :, i))

        end do

    end subroutine multi_polynomial_rr_parameters_fill_parameters

    subroutine multi_polynomial_rr_initial_states_fill_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k
        real(sp) :: l, u
        real(sp), dimension(mesh%nrow, mesh%ncol) :: wa2d, norm_desc

        ! RR initial states is second control kind
        j = parameters%control%nbk(1)

        do i = 1, setup%nrrs

            if (options%optimize%rr_initial_states(i) .eq. 0) cycle

            j = j + 1

            wa2d = parameters%control%x(j)

            do k = 1, setup%nd

                if (options%optimize%rr_initial_states_descriptor(k, i) .eq. 0) cycle

                j = j + 2

                norm_desc = (input_data%physio_data%descriptor(:, :, k) - input_data%physio_data%l_descriptor(k))/ &
                & (input_data%physio_data%u_descriptor(k) - input_data%physio_data%l_descriptor(k))

                norm_desc = norm_desc**parameters%control%x(j)

                wa2d = wa2d + parameters%control%x(j - 1)*norm_desc

            end do

            l = options%optimize%l_rr_parameters(i)
            u = options%optimize%u_rr_parameters(i)

            call scaled_sigmoide2d(wa2d, l, u, parameters%rr_initial_states%values(:, :, i))

        end do

    end subroutine multi_polynomial_rr_initial_states_fill_parameters

    subroutine serr_mu_parameters_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k

        ! SErr mu parameters is third control kind
        j = sum(parameters%control%nbk(1:2))

        do i = 1, setup%nsep_mu

            if (options%optimize%serr_mu_parameters(i) .eq. 0) cycle

            do k = 1, mesh%ng

                if (options%cost%gauge(k) .eq. 0) cycle

                j = j + 1

                parameters%serr_mu_parameters%values(k, i) = parameters%control%x(j)

            end do

        end do

    end subroutine serr_mu_parameters_fill_parameters

    subroutine serr_sigma_parameters_fill_parameters(setup, mesh, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer :: i, j, k

        ! SErr mu parameters is fourth control kind
        j = sum(parameters%control%nbk(1:3))

        do i = 1, setup%nsep_sigma

            if (options%optimize%serr_sigma_parameters(i) .eq. 0) cycle

            do k = 1, mesh%ng

                if (options%cost%gauge(k) .eq. 0) cycle

                j = j + 1

                parameters%serr_sigma_parameters%values(k, i) = parameters%control%x(j)

            end do

        end do

    end subroutine serr_sigma_parameters_fill_parameters

    subroutine fill_parameters(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        select case (options%optimize%mapping)

        case ("uniform")

            call uniform_rr_parameters_fill_parameters(setup, mesh, parameters, options)
            call uniform_rr_initial_states_fill_parameters(setup, mesh, parameters, options)

        case ("distributed")

            call distributed_rr_parameters_fill_parameters(setup, mesh, parameters, options)
            call distributed_rr_initial_states_fill_parameters(setup, mesh, parameters, options)

        case ("multi-linear")

            call multi_linear_rr_parameters_fill_parameters(setup, mesh, input_data, parameters, options)
            call multi_linear_rr_initial_states_fill_parameters(setup, mesh, input_data, parameters, options)

        case ("multi-polynomial")

            call multi_polynomial_rr_parameters_fill_parameters(setup, mesh, input_data, parameters, options)
            call multi_polynomial_rr_initial_states_fill_parameters(setup, mesh, input_data, parameters, options)

        end select

        ! Directly working with hyper parameters
        call serr_mu_parameters_fill_parameters(setup, mesh, parameters, options)
        call serr_sigma_parameters_fill_parameters(setup, mesh, parameters, options)

    end subroutine fill_parameters

    subroutine parameters_to_control(setup, mesh, input_data, parameters, options)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(OptionsDT), intent(in) :: options

        integer, dimension(size(parameters%control%nbk)) :: nbk

        call get_control_sizes(setup, mesh, options, nbk)

        call ControlDT_initialise(parameters%control, nbk)

        call fill_control(setup, mesh, input_data, parameters, options)

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

        call fill_parameters(setup, mesh, input_data, parameters, options)

    end subroutine control_to_parameters

end module mwd_parameters_manipulation
