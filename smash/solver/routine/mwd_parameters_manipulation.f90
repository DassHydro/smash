!%      This module `mwd_parameters_manipulation` encapsulates all SMASH parameters manipulation.
!%      This module is wrapped and differentiated.
!%
!%      set_parameters interface:
!%
!%      module procedure set0d_parameters
!%      module procedure set1d_parameters
!%      module procedure set3d_parameters
!%
!%      set_hyper_parameters interface:
!%
!%      module procedure set0d_hyper_parameters
!%      module procedure set1d_hyper_parameters
!%      module procedure set3d_hyper_parameters
!%
!%      contains
!%
!%      [1]   get_parameters
!%      [2]   set0d_parameters
!%      [3]   set1d_parameters
!%      [4]   set3d_parameters
!%      [5]   normalize_parameters
!%      [6]   denormalize_parameters
!%      [7]   get_hyper_parameters
!%      [8]   set0d_hyper_parameters
!%      [9]   set1d_hyper_parameters
!%      [10]  set3d_hyper_parameters
!%      [11]  hyper_parameters_to_parameters

module mwd_parameters_manipulation

    use md_constant
    use mwd_setup
    use mwd_mesh
    use mwd_input_data
    use mwd_parameters

    implicit none

    interface set_parameters

        module procedure set0d_parameters
        module procedure set1d_parameters
        module procedure set3d_parameters

    end interface set_parameters

    interface set_hyper_parameters

        module procedure set0d_hyper_parameters
        module procedure set1d_hyper_parameters
        module procedure set3d_hyper_parameters

    end interface set_hyper_parameters

contains

!%      TODO comment
    subroutine get_parameters(mesh, parameters, a)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(in) :: parameters
        real(sp), dimension(mesh%nrow, mesh%ncol, GNP), intent(inout) :: a

        a(:, :, 1) = parameters%ci(:, :)
        a(:, :, 2) = parameters%cp(:, :)
        a(:, :, 3) = parameters%beta(:, :)
        a(:, :, 4) = parameters%cft(:, :)
        a(:, :, 5) = parameters%cst(:, :)
        a(:, :, 6) = parameters%alpha(:, :)
        a(:, :, 7) = parameters%exc(:, :)

        a(:, :, 8) = parameters%b(:, :)
        a(:, :, 9) = parameters%cusl1(:, :)
        a(:, :, 10) = parameters%cusl2(:, :)
        a(:, :, 11) = parameters%clsl(:, :)
        a(:, :, 12) = parameters%ks(:, :)
        a(:, :, 13) = parameters%ds(:, :)
        a(:, :, 14) = parameters%dsm(:, :)
        a(:, :, 15) = parameters%ws(:, :)

        a(:, :, 16) = parameters%lr(:, :)

    end subroutine get_parameters

    subroutine set3d_parameters(mesh, parameters, a)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        real(sp), dimension(mesh%nrow, mesh%ncol, GNP), intent(in) :: a

        parameters%ci(:, :) = a(:, :, 1)
        parameters%cp(:, :) = a(:, :, 2)
        parameters%beta(:, :) = a(:, :, 3)
        parameters%cft(:, :) = a(:, :, 4)
        parameters%cst(:, :) = a(:, :, 5)
        parameters%alpha(:, :) = a(:, :, 6)
        parameters%exc(:, :) = a(:, :, 7)

        parameters%b(:, :) = a(:, :, 8)
        parameters%cusl1(:, :) = a(:, :, 9)
        parameters%cusl2(:, :) = a(:, :, 10)
        parameters%clsl(:, :) = a(:, :, 11)
        parameters%ks(:, :) = a(:, :, 12)
        parameters%ds(:, :) = a(:, :, 13)
        parameters%dsm(:, :) = a(:, :, 14)
        parameters%ws(:, :) = a(:, :, 15)

        parameters%lr(:, :) = a(:, :, 16)

    end subroutine set3d_parameters

    subroutine set1d_parameters(mesh, parameters, a)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        real(sp), dimension(GNP), intent(in) :: a

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: a3d
        integer :: i

        do i = 1, GNP

            a3d(:, :, i) = a(i)

        end do

        call set3d_parameters(mesh, parameters, a3d)

    end subroutine set1d_parameters

    subroutine set0d_parameters(mesh, parameters, a)

        implicit none

        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters
        real(sp), intent(in) :: a

        real(sp), dimension(GNP) :: a1d

        a1d(:) = a

        call set1d_parameters(mesh, parameters, a1d)

    end subroutine set0d_parameters

    subroutine normalize_parameters(setup, mesh, parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: a
        real(sp) :: lb, ub
        integer :: i

        call get_parameters(mesh, parameters, a)

        do i = 1, GNP

            lb = setup%optimize%lb_parameters(i)
            ub = setup%optimize%ub_parameters(i)

            a(:, :, i) = (a(:, :, i) - lb)/(ub - lb)
            
        end do

        call set_parameters(mesh, parameters, a)

    end subroutine normalize_parameters

    subroutine denormalize_parameters(setup, mesh, parameters)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(ParametersDT), intent(inout) :: parameters

        real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: a
        real(sp) :: lb, ub
        integer :: i

        call get_parameters(mesh, parameters, a)

        do i = 1, GNP

            lb = setup%optimize%lb_parameters(i)
            ub = setup%optimize%ub_parameters(i)

            a(:, :, i) = a(:, :, i)*(ub - lb) + lb

        end do

        call set_parameters(mesh, parameters, a)

    end subroutine denormalize_parameters

    subroutine get_hyper_parameters(setup, hyper_parameters, a)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(in) :: hyper_parameters
        real(sp), dimension(setup%optimize%nhyper, 1, GNP), intent(inout) :: a

        a(:, :, 1) = hyper_parameters%ci(:, :)
        a(:, :, 2) = hyper_parameters%cp(:, :)
        a(:, :, 3) = hyper_parameters%beta(:, :)
        a(:, :, 4) = hyper_parameters%cft(:, :)
        a(:, :, 5) = hyper_parameters%cst(:, :)
        a(:, :, 6) = hyper_parameters%alpha(:, :)
        a(:, :, 7) = hyper_parameters%exc(:, :)

        a(:, :, 8) = hyper_parameters%b(:, :)
        a(:, :, 9) = hyper_parameters%cusl1(:, :)
        a(:, :, 10) = hyper_parameters%cusl2(:, :)
        a(:, :, 11) = hyper_parameters%clsl(:, :)
        a(:, :, 12) = hyper_parameters%ks(:, :)
        a(:, :, 13) = hyper_parameters%ds(:, :)
        a(:, :, 14) = hyper_parameters%dsm(:, :)
        a(:, :, 15) = hyper_parameters%ws(:, :)

        a(:, :, 16) = hyper_parameters%lr(:, :)

    end subroutine get_hyper_parameters

    subroutine set3d_hyper_parameters(setup, hyper_parameters, a)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
        real(sp), dimension(setup%optimize%nhyper, 1, GNP), intent(in) :: a

        hyper_parameters%ci(:, :) = a(:, :, 1)
        hyper_parameters%cp(:, :) = a(:, :, 2)
        hyper_parameters%beta(:, :) = a(:, :, 3)
        hyper_parameters%cft(:, :) = a(:, :, 4)
        hyper_parameters%cst(:, :) = a(:, :, 5)
        hyper_parameters%alpha(:, :) = a(:, :, 6)
        hyper_parameters%exc(:, :) = a(:, :, 7)

        hyper_parameters%b(:, :) = a(:, :, 8)
        hyper_parameters%cusl1(:, :) = a(:, :, 9)
        hyper_parameters%cusl2(:, :) = a(:, :, 10)
        hyper_parameters%clsl(:, :) = a(:, :, 11)
        hyper_parameters%ks(:, :) = a(:, :, 12)
        hyper_parameters%ds(:, :) = a(:, :, 13)
        hyper_parameters%dsm(:, :) = a(:, :, 14)
        hyper_parameters%ws(:, :) = a(:, :, 15)

        hyper_parameters%lr(:, :) = a(:, :, 16)

    end subroutine set3d_hyper_parameters

    subroutine set1d_hyper_parameters(setup, hyper_parameters, a)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
        real(sp), dimension(GNP), intent(in) :: a

        real(sp), dimension(setup%optimize%nhyper, 1, GNP) :: a3d
        integer :: i

        do i = 1, GNP

            a3d(:, :, i) = a(i)

        end do

        call set3d_hyper_parameters(setup, hyper_parameters, a3d)

    end subroutine set1d_hyper_parameters

    subroutine set0d_hyper_parameters(setup, hyper_parameters, a)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(Hyper_ParametersDT), intent(inout) :: hyper_parameters
        real(sp), intent(in) :: a

        real(sp), dimension(GNP) :: a1d

        a1d(:) = a

        call set1d_hyper_parameters(setup, hyper_parameters, a1d)

    end subroutine set0d_hyper_parameters

!%      TODO comment
    subroutine hyper_parameters_to_parameters(hyper_parameters, &
    & parameters, setup, mesh, input_data)

        implicit none

        type(Hyper_ParametersDT), intent(in) :: hyper_parameters
        type(ParametersDT), intent(inout) :: parameters
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data

        real(sp), dimension(setup%optimize%nhyper, 1, GNP) :: hyper_parameters_matrix
        real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: parameters_matrix
        real(sp), dimension(mesh%nrow, mesh%ncol) :: d, dpb
        integer :: i, j
        real(sp) :: a, b

        call get_hyper_parameters(setup, hyper_parameters, hyper_parameters_matrix)
        call get_parameters(mesh, parameters, parameters_matrix)

        !% Add mask later here
        !% 1 in dim2 will be replace with k and apply where on Omega
        do i = 1, GNP

            parameters_matrix(:, :, i) = hyper_parameters_matrix(1, 1, i)

            do j = 1, setup%nd

                d = input_data%descriptor(:, :, j)

                select case (trim(setup%optimize%mapping))

                case ("hyper-linear")

                    a = hyper_parameters_matrix(j + 1, 1, i)
                    b = 1._sp

                case ("hyper-polynomial")

                    a = hyper_parameters_matrix(2*j, 1, i)
                    b = hyper_parameters_matrix(2*j + 1, 1, i)

                end select

                dpb = d**b

                parameters_matrix(:, :, i) = parameters_matrix(:, :, i) + a*dpb

            end do

            !% sigmoid transformation lambda = 1
            parameters_matrix(:, :, i) = (setup%optimize%ub_parameters(i) - setup%optimize%lb_parameters(i)) &
                                        & *(1._sp/(1._sp + exp(-parameters_matrix(:, :, i)))) + setup%optimize%lb_parameters(i)

        end do

        call set_parameters(mesh, parameters, parameters_matrix)

    end subroutine hyper_parameters_to_parameters

end module mwd_parameters_manipulation
