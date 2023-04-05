!%      This module `mw_derived_type_update` encapsulates all SMASH derived type update routines.
!%      This module is wrapped

module mw_derived_type_update

    use md_constant, only: sp, dp, GLB_PARAMETERS, GUB_PARAMETERS, GLB_STATES, GUB_STATES
    use mwd_setup, only: Optimize_SetupDT

    implicit none

contains

    subroutine reset_optimize_setup(this)

        implicit none

        type(Optimize_SetupDT), intent(inout) :: this

        this%algorithm = "..."

        this%jobs_fun = "..."
        this%wjobs_fun = 0._sp

        this%wjreg = 0._sp
        this%jreg_fun = "..."
        this%wjreg_fun = 1._sp

        this%reg_descriptors_for_params = 0
        this%reg_descriptors_for_states = 0

        this%njf = 0
        this%njr = 0

        this%verbose = .true.

        this%mapping = "..."

        this%denormalize_forward = .false.

        this%nhyper = 0

        this%optimize_start_step = 1

        this%maxiter = 100

        this%optim_parameters = 0
        this%optim_states = 0

        this%lb_parameters = GLB_PARAMETERS
        this%ub_parameters = GUB_PARAMETERS

        this%lb_states = GLB_STATES
        this%ub_states = GUB_STATES

        this%wgauge = 1._sp/size(this%wgauge)

        this%mask_event = 0

    end subroutine reset_optimize_setup

    subroutine update_optimize_setup_optimize_args(this, mapping, ntime_step, nd, ng, njf)

        implicit none

        type(Optimize_SetupDT), intent(inout) :: this
        character(len=*), intent(in) :: mapping
        integer, intent(in) :: ntime_step, nd, ng, njf

        this%mapping = mapping

        select case (trim(this%mapping))

        case ("hyper-linear")

            this%nhyper = (1 + nd)

        case ("hyper-polynomial")

            this%nhyper = (1 + 2*nd)

        end select

        this%njf = njf

        if (this%njf .ne. size(this%jobs_fun)) then

            if (allocated(this%jobs_fun)) deallocate (this%jobs_fun)
            if (allocated(this%wjobs_fun)) deallocate (this%wjobs_fun)
            allocate (this%jobs_fun(this%njf))
            allocate (this%wjobs_fun(this%njf))
            this%jobs_fun = "..."
            this%wjobs_fun = 0._sp

        end if

        if (ng .ne. size(this%wgauge)) then

            if (allocated(this%wgauge)) deallocate (this%wgauge)
            allocate (this%wgauge(ng))
            this%wgauge = 1._sp/ng

        end if

        if (ng .ne. size(this%mask_event, 1) .or. ntime_step .ne. size(this%mask_event, 2)) then

            if (allocated(this%mask_event)) deallocate (this%mask_event)
            allocate (this%mask_event(ng, ntime_step))
            this%mask_event = 0

        end if

    end subroutine update_optimize_setup_optimize_args

    subroutine update_optimize_setup_optimize_options(this, njr)

        implicit none

        type(Optimize_SetupDT), intent(inout) :: this
        integer, intent(in) :: njr

        this%njr = njr

        if (this%njr .ne. size(this%jreg_fun)) then

            if (allocated(this%jreg_fun)) deallocate (this%jreg_fun)
            if (allocated(this%wjreg_fun)) deallocate (this%wjreg_fun)
            allocate (this%jreg_fun(this%njr))
            allocate (this%wjreg_fun(this%njr))

            this%jreg_fun = "..."
            this%wjreg_fun = 1._sp

        end if

    end subroutine update_optimize_setup_optimize_options

end module mw_derived_type_update
