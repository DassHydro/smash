!%      This module `mw_derived_type_update` encapsulates all SMASH derived type update routines.
!%      This module is wrapped

module mw_derived_type_update

   use md_constant, only: sp, dp, GLB_PARAMETERS, GUB_PARAMETERS, GLB_STATES, GUB_STATES
   use mwd_setup, only: Optimize_SetupDT

   implicit none

contains

   subroutine update_optimize_setup(this, ntime_step, nd, ng, mapping, njf)

      implicit none

      type(Optimize_SetupDT), intent(inout) :: this
      integer, intent(in) :: ntime_step, nd, ng, njf
      character(len=*), intent(in) :: mapping

      if (ng .ne. size(this%wgauge)) then

         !% Not necessary to check allocated statement but safer
         if (allocated(this%wgauge)) deallocate (this%wgauge)
         allocate (this%wgauge(ng))

         this%wgauge = 1._sp/ng

      end if

      this%mapping = mapping

      this%normalize_forward = .false.

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

      if (ng .ne. size(this%mask_event, 1) .or. ntime_step .ne. size(this%mask_event, 2)) then

         if (allocated(this%mask_event)) deallocate (this%mask_event)
         allocate (this%mask_event(ng, ntime_step))
         this%mask_event = 0

      end if

      this%lb_parameters = GLB_PARAMETERS
      this%ub_parameters = GUB_PARAMETERS
      this%lb_states = GLB_STATES
      this%ub_states = GUB_STATES

      this%optim_parameters = 0
      this%optim_states = 0

   end subroutine update_optimize_setup

end module mw_derived_type_update
