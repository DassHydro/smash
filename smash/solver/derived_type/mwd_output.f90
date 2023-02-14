!%      This module `mwd_output` encapsulates all SMASH output.
!%      This module is wrapped and differentiated.
!%
!%      OutputDT type:
!%
!%      </> Public
!%      ========================== =====================================
!%      `Variables`                Description
!%      ========================== =====================================
!%      ``qsim``                   Simulated discharge at gauge            [m3/s]
!%      ``qsim_domain``            Simulated discharge whole domain        [m3/s]
!%      ``sparse_qsim_domain``     Sparse simulated discharge whole domain [m3/s]
!%      ``net_prcp_domain``        Net precipitaition whole domain         [mm/dt]
!%      ``sparse_net_prcp_domain`` Sparse net precipitation whole domain   [mm/dt]
!%      ``cost``                   Cost value
!%      ``fstates``                Final states (StatesDT)
!%      ========================== =====================================
!%
!%      contains
!%
!%      [1] OutputDT_initialise

module mwd_output

   use md_constant !% only: sp
   use mwd_setup  !% only: SetupDT
   use mwd_mesh   !%only: MeshDT
   use mwd_states !%only: StatesDT, StatesDT_initialise

   implicit none

   type :: OutputDT

      !% Notes
      !% -----
      !% OutputDT Derived Type.

      real(sp), dimension(:, :), allocatable :: qsim
      real(sp), dimension(:, :, :), allocatable :: qsim_domain
      real(sp), dimension(:, :), allocatable :: sparse_qsim_domain

      real(sp), dimension(:, :, :), allocatable :: net_prcp_domain
      real(sp), dimension(:, :), allocatable   :: sparse_net_prcp_domain

      real(sp) :: cost

      type(StatesDT) :: fstates

   end type OutputDT

contains

   subroutine OutputDT_initialise(this, setup, mesh)

      !% Notes
      !% -----
      !% OutputDT initialisation subroutine.

      implicit none

      type(OutputDT), intent(inout) :: this
      type(SetupDT), intent(inout) :: setup
      type(MeshDT), intent(inout) :: mesh

      if (mesh%ng .gt. 0) then

         allocate (this%qsim(mesh%ng, setup%ntime_step))
         this%qsim = -99._sp

      end if

      if (setup%save_qsim_domain) then

         if (setup%sparse_storage) then

            allocate (this%sparse_qsim_domain(mesh%nac, &
            & setup%ntime_step))
            this%sparse_qsim_domain = -99._sp

         else

            allocate (this%qsim_domain(mesh%nrow, mesh%ncol, &
            & setup%ntime_step))
            this%qsim_domain = -99._sp

         end if

      end if

      if (setup%save_net_prcp_domain) then

         if (setup%sparse_storage) then

            allocate (this%sparse_net_prcp_domain(mesh%nac, &
            & setup%ntime_step))
            this%sparse_net_prcp_domain = -99._sp

         else

            allocate (this%net_prcp_domain(mesh%nrow, mesh%ncol, &
            & setup%ntime_step))
            this%net_prcp_domain = -99._sp

         end if

      end if

      call StatesDT_initialise(this%fstates, mesh)

   end subroutine OutputDT_initialise

end module mwd_output
