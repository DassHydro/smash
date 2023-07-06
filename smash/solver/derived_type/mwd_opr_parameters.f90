!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%
!%      - Opr_ParametersDT
!%
!%          ========================== =====================================
!%          `Variables`                Description
!%          ========================== =====================================
!%          ``ci``                     GR interception capacity
!%          ``cp``                     GR production capacity
!%          ``cft``                    GR first transfer capacity
!%          ``cst``                    GR second transfer capacity
!%          ``kexc``                   GR exchange flux
!%          ``llr``                    Linear routing lag time
!%          ``akw``                    Kinematic wave alpha parameter
!%          ``bkw``                    Kinematic wave beta parameter
!%
!%
!%      Subroutine
!%      ----------
!%
!%      - Opr_ParametersDT_initialise
!%      - Opr_ParametersDT_copy

module mwd_opr_parameters

   use md_constant !% only: sp
   use mwd_mesh !% only: MeshDT

   implicit none

   type Opr_ParametersDT

      real(sp), dimension(:, :), allocatable :: ci
      real(sp), dimension(:, :), allocatable :: cp
      real(sp), dimension(:, :), allocatable :: cft
      real(sp), dimension(:, :), allocatable :: cst
      real(sp), dimension(:, :), allocatable :: kexc

      real(sp), dimension(:, :), allocatable :: llr
      real(sp), dimension(:, :), allocatable :: akw
      real(sp), dimension(:, :), allocatable :: bkw

   end type Opr_ParametersDT

contains

   subroutine Opr_ParametersDT_initialise(this, mesh)
      !% Default parameters value will be handled in Python

      implicit none

      type(Opr_ParametersDT), intent(inout) :: this
      type(MeshDT), intent(in) :: mesh

      allocate (this%ci(mesh%nrow, mesh%ncol))
      this%ci = 0._sp

      allocate (this%cp(mesh%nrow, mesh%ncol))
      this%cp = 0._sp

      allocate (this%cft(mesh%nrow, mesh%ncol))
      this%cft = 0._sp

      allocate (this%cst(mesh%nrow, mesh%ncol))
      this%cst = 0._sp

      allocate (this%kexc(mesh%nrow, mesh%ncol))
      this%kexc = 0._sp

      allocate (this%llr(mesh%nrow, mesh%ncol))
      this%llr = 0._sp

      allocate (this%akw(mesh%nrow, mesh%ncol))
      this%akw = 0._sp

      allocate (this%bkw(mesh%nrow, mesh%ncol))
      this%bkw = 0._sp

   end subroutine Opr_ParametersDT_initialise

   subroutine Opr_ParametersDT_copy(this, this_copy)

      implicit none

      type(Opr_ParametersDT), intent(in) :: this
      type(Opr_ParametersDT), intent(out) :: this_copy

      this_copy = this

   end subroutine Opr_ParametersDT_copy

end module mwd_opr_parameters
