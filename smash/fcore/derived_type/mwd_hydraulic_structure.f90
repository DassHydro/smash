!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Hydraulic_StructureDT
!%          Container for hydraulic structure data such as dams and inflow ...
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``response_data``        Response_DataDT
!%          ``u_response_data``      U_Response_DataDT
!%          ======================== =======================================
!%
!%      - DamDT
!%          Container data specific to a dam: two law H(Q) and H(V) with H the elevation in the dam
!%          Q the outflow discharge and V the volume inside the dam
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``dam_hq``               Relation water elevation (H) / outflow discharge
!%          ``dam_hv``               Relation water elevation / dam volume
!%          ======================== =======================================
!%      - InflowDT
!%          Container for a chronic of inflows
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``inflow``               Chronic of inflows
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Hydraulic_StructureDT_initialise
!%      - Hydraulic_StructureDT_copy
!%      - DamDT_initialise
!%      - DamDT_copy
!%      - InflowDT_initialise
!%      - InflowDT_copy
module mwd_hydraulic_structure

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT

    type DamDT
        real(sp), dimension(:, :, :), allocatable :: dam_hv
        real(sp), dimension(:, :, :), allocatable :: dam_hq
    end type DamDT

    type InflowDT
        real(sp), dimension(:, :), allocatable :: inflow
    end type InflowDT

    type Hydraulic_StructureDT

        type(DamDT) :: dam_structure
        type(InflowDT) :: inflow_structure

    end type Hydraulic_StructureDT

contains

    subroutine Hydraulic_StructureDT_initialise(this, setup, mesh)

        implicit none

        type(Hydraulic_StructureDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call DamDT_initialise(this%dam_structure, mesh)
        call InflowDT_initialise(this%inflow_structure, setup, mesh)

    end subroutine Hydraulic_StructureDT_initialise

    subroutine DamDT_initialise(this, mesh)

        implicit none

        type(DamDT), intent(inout) :: this
        type(MeshDT), intent(in) :: mesh

        integer :: nmax_val

        nmax_val = 100

        allocate (this%dam_hv(mesh%ndam, 2, nmax_val))
        allocate (this%dam_hq(mesh%ndam, 2, nmax_val))

        this%dam_hv = -99.
        this%dam_hq = -99.

    end subroutine DamDT_initialise

    subroutine InflowDT_initialise(this, setup, mesh)

        implicit none

        type(InflowDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        allocate (this%inflow(mesh%ninflow, setup%ntime_step))

        this%inflow = -99.

    end subroutine InflowDT_initialise

    subroutine Hydraulic_StructureDT_copy(this, this_copy)

        implicit none

        type(Hydraulic_StructureDT), intent(in) :: this
        type(Hydraulic_StructureDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Hydraulic_StructureDT_copy

    subroutine DamDT_copy(this, this_copy)

        implicit none

        type(DamDT), intent(in) :: this
        type(DamDT), intent(out) :: this_copy

        this_copy = this

    end subroutine DamDT_copy

    subroutine InflowDT_copy(this, this_copy)

        implicit none

        type(InflowDT), intent(in) :: this
        type(InflowDT), intent(out) :: this_copy

        this_copy = this

    end subroutine InflowDT_copy

end module mwd_hydraulic_structure
