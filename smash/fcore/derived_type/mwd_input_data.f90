!%      (MWD) Module Wrapped and Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Input_DataDT
!%          Container for all user input data (not only forcing data but all inputs
!%          needed to run and/or optimize the model). This data are not meant to be
!%          changed at runtime once read.
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``response_data``        Response_DataDT
!%          ``u_response_data``      U_Response_DataDT
!%          ``physio_data``          Physio_DataDT
!%          ``atmos_data``           Atmos_DataDT
!%          ======================== =======================================
!%
!%      Subroutine
!%      ----------
!%
!%      - Input_DataDT_initialise
!%      - Input_DataDT_copy

module mwd_input_data

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_response_data !%: only: ResponseDataDT, ResponseDataDT_initialise
    use mwd_u_response_data !%: only: U_ResponseDataDT, U_ResponseDataDT_initialise
    use mwd_physio_data !%: only: Physio_DataDT, Physio_DataDT_initialise
    use mwd_atmos_data !%: only: Atmos_DataDT, Atmos_DataDT_initialise

    implicit none

    type Input_DataDT

        type(Response_DataDT) :: response_data

        type(U_Response_DataDT) :: u_response_data

        type(Physio_DataDT) :: physio_data

        type(Atmos_DataDT) :: atmos_data

    end type Input_DataDT

contains

    subroutine Input_DataDT_initialise(this, setup, mesh)

        implicit none

        type(Input_DataDT), intent(inout) :: this
        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh

        call Response_DataDT_initialise(this%response_data, setup, mesh)

        call U_Response_DataDT_initialise(this%u_response_data, setup, mesh)

        call Physio_DataDT_initialise(this%physio_data, setup, mesh)

        call Atmos_DataDT_initialise(this%atmos_data, setup, mesh)

    end subroutine Input_DataDT_initialise

    subroutine Input_DataDT_copy(this, this_copy)

        implicit none

        type(Input_DataDT), intent(in) :: this
        type(Input_DataDT), intent(out) :: this_copy

        this_copy = this

    end subroutine Input_DataDT_copy

end module mwd_input_data
