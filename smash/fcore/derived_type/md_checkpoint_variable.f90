!%      (MD) Module Differentiated.
!%
!%      Type
!%      ----
!%
!%      - Checkpoint_VariableDT
!%          Checkpoint variables passed to simulation_checkpoint subroutine. It stores variables that must
!%          be checkpointed by the adjoint model (i.e. variables that are push/pop each time step)
!%
!%          ======================== =======================================
!%          `Variables`              Description
!%          ======================== =======================================
!%          ``ac_rr_parameters``     Active cell rainfall-runoff parameters
!%          ``ac_rr_states``         Active cell rainfall-runoff states
!%          ``ac_mlt``               Active cell melt flux (snow module output)
!%          ``ac_qtz``               Active cell elemental discharge with time buffer (hydrological module output)
!%          ``ac_qz``                Active cell surface discharge with time buffer (routing module output)
!%          ======================== =======================================

module md_checkpoint_variable

    use md_constant !% only: sp

    implicit none

    type Checkpoint_VariableDT

        real(sp), dimension(:, :), allocatable :: ac_rr_parameters
        real(sp), dimension(:, :), allocatable :: ac_rr_states
        real(sp), dimension(:), allocatable :: ac_mlt
        real(sp), dimension(:, :), allocatable :: ac_qtz, ac_qz

    end type Checkpoint_VariableDT

end module md_checkpoint_variable
