!%      (MD) Module Differentiated.
!%
!%      Subroutine
!%      ----------
!%
!%      gr_a_forward

module md_forward_structure

   use md_constant !% only: sp
   use mwd_setup !% only: SetupDT
   use mwd_mesh !% only: MeshDT
   use mwd_input_data !% only: Input_DataDT
   use mwd_opr_parameters !% only: GR_A_Opr_ParametersDT
   use mwd_opr_states !% only: GR_A_Opr_StatesDT
   use mwd_output !% only: OutputDT
   use mwd_options !% only: OptionsDT
   use mwd_returns !% only: ReturnsDT
   use md_gr_operator !% only: gr_interception, gr_production, gr_exchange, &
   !% & gr_transfer
   use md_routing_operator !% only: upstream_discharge, linear_routing

   implicit none

contains

   subroutine gr_a_forward(setup, mesh, input_data, opr_parameters, opr_states, output, options, returns)

      implicit none

      !% =================================================================================================================== %!
      !%   Derived Type Variables (shared)
      !% =================================================================================================================== %!

      type(SetupDT), intent(in) :: setup
      type(MeshDT), intent(in) :: mesh
      type(Input_DataDT), intent(in) :: input_data
      type(GR_A_Opr_ParametersDT), intent(in) :: opr_parameters
      type(GR_A_Opr_StatesDT), intent(inout) :: opr_states
      type(OutputDT), intent(inout) :: output
      type(OptionsDT), intent(in) :: options
      type(ReturnsDT), intent(inout) :: returns

      !% =================================================================================================================== %!
      !%   Local Variables (private)
      !% =================================================================================================================== %!

      real(sp), dimension(mesh%nrow, mesh%ncol) :: q, qt
      real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
      & qr, qd, qup, qrout
      integer :: t, i, row, col, k, g

      !% =================================================================================================================== %!
      !%   Begin subroutine
      !% =================================================================================================================== %!

      do t = 1, setup%ntime_step !% [ DO TIME ]

         !$omp parallel do num_threads(options%comm%ncpu) shared(setup, mesh, input_data, &
         !$omp& opr_parameters, opr_states, output, qt), &
         !$omp& private(i, ei, pn, en, pr, perc, l, prr, prd, qr, qd, row, col, prcp, pet)
         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            !% =============================================================================================================== %!
            !%   Local Variables Initialisation for time step (t) and cell (i)
            !% =============================================================================================================== %!

            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
            prd = 0._sp
            qr = 0._sp
            qd = 0._sp

            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
            !% =========================================================================================================== %!

            row = mesh%path(1, i)
            col = mesh%path(2, i)
            if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

            !% ======================================================================================================= %!
            !%   Global/Local active cell
            !% ======================================================================================================= %!

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            if (setup%sparse_storage) then

               prcp = input_data%atmos_data%sparse_prcp(k, t)
               pet = input_data%atmos_data%sparse_pet(k, t)

            else

               prcp = input_data%atmos_data%prcp(row, col, t)
               pet = input_data%atmos_data%pet(row, col, t)

            end if

            if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

               !% =============================================================================================== %!
               !%   Interception module
               !% =============================================================================================== %!

               ei = min(pet, prcp)

               pn = max(0._sp, prcp - ei)

               en = pet - ei

               !% =============================================================================================== %!
               !%   Production module
               !% =============================================================================================== %!

               call gr_production(pn, en, opr_parameters%cp(row, col), 1000._sp, &
               & opr_states%hp(row, col), pr, perc)

               !% =============================================================================================== %!
               !%   Exchange module
               !% =============================================================================================== %!

               call gr_exchange(opr_parameters%exc(row, col), opr_states%hft(row, col), l)

            end if !% [ END IF PRCP GAP ]

            !% =================================================================================================== %!
            !%   Transfer module
            !% =================================================================================================== %!

            prr = 0.9_sp*(pr + perc) + l
            prd = 0.1_sp*(pr + perc)

            call gr_transfer(5._sp, prcp, prr, opr_parameters%cft(row, col), opr_states%hft(row, col), qr)

            qd = max(0._sp, prd + l)

            qt(row, col) = (qr + qd)

         end do
         !$omp end parallel do

         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            qup = 0._sp
            qrout = 0._sp

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            !% =================================================================================================== %!
            !%   Routing module
            !% =================================================================================================== %!

            call upstream_discharge(mesh%nrow, mesh%ncol, setup%dt, &
            & mesh%dx(row, col), mesh%dy(row, col), row, col, mesh%flwdir, &
            & mesh%flwacc, q, qup)

            call linear_routing(setup%dt, qup, opr_parameters%lr(row, col), opr_states%hlr(row, col), qrout)

            q(row, col) = ((qt(row, col)*mesh%dx(row, col)*mesh%dy(row, col)) &
            & + (qrout*(mesh%flwacc(row, col) - mesh%dx(row, col)*mesh%dy(row, col)))) &
            & *1e-3_sp/setup%dt

         end do !% [ END DO SPACE ]

         !% =============================================================================================================== %!
         !%   Store simulated discharge at gauge
         !% =============================================================================================================== %!

         do g = 1, mesh%ng

            output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

         end do

      end do !% [ END DO TIME ]

   end subroutine gr_a_forward

   subroutine gr_b_forward(setup, mesh, input_data, opr_parameters, opr_states, output, options, returns)

      implicit none

      !% =================================================================================================================== %!
      !%   Derived Type Variables (shared)
      !% =================================================================================================================== %!

      type(SetupDT), intent(in) :: setup
      type(MeshDT), intent(in) :: mesh
      type(Input_DataDT), intent(in) :: input_data
      type(GR_B_Opr_ParametersDT), intent(in) :: opr_parameters
      type(GR_B_Opr_StatesDT), intent(inout) :: opr_states
      type(OutputDT), intent(inout) :: output
      type(OptionsDT), intent(in) :: options
      type(ReturnsDT), intent(inout) :: returns

      !% =================================================================================================================== %!
      !%   Local Variables (private)
      !% =================================================================================================================== %!

      real(sp), dimension(mesh%nrow, mesh%ncol) :: q, qt
      real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
      & qr, qd, qup, qrout
      integer :: t, i, row, col, k, g

      !% =================================================================================================================== %!
      !%   Begin subroutine
      !% =================================================================================================================== %!

      do t = 1, setup%ntime_step !% [ DO TIME ]

         !$omp parallel do num_threads(options%comm%ncpu) shared(setup, mesh, input_data, &
         !$omp& opr_parameters, opr_states, output, qt), &
         !$omp& private(i, ei, pn, en, pr, perc, l, prr, prd, qr, qd, row, col, prcp, pet)
         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            !% =============================================================================================================== %!
            !%   Local Variables Initialisation for time step (t) and cell (i)
            !% =============================================================================================================== %!

            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
            prd = 0._sp
            qr = 0._sp
            qd = 0._sp

            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
            !% =========================================================================================================== %!

            row = mesh%path(1, i)
            col = mesh%path(2, i)
            if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

            !% ======================================================================================================= %!
            !%   Global/Local active cell
            !% ======================================================================================================= %!

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            if (setup%sparse_storage) then

               prcp = input_data%atmos_data%sparse_prcp(k, t)
               pet = input_data%atmos_data%sparse_pet(k, t)

            else

               prcp = input_data%atmos_data%prcp(row, col, t)
               pet = input_data%atmos_data%pet(row, col, t)

            end if

            if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

               !% =============================================================================================== %!
               !%   Interception module
               !% =============================================================================================== %!

               call gr_interception(prcp, pet, opr_parameters%ci(row, col), opr_states%hi(row, col), pn, ei)

               en = pet - ei

               !% =============================================================================================== %!
               !%   Production module
               !% =============================================================================================== %!

               call gr_production(pn, en, opr_parameters%cp(row, col), 1000._sp, &
               & opr_states%hp(row, col), pr, perc)

               !% =============================================================================================== %!
               !%   Exchange module
               !% =============================================================================================== %!

               call gr_exchange(opr_parameters%exc(row, col), opr_states%hft(row, col), l)

            end if !% [ END IF PRCP GAP ]

            !% =================================================================================================== %!
            !%   Transfer module
            !% =================================================================================================== %!

            prr = 0.9_sp*(pr + perc) + l
            prd = 0.1_sp*(pr + perc)

            call gr_transfer(5._sp, prcp, prr, opr_parameters%cft(row, col), opr_states%hft(row, col), qr)

            qd = max(0._sp, prd + l)

            qt(row, col) = (qr + qd)

         end do
         !$omp end parallel do

         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            qup = 0._sp
            qrout = 0._sp

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            !% =================================================================================================== %!
            !%   Routing module
            !% =================================================================================================== %!

            call upstream_discharge(mesh%nrow, mesh%ncol, setup%dt, &
            & mesh%dx(row, col), mesh%dy(row, col), row, col, mesh%flwdir, &
            & mesh%flwacc, q, qup)

            call linear_routing(setup%dt, qup, opr_parameters%lr(row, col), opr_states%hlr(row, col), qrout)

            q(row, col) = ((qt(row, col)*mesh%dx(row, col)*mesh%dy(row, col)) &
            & + (qrout*(mesh%flwacc(row, col) - mesh%dx(row, col)*mesh%dy(row, col)))) &
            & *1e-3_sp/setup%dt

         end do !% [ END DO SPACE ]

         !% =============================================================================================================== %!
         !%   Store simulated discharge at gauge
         !% =============================================================================================================== %!

         do g = 1, mesh%ng

            output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

         end do

      end do !% [ END DO TIME ]

   end subroutine gr_b_forward

   subroutine gr_c_forward(setup, mesh, input_data, opr_parameters, opr_states, output, options, returns)

      implicit none

      !% =================================================================================================================== %!
      !%   Derived Type Variables (shared)
      !% =================================================================================================================== %!

      type(SetupDT), intent(in) :: setup
      type(MeshDT), intent(in) :: mesh
      type(Input_DataDT), intent(in) :: input_data
      type(GR_C_Opr_ParametersDT), intent(in) :: opr_parameters
      type(GR_C_Opr_StatesDT), intent(inout) :: opr_states
      type(OutputDT), intent(inout) :: output
      type(OptionsDT), intent(in) :: options
      type(ReturnsDT), intent(inout) :: returns

      !% =================================================================================================================== %!
      !%   Local Variables (private)
      !% =================================================================================================================== %!

      real(sp), dimension(mesh%nrow, mesh%ncol) :: q, qt
      real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prl, prd, &
      & qr, ql, qd, qup, qrout
      integer :: t, i, row, col, k, g

      !% =================================================================================================================== %!
      !%   Begin subroutine
      !% =================================================================================================================== %!

      do t = 1, setup%ntime_step !% [ DO TIME ]

         !$omp parallel do num_threads(options%comm%ncpu) shared(setup, mesh, input_data, &
         !$omp& opr_parameters, opr_states, output, qt), &
         !$omp& private(i, ei, pn, en, pr, perc, l, prr, prl, prd, qr, qd, row, col, prcp, pet)
         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            !% =============================================================================================================== %!
            !%   Local Variables Initialisation for time step (t) and cell (i)
            !% =============================================================================================================== %!

            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            l = 0._sp
            prr = 0._sp
            prl = 0._sp
            prd = 0._sp
            qr = 0._sp
            ql = 0._sp
            qd = 0._sp

            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
            !% =========================================================================================================== %!

            row = mesh%path(1, i)
            col = mesh%path(2, i)
            if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

            !% ======================================================================================================= %!
            !%   Global/Local active cell
            !% ======================================================================================================= %!

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            if (setup%sparse_storage) then

               prcp = input_data%atmos_data%sparse_prcp(k, t)
               pet = input_data%atmos_data%sparse_pet(k, t)

            else

               prcp = input_data%atmos_data%prcp(row, col, t)
               pet = input_data%atmos_data%pet(row, col, t)

            end if

            if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

               !% =============================================================================================== %!
               !%   Interception module
               !% =============================================================================================== %!

               call gr_interception(prcp, pet, opr_parameters%ci(row, col), opr_states%hi(row, col), pn, ei)

               en = pet - ei

               !% =============================================================================================== %!
               !%   Production module
               !% =============================================================================================== %!

               call gr_production(pn, en, opr_parameters%cp(row, col), 1000._sp, &
               & opr_states%hp(row, col), pr, perc)

               !% =============================================================================================== %!
               !%   Exchange module
               !% =============================================================================================== %!

               call gr_exchange(opr_parameters%exc(row, col), opr_states%hft(row, col), l)

            end if !% [ END IF PRCP GAP ]

            !% =================================================================================================== %!
            !%   Transfer module
            !% =================================================================================================== %!

            prr = 0.9_sp*0.6_sp*(pr + perc) + l
            prl = 0.9_sp*0.4_sp*(pr + perc)
            prd = 0.1_sp*(pr + perc)

            call gr_transfer(5._sp, prcp, prr, opr_parameters%cft(row, col), opr_states%hft(row, col), qr)

            call gr_transfer(5._sp, prcp, prl, opr_parameters%cst(row, col), opr_states%hst(row, col), ql)

            qd = max(0._sp, prd + l)

            qt = (qr + ql + qd)

         end do
         !$omp end parallel do

         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            qup = 0._sp
            qrout = 0._sp

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            !% =================================================================================================== %!
            !%   Routing module
            !% =================================================================================================== %!

            call upstream_discharge(mesh%nrow, mesh%ncol, setup%dt, &
            & mesh%dx(row, col), mesh%dy(row, col), row, col, mesh%flwdir, &
            & mesh%flwacc, q, qup)

            call linear_routing(setup%dt, qup, opr_parameters%lr(row, col), opr_states%hlr(row, col), qrout)

            q(row, col) = ((qt(row, col)*mesh%dx(row, col)*mesh%dy(row, col)) &
            & + (qrout*(mesh%flwacc(row, col) - mesh%dx(row, col)*mesh%dy(row, col)))) &
            & *1e-3_sp/setup%dt

         end do !% [ END DO SPACE ]

         !% =============================================================================================================== %!
         !%   Store simulated discharge at gauge
         !% =============================================================================================================== %!

         do g = 1, mesh%ng

            output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

         end do

      end do !% [ END DO TIME ]

   end subroutine gr_c_forward

   subroutine gr_d_forward(setup, mesh, input_data, opr_parameters, opr_states, output, options, returns)

      implicit none

      !% =================================================================================================================== %!
      !%   Derived Type Variables (shared)
      !% =================================================================================================================== %!

      type(SetupDT), intent(in) :: setup
      type(MeshDT), intent(in) :: mesh
      type(Input_DataDT), intent(in) :: input_data
      type(GR_D_Opr_ParametersDT), intent(in) :: opr_parameters
      type(GR_D_Opr_StatesDT), intent(inout) :: opr_states
      type(OutputDT), intent(inout) :: output
      type(OptionsDT), intent(in) :: options
      type(ReturnsDT), intent(inout) :: returns

      !% =================================================================================================================== %!
      !%   Local Variables (private)
      !% =================================================================================================================== %!

      real(sp), dimension(mesh%nrow, mesh%ncol) :: q, qt
      real(sp) :: prcp, pet, ei, pn, en, pr, perc, prr, &
      & qr, qup, qrout
      integer :: t, i, row, col, k, g

      !% =================================================================================================================== %!
      !%   Begin subroutine
      !% =================================================================================================================== %!

      do t = 1, setup%ntime_step !% [ DO TIME ]

         !$omp parallel do num_threads(options%comm%ncpu) shared(setup, mesh, input_data, &
         !$omp& opr_parameters, opr_states, output, qt), &
         !$omp& private(i, ei, pn, en, pr, perc, prr, qr, row, col, prcp, pet)
         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            !% =============================================================================================================== %!
            !%   Local Variables Initialisation for time step (t) and cell (i)
            !% =============================================================================================================== %!

            ei = 0._sp
            pn = 0._sp
            en = 0._sp
            pr = 0._sp
            perc = 0._sp
            prr = 0._sp
            qr = 0._sp

            !% =========================================================================================================== %!
            !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
            !% =========================================================================================================== %!

            row = mesh%path(1, i)
            col = mesh%path(2, i)
            if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

            !% ======================================================================================================= %!
            !%   Global/Local active cell
            !% ======================================================================================================= %!

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            if (setup%sparse_storage) then

               prcp = input_data%atmos_data%sparse_prcp(k, t)
               pet = input_data%atmos_data%sparse_pet(k, t)

            else

               prcp = input_data%atmos_data%prcp(row, col, t)
               pet = input_data%atmos_data%pet(row, col, t)

            end if

            if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

               !% =============================================================================================== %!
               !%   Interception module
               !% =============================================================================================== %!

               ei = min(pet, prcp)

               pn = max(0._sp, prcp - ei)

               en = pet - ei

               !% =============================================================================================== %!
               !%   Production module
               !% =============================================================================================== %!

               call gr_production(pn, en, opr_parameters%cp(row, col), 1000._sp, &
               & opr_states%hp(row, col), pr, perc)

            end if !% [ END IF PRCP GAP ]

            !% =================================================================================================== %!
            !%   Transfer module
            !% =================================================================================================== %!

            prr = pr + perc

            call gr_transfer(5._sp, prcp, prr, opr_parameters%cft(row, col), opr_states%hft(row, col), qr)

            qt(row, col) = qr

         end do
         !$omp end parallel do

         do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

            qup = 0._sp
            qrout = 0._sp

            row = mesh%path(1, i)
            col = mesh%path(2, i)

            if (mesh%active_cell(row, col) .eq. 0 .or. mesh%local_active_cell(row, col) .eq. 0) cycle !% [ CYCLE ACTIVE CELL ]

            !% =================================================================================================== %!
            !%   Routing module
            !% =================================================================================================== %!

            call upstream_discharge(mesh%nrow, mesh%ncol, setup%dt, &
            & mesh%dx(row, col), mesh%dy(row, col), row, col, mesh%flwdir, &
            & mesh%flwacc, q, qup)

            call linear_routing(setup%dt, qup, opr_parameters%lr(row, col), opr_states%hlr(row, col), qrout)

            q(row, col) = ((qt(row, col)*mesh%dx(row, col)*mesh%dy(row, col)) &
            & + (qrout*(mesh%flwacc(row, col) - mesh%dx(row, col)*mesh%dy(row, col)))) &
            & *1e-3_sp/setup%dt

         end do !% [ END DO SPACE ]

         !% =============================================================================================================== %!
         !%   Store simulated discharge at gauge
         !% =============================================================================================================== %!

         do g = 1, mesh%ng

            output%sim_response%q(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

         end do

      end do !% [ END DO TIME ]

   end subroutine gr_d_forward

end module md_forward_structure
