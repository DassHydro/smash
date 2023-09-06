.. _api_reference.principal_methods.model:

=====
Model
=====

.. currentmodule:: smash

Constructor
***********
.. autosummary::
   :toctree: smash/
   
   Model
   Model.copy
   
Attributes
**********
.. autosummary::
   :toctree: smash/

   Model.setup
   Model.mesh
   Model.obs_response
   Model.physio_data
   Model.atmos_data
   Model.opr_parameters
   Model.opr_initial_states
   Model.sim_response
   Model.opr_final_states
   
Simulation
**********
.. autosummary::
   :toctree: smash/

   Model.forward_run
   Model.optimize
   Model.multiset_estimate

Parameters/States
*****************
.. autosummary::
   :toctree: smash/

   Model.get_opr_parameters
   Model.get_opr_initial_states
   Model.get_opr_final_states
   Model.set_opr_parameters
   Model.set_opr_initial_states
   Model.get_opr_parameters_bounds
   Model.get_opr_initial_states_bounds