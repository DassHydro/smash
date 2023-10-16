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
   Model.response_data
   Model.u_response_data
   Model.physio_data
   Model.atmos_data
   Model.rr_parameters
   Model.rr_initial_states
   Model.serr_mu_parameters
   Model.serr_sigma_parameters
   Model.response
   Model.rr_final_states
   
Simulation
**********
.. autosummary::
   :toctree: smash/

   Model.forward_run
   Model.optimize
   Model.multiset_estimate
   Model.bayesian_optimize

Parameters/States
*****************
.. autosummary::
   :toctree: smash/

   Model.get_rr_parameters
   Model.get_rr_initial_states
   Model.get_serr_mu_parameters
   Model.get_serr_sigma_parameters
   Model.get_rr_final_states
   Model.set_rr_parameters
   Model.set_rr_initial_states
   Model.set_serr_mu_parameters
   Model.set_serr_sigma_parameters
   Model.get_rr_parameters_bounds
   Model.get_rr_initial_states_bounds
   Model.get_serr_mu_parameters_bounds
   Model.get_serr_sigma_parameters_bounds
   Model.get_serr_mu
   Model.get_serr_sigma
