.. _api_reference.model:

=====
Model
=====

.. currentmodule:: smash

Constructor
***********
.. autosummary::
   :toctree: smash/
   
   Model
   
Attributes
**********
.. autosummary::
   :toctree: smash/

   Model.setup
   Model.mesh
   Model.input_data
   Model.parameters
   Model.states
   Model.output
   
Simulation
**********
.. autosummary::
   :toctree: smash/

   Model.run
   Model.multiple_run
   Model.optimize

.. toctree::
   :hidden:
   :maxdepth: 1

   optimize_sbs
   optimize_nelder-mead
   optimize_l-bfgs-b

.. autosummary::
   :toctree: smash/

   Model.bayes_estimate
   Model.bayes_optimize
   Model.ann_optimize

Event segmentation
******************
.. autosummary::
   :toctree: smash/

   Model.event_segmentation

Signatures
**********
.. autosummary::
   :toctree: smash/

   Model.signatures
   Model.signatures_sensitivity

Precipitation indices
*********************
.. autosummary::
   :toctree: smash/

   Model.prcp_indices
   
Others
******
.. autosummary::
   :toctree: smash/

   Model.copy
   Model.get_bound_constraints
