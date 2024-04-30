.. _api_reference.sub-packages.factory:

=======
Factory
=======

.. currentmodule:: smash.factory

This sub-package provides methods for creating essential elements, that are utilized by the Model object without requiring its prior instantiation.

Dataset Loading
***************
.. autosummary::
      :toctree: smash/

      load_dataset

Mesh Generation
***************
.. autosummary::
      :toctree: smash/

      generate_mesh

Neural Network Configuration
****************************
.. autosummary::
      :toctree: smash/

      Net

.. toctree::
      :hidden:
      :maxdepth: 1

      smash/smash.factory.Net.layers
      smash/smash.factory.Net.history
      net/add_dense
      net/add_activation
      net/add_scale
      net/add_dropout

Sample Generation
*****************
.. autosummary::
      :toctree: smash/

      generate_samples

