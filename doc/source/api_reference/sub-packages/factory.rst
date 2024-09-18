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

      detect_sink
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
      smash/smash.factory.Net.add_dense
      smash/smash.factory.Net.add_conv2d
      smash/smash.factory.Net.add_scale
      smash/smash.factory.Net.add_flatten
      smash/smash.factory.Net.add_dropout
      smash/smash.factory.Net.copy
      smash/smash.factory.Net.set_trainable
      smash/smash.factory.Net.set_weight
      smash/smash.factory.Net.set_bias
      smash/smash.factory.Net.get_weight
      smash/smash.factory.Net.get_bias
      smash/smash.factory.Net.forward_pass

Sample Generation
*****************
.. autosummary::
      :toctree: smash/

      generate_samples

