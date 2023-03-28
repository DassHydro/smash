.. _api_reference.net:

===
Net
===

.. currentmodule:: smash

Constructor
***********
.. autosummary::
   :toctree: smash/
   
   Net

Attributes
**********
.. autosummary::
   :toctree: smash/

   Net.layers
   Net.history

Functions
*********
.. autosummary::
   :toctree: smash/

   Net.add

.. toctree::
   :hidden:
   :maxdepth: 1

   add_dense
   add_activation
   add_scale
   add_dropout

.. autosummary::
   :toctree: smash/

   Net.compile

.. toctree::
   :hidden:
   :maxdepth: 1

   compile_sgd
   compile_adam
   compile_adagrad
   compile_rmsprop

.. autosummary::
   :toctree: smash/

   Net.copy
   Net.set_trainable