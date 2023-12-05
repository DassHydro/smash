.. _api_reference:

=============
API Reference
=============

This page gives an overview of all public objects, functions, and methods available in the `smash` library. 
The core functionality of the API revolves around the primary `smash.Model` object, which serves as the central component for modeling and simulations.

Additionally, the following sub-packages provide access to essential tools and functionalities:

- `smash.factory`: Methods for creating essential elements, that are utilized by the Model object without requiring its prior instantiation.
- `smash.io`: Methods for handling input and output operations related to data objects.

.. toctree::
   :maxdepth: 2

   principal_methods/model
   principal_methods/simulation
   principal_methods/signal_analysis

.. toctree::
    :maxdepth: 2

    sub-packages/factory
    sub-packages/io

.. toctree::
    :maxdepth: 2

    returned_objects/index

.. toctree::
    :maxdepth: 2

    fortran/index
