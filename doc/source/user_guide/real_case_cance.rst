.. _user_guide.real_case_cance:

=================
Real case - Cance
=================

A real case is considered: the Cance river catchment at Sarras, a right bank tributary of the Rhône river. 

.. image:: ../_static/real_case_cance_catchment.png
    :width: 400
    :align: center

First, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import numpy as np
    import matplotlib.pyplot as plt

---------------------   
Model object creation
---------------------

Creating a :class:`.Model` requires two input arguments: ``setup`` and ``mesh``. For this case, it is possible to directly load the both input dictionnaries using the :meth:`smash.load_dataset` method.


.. ipython:: python

    setup, mesh = smash.load_dataset("Cance")
    
.. _setup_argument_creation:

Setup argument creation
***********************
    
``setup`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary setup Fortran arrays). 

.. note::
    
    Each key and associated values that can be passed into the ``setup`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.model_initialization.setup>`.
    
Compared to the :ref:`user_guide.practice_case`, more options have been filled in the ``setup`` dictionary.

.. ipython:: python

    setup
    
To get into the details, especially for the new options:

- ``dt``: the calculation time step in s.


In a similar way to the :ref:`user_guide.practice_case`, we fill in the time options of the :class:`.Model`.

.. ipython:: python

    time_options = {
        "dt": 3_600,
        "start_time": "2014-09-15 00:00",
        "end_time": "2020-11-14 00:00",
    }

Then, we fill in the arguments to ask during the initialization of the :class:`.Model` to read the observed discharges, the precipitation and the evapotranspiration (PET).

- Observed dicharge
    The observed discharge for one catchment is read from a ``.csv`` file with the following structure: 

    .. csv-table:: V3524010.csv
        :align: center
        :header: "200601010000"
        :width: 50

        -99.000
        -99.000
        ...
        1.180
        1.185
        
    It is a single-column ``.csv`` file containing the observed discharge values in m\ :sup:`3` \/s (negative values correspond to a gap in the chronicle) and whose header is the first time step of the chronicle.
    The name of the file, for any catchment, must contains the code of the gauge which is filled in the mesh dictionary.
    
    .. note::
    
        The time step of the header does not have to match the first simulation time step. `smash` manages to read the corresponding lines from ``start_time``, ``end_time`` and ``dt``.


- Rain precipitation

    ...
