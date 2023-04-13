.. _user_guide.in_depth.multiple_run:

============
Multiple run
============

Here, we aim to compute multiple forward run of a single Model object (in parallel) in order to retrieve the cost and/or discharge values for each parameters set tested.

To get started, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import numpy as np

---------------------
Model object creation
---------------------

First, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Cance`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)
