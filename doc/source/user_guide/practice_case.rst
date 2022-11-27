.. _user_guide.practice_case:

=============
Practice case
=============

The Practice case is an introduction to `smash` for new users. The objective of this section is to create the dataset to run `smash` from scratch and get an overview of what is available. More details are provided on a real case available in the User Guide section: :ref:`user_guide.real_case_cance`.

For this case, a fictitious square-shaped catchment of size 10 x 10 km² will be created with the following drained area and flow directions:

.. image:: ../_static/flwdir_da_Practice_case.png
    :width: 750
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
    
.. warning::

    - The wrapping of Fortran code in Python requires the use of the `f90wrap <https://github.com/jameskermode/f90wrap>`__ package, which itself uses `f2py <https://numpy.org/doc/stable/f2py/>`__. Thus, the `NumPy <https://numpy.org/>`__ package is essential in the management of arguments/tables. A knowledge of this package is advised in the use of `smash`.
    
    - The `Matplotlib <https://matplotlib.org/>`__ package is the visualization package used in the `smash` documentation but any tool can be used.
    
---------------------   
Model object creation
---------------------

Creating a :class:`.Model` requires two input arguments: ``setup`` and ``mesh``.


.. _user_guide.practice_case.setup_argument_creation:

Setup argument creation
***********************
    
``setup`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary Fortran arrays). 

.. note::
    
    Each key and associated values that can be passed into the ``setup`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.model_initialization.setup>`.

A minimal ``setup`` configuration is:

- ``dt``: the calculation time step in s,

- ``start_time``: the beginning of the simulation,

- ``end_time``: the end of the simulation.

.. ipython:: python

    setup = {
        "dt": 3_600,
        "start_time": "2020-01-01 00:00",
        "end_time": "2020-01-04 00:00",
    }
    
.. _user_guide.practice_case.mesh_argument_creation:
    
Mesh argument creation
**********************

``mesh`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary Fortran arrays). 

.. note::
    
    - Each key and associated values that can be passed into the ``mesh`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.model_initialization.mesh>`.
    
    - In the Practice case, because the catchment is ficticious, we create the ``mesh`` dictionary ourselves. In the case of a real catchment, the meshing generation can be done automatically via the meshing method :meth:`smash.generate_mesh`. More details can be found in the User Guide section: :ref:`user_guide.real_case_cance`.

First part of  ``mesh`` configuration is:

- ``dx``: the calculation spatial step in m,

- ``nrow``: the number of rows,

- ``ncol``: the number of columns,

- ``ng``: the number of gauges,

- ``nac``: the number of cells that contribute to any gauge discharge (here the full grid contributes),

- ``area``: the catchment area in m²,

- ``gauge_pos``: the gauge position in the grid (here lower right corner [9,9]),

- ``code``: the gauge code.

.. ipython:: python

    dx = 1_000
    (nrow, ncol) = (10, 10)

    mesh = {
        "dx": dx,
        "nrow": nrow,
        "ncol": ncol,
        "ng": 1,
        "nac": nrow * ncol,
        "area": nrow * ncol * (dx ** 2),
        "gauge_pos": np.array([9, 9], dtype=np.int32),
        "code": np.array(["Practice_case"])
    }

Second part of ``mesh`` configuration is:

- ``flwdir``: the flow directions,

- ``drained_area``: the drained area in number of cells.

.. ipython:: python

    mesh["flwdir"] = np.array(
        [
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [3, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        [3, 3, 4, 5, 5, 5, 5, 5, 5, 5],
        [3, 3, 3, 4, 5, 5, 5, 5, 5, 5],
        [3, 3, 3, 3, 4, 5, 5, 5, 5, 5],
        [3, 3, 3, 3, 3, 4, 5, 5, 5, 5],
        [3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
        [3, 3, 3, 3, 3, 3, 3, 4, 5, 5],
        [3, 3, 3, 3, 3, 3, 3, 3, 4, 5],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 4],
        ],
        dtype=np.int32,
    )
    
    mesh["drained_area"] = np.array(
        [
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 4, 2, 2, 2, 2, 2, 2, 2, 2],
               [1, 2, 9, 3, 3, 3, 3, 3, 3, 3],
               [1, 2, 3, 16, 4, 4, 4, 4, 4, 4],
               [1, 2, 3, 4, 25, 5, 5, 5, 5, 5],
               [1, 2, 3, 4, 5, 36, 6, 6, 6, 6],
               [1, 2, 3, 4, 5, 6, 49, 7, 7, 7],
               [1, 2, 3, 4, 5, 6, 7, 64, 8, 8],
               [1, 2, 3, 4, 5, 6, 7, 8, 81, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            ],
            dtype=np.int32,
        )


Finally, the calculation path (``path``) must be provided (ascending order of drained area). This can be directly computed from ``drained_area`` and NumPy methods.

.. ipython:: python

    ind_path = np.unravel_index(np.argsort(mesh["drained_area"], axis=None),
         mesh["drained_area"].shape)

    mesh["path"] = np.zeros(shape=(2, mesh["drained_area"].size), 
        dtype=np.int32)

    mesh["path"][0, :] = ind_path[0]
    mesh["path"][1, :] = ind_path[1]
    

Once ``setup`` and ``mesh`` are filled in, a :class:`.Model` object can be created:

.. ipython:: python
    
    model = smash.Model(setup, mesh)

    model

.. note::

    The representation of the :class:`.Model` object is very simple and only displays the structure used, the dimensions and the last action that updated the object. More information on what the object contains is available below.
    
-------------
Viewing Model
-------------

Once the :class:`.Model` object is created, it is possible to visualize what it contains through 6 attributes. This 6 attributes are Python classes that are derived from the wrapping of Fortran derived types.

.. note::

    See details in the :ref:`api_reference` for the attributes:
    
    - :attr:`.Model.setup`
    
    - :attr:`.Model.mesh`
    
    - :attr:`.Model.input_data`
    
    - :attr:`.Model.parameters`
    
    - :attr:`.Model.states`
    
    - :attr:`.Model.output`

Setup
*****

The :attr:`.Model.setup` attribute contains a set of arguments necessary to initialize the :class:`.Model`. We have in the :ref:`user_guide.practice_case.setup_argument_creation` part given values for the arguments ``dt``, ``start_time`` and ``end_time``. These values can be retrieved in the following way:

.. ipython:: python

    model.setup.dt, model.setup.start_time, model.setup.end_time
    
The other :attr:`.Model.setup` arguments can also be viewed even if they have not been directly defined in the :class:`.Model` initialization. These arguments have default values in the code:

.. ipython:: python

    model.setup.structure, model.setup.prcp_indice
    
If you are using IPython, tab completion allows you to visualize all the attributes and methods:

.. ipython:: python
    
    @verbatim
    model.setup.<TAB>
    model.setup.copy(                   model.setup.prcp_directory
    model.setup.daily_interannual_pet   model.setup.prcp_format
    model.setup.descriptor_directory    model.setup.prcp_indice
    model.setup.descriptor_format       model.setup.qobs_directory
    model.setup.descriptor_name         model.setup.read_descriptor
    model.setup.dt                      model.setup.read_pet
    model.setup.end_time                model.setup.read_prcp
    model.setup.from_handle(            model.setup.read_qobs
    model.setup.mean_forcing            model.setup.save_net_prcp_domain
    model.setup.pet_conversion_factor   model.setup.save_qsim_domain
    model.setup.pet_directory           model.setup.sparse_storage
    model.setup.pet_format              model.setup.start_time
    model.setup.prcp_conversion_factor  model.setup.structure
    
Mesh
****

The :attr:`.Model.mesh` attribute contains a set of arguments necessary to initialize the :class:`.Model`. We have in the :ref:`user_guide.practice_case.mesh_argument_creation` part given values for multiple arguments. These values can be retrieved in the following way:

.. ipython:: python

    model.mesh.dx, model.mesh.nrow, model.mesh.ncol
    
NumPy array can also be viewed:

.. ipython:: python

    model.mesh.drained_area
    
Or plotted using Matplotlib.

.. ipython:: python
    
    plt.imshow(model.mesh.drained_area, cmap="Spectral");
    plt.colorbar(label="Number of cells");
    @savefig da_pc_user_guide.png
    plt.title("Practice case - Drained Area");

If you are using IPython, tab completion allows you to visualize all the attributes and methods:

.. ipython:: python
    
    @verbatim
    model.mesh.<TAB>
    model.mesh.active_cell   model.mesh.gauge_pos
    model.mesh.area          model.mesh.nac
    model.mesh.code          model.mesh.ncol
    model.mesh.copy(         model.mesh.ng
    model.mesh.drained_area  model.mesh.nrow
    model.mesh.dx            model.mesh.path
    model.mesh.flwdir        model.mesh.xmin
    model.mesh.flwdst        model.mesh.ymax
    model.mesh.from_handle(


Input Data
**********

The :attr:`.Model.input_data` attribute contains a set of arguments storing :class:`.Model` input data (i.e. atmospheric forcings, observed discharge ...). As we did not specify in the :ref:`user_guide.practice_case.setup_argument_creation` part a reading of input data, all tables are empty but allocated according to the size of the grid and the simulation period. 

For example, the observed discharge is a NumPy array of shape (1, 72). There is 1 gauge in the grid and the simulation period is up to 72 time steps. The value -99 indicates no data.

.. ipython:: python

    model.input_data.qobs
    
    model.input_data.qobs.shape
    
Precipitation is also a NumPy array but of shape (10, 10, 72). The number of rows and columns is 10 and same as the observed dicharge, the simulation period is up to 72 time steps.

.. ipython:: python

    model.input_data.prcp.shape

If you are using IPython, tab completion allows you to visualize all the attributes and methods:

.. ipython:: python
    
    @verbatim
    model.input_data.<TAB>
    model.input_data.copy(         model.input_data.prcp
    model.input_data.descriptor    model.input_data.prcp_indice
    model.input_data.from_handle(  model.input_data.qobs
    model.input_data.mean_pet      model.input_data.sparse_pet
    model.input_data.mean_prcp     model.input_data.sparse_prcp
    model.input_data.pet    
    
.. warning::

    It can happen, depending on the :class:`.Model` initialization, that some arguments of type NumPy array are not accessible (unallocated array in the Fortran code). For example, we did not ask in the ``setup`` to store catchment descriptors. Access to this variable is therefore impossible and the code will return the following error:
    
    .. ipython:: python
        :okexcept:
            
        model.input_data.descriptor
        
Parameters and States
*********************

The :attr:`.Model.parameters` and :attr:`.Model.states` attributes contain a set of arguments storing :class:`.Model` parameters and states. This attributes contain only NumPy arrays of shape (10, 10) (i.e. number of rows and columns in the grid).

.. ipython:: python
    
    model.parameters.cp.shape, model.states.hp.shape
    
This arrays are filled in with uniform default values.

.. ipython:: python
    
    model.parameters.cp, model.states.hp
    
.. note:: 

    The :attr:`.Model.states` attribute stores the **initial** states :math:`h(x,0)`.
    
If you are using IPython, tab completion allows you to visualize all the attributes and methods:

.. ipython:: python
    
    @verbatim
    model.parameters.<TAB>
    model.parameters.alpha         model.parameters.cusl1
    model.parameters.b             model.parameters.cusl2
    model.parameters.beta          model.parameters.ds
    model.parameters.cft           model.parameters.dsm
    model.parameters.ci            model.parameters.exc
    model.parameters.clsl          model.parameters.from_handle(
    model.parameters.copy(         model.parameters.ks
    model.parameters.cp            model.parameters.lr
    model.parameters.cst           model.parameters.ws

    
.. ipython:: python
    
    @verbatim
    model.states.<TAB>
    model.states.copy(         model.states.hlsl
    model.states.from_handle(  model.states.hp
    model.states.hft           model.states.hst
    model.states.hi            model.states.husl1
    model.states.hlr           model.states.husl2
    
Output
******

The last attribute, :attr:`.Model.output`, contains a set of arguments storing :class:`.Model` outputs (i.e. simulated discharge, final states, cost ...). The attribute values are empty as long as no simulation has been run.

If you are using IPython, tab completion allows you to visualize all the attributes and methods:

.. ipython:: python
    
    @verbatim
    model.output.<TAB>
    model.output.an                   model.output.parameters_gradient
    model.output.copy(                model.output.qsim
    model.output.cost                 model.output.qsim_domain
    model.output.from_handle(         model.output.sp1
    model.output.fstates              model.output.sp2
    model.output.ian                  model.output.sparse_qsim_domain


------------------
Input Data filling
------------------

To run a simulation, the :class:`.Model` needs at least one precipitation and potential evapotranspiration (PET) chronicle. In this Practice case, we will impose a triangular precipitation over the simulation period, uniform on the domain and a zero PET.

.. ipython:: python

    prcp = np.zeros(shape=model.input_data.prcp.shape[2], dtype=np.float32)
    
    tri = np.linspace(0, 6.25, 10)
    
    prcp[0:10] = tri
    
    prcp[9:19] = np.flip(tri)
    
    model.input_data.prcp = np.broadcast_to(
        prcp,
        model.input_data.prcp.shape,
    )

    model.input_data.pet = 0.
    
Checking on any cell the precipitation values:

.. ipython:: python

    plt.plot(model.input_data.prcp[0,0,:]);
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Precipitation $(mm/h)$");
    @savefig prpc_pc_user_guide.png
    plt.title("Precipitation on cell (0,0)");
   
    
---
Run
---

Forward run
***********

The :class:`.Model` is finally ready to be run using the :meth:`.Model.run` method:
    
.. ipython:: python

    model.run(inplace=True);

    model
    
Once the run is done, it is possible to access the simulated discharge on the gauge via the :attr:`.Model.output` and to plot a hydrograph.
    
    
.. ipython:: python

    plt.plot(model.output.qsim[0,:]);
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Simulated discharge $(m^3/s)$");
    @savefig qsim_fwd_pc_user_guide.png
    plt.title(model.mesh.code[0]);
    
    

This hydrograph is the result of a forward run of the code with the default structure (``gr-a``), parameters and initial states.
    
Optimization
************

To perform an optimization, observed discharge must be provided to :class:`.Model`. Since the Practice case is a ficticious catchment, we will use the simulated data from the previous forward run as observed discharge.

.. ipython:: python

    model.input_data.qobs = model.output.qsim.copy()
    
Next, we will perturb the production parameter :math:`\mathrm{cp}` to generate a hydrograph different from the previous one.

.. ipython:: python

    model.parameters.cp = 1
    
Re run to see the difference between the hydrographs.

.. ipython:: python

    model.run(inplace=True);
    
    plt.plot(model.input_data.qobs[0,:], label="Observed discharge");
    plt.plot(model.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.legend();
    @savefig qsim_fwd2_pc_user_guide.png
    plt.title(model.mesh.code[0]);
    
Finally, perform a spatially uniform calibration (which is default optimization) of the parameter :math:`\mathrm{cp}` with the :meth:`.Model.optimize` method:

.. ipython:: python

    model.optimize(control_vector="cp", inplace=True);
    
    model.parameters.cp

    model

.. ipython:: python
    
    plt.plot(model.input_data.qobs[0,:], label="Observed discharge");
    plt.plot(model.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.legend();
    @savefig qsim_opt_pc_user_guide.png
    plt.title(model.mesh.code[0]);
    
.. note::
    
    In the Practice case, we will not go into the details of the optimization which is an essential part of the `smash` calculation code. To go further, details can be found for the use of the :meth:`.Model.optimize` method in the User Guide section: :ref:`user_guide.real_case_cance` and for the algorithms in the Model description section: (TODO link).
    
    
------------
Getting data
------------

The last step is to save what we have entered in :class:`.Model` (i.e. ``setup`` and ``mesh`` dictionaries) and the :class:`.Model` itself.


Setup argument in/out
*********************

The setup dictionary ``setup``, which was created in the section :ref:`user_guide.practice_case.setup_argument_creation`, can be saved in `YAML <https://yaml.org/spec/1.2.2/>`__ format via the method :meth:`smash.save_setup`.

.. ipython:: python

    smash.save_setup(setup, "setup.yaml")
    
A file named ``setup.yaml`` has been created in the current working directory containing the ``setup`` dictionary information. This file can itself be opened in order to recover our initial ``setup`` dictionary via the method :meth:`smash.read_setup`.

.. ipython:: python

    setup2 = smash.read_setup("setup.yaml")
    
    setup2
    
Mesh argument in/out
********************

In a similar way to ``setup`` dictionary, the ``mesh`` dictionary created in the section :ref:`user_guide.practice_case.mesh_argument_creation` can be saved to file via the method :meth:`smash.save_mesh`. However, 3D NumPy arrays cannot be saved in YAML format, so the ``mesh`` is saved in `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ format.

.. ipython:: python

    smash.save_mesh(mesh, "mesh.hdf5")
    
A file named ``mesh.hdf5`` has been created in the current working directory containing the ``mesh`` dictionary information. This file can itself be opened in order to recover our initial ``mesh`` dictionary via the method :meth:`smash.read_mesh`.

.. ipython:: python

    mesh2 = smash.read_mesh("mesh.hdf5")
    
    mesh2
    
A new :class:`.Model` object can be created from the read files (same as the first one).

.. ipython:: python

    model2 = smash.Model(setup2, mesh2)

    model2
    
Model in/out
************

The :class:`.Model` object can also be saved to file. Like the ``mesh``, it will be saved in HDF5 format using the :meth:`smash.save_model` method. Here, we will save the :class:`.Model` object ``model`` after optimization.

.. ipython:: python

    smash.save_model(model, "model.hdf5")

A file named ``model.hdf5`` has been created in the current working directory containing the ``model`` object information. This file can itself be opened in order to recover our initial ``model`` object via the method :meth:`smash.read_model`.

.. ipython:: python

    model3 = smash.read_model("model.hdf5")

    model3

``model3`` is directly the :class:`.Model` object itself on which the methods associated with the object are applicable.

.. ipython:: python

    model3.run(inplace=True);

    model3
