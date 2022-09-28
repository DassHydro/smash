.. _user_guide.real_case_cance:

=================
Real case - Cance
=================

A real case is considered: ``the Cance river catchment at Sarras``, a right bank tributary of the Rhône river. 

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
    
.. _user_guide.setup_argument:

Setup argument
**************
    
``setup`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary setup Fortran arrays). 

.. note::
    
    Each key and associated values that can be passed into the ``setup`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.model_initialization.setup>`.
    
Compared to the :ref:`user_guide.practice_case`, more options have been filled in the ``setup`` dictionary.

.. ipython:: python

    setup
    
To get into the details:

- ``dt``: the calculation time step in s,

- ``start_time``: the beginning of the simulation,

- ``end_time``: the end of the simulation,

- ``read_qobs``: whether or not to read observed discharges files,

- ``qobs_directory``: the path to the observed discharges files (this path is automatically generated when you load the data),

- ``read_prcp``: whether or not to read precipitation files,

- ``prcp_format``: the precipitation files format (``tif`` format is the only available at the moment),

- ``prcp_conversion_factor``: the precipitation conversion factor (the precipitation value will be **multiplied** by the conversion factor),

- ``prcp_directory``: the path to the precipitaion files (this path is automatically generated when you load the data),

- ``read_pet``: whether or not to read potential evapotranspiration files,

- ``pet_format``: the potential evapotranspiration files format (``tif`` format is the only available at the moment),

- ``pet_conversion_factor``: the potential evapotranspiration conversion factor (the potential evapotranspiration value will be **multiplied** by the conversion factor),

- ``daily_interannual_pet``: whether or not to read potential evapotranspiration file as daily interannual value desaggregated to the corresponding time step ``dt``,

- ``pet_directory``: the path to the potential evapotranspiration files (this path is automatically generated when you load the data),

- ``exchange_module``: Choice of the exchange module (``1`` is GR4 exchange module (TODO ref)),

- ``routing module``: Choice of the routing module (``1`` is linear routing (TODO ref)).

Before going into the explanation of the ``mesh``, the following section details the structure of the observed discharges, precipitation and potential evapotranspiration files read.

Input data files structure
**************************

Observed dicharge
'''''''''''''''''

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
The name of the file, for any catchment, must contains the code of the gauge which is filled in the ``mesh`` dictionary.
    
.. note::
    
    The time step of the header does not have to match the first simulation time step. `smash` manages to read the corresponding lines from ``start_time``, ``end_time`` and ``dt``.


Precipitation
'''''''''''''

The precipitation files must be store for each each time step of the simulation. For one time step, `smash` will recursively search in the ``prcp_directory``, a file with the following name structure: ``*<%Y%m%d%H%M>*.<prcp_format>``.
An example of file name in tif format for the date 2014-09-15 00:00: ``prcp_201409150000.tif``. The spatial resolution must be identical to the spatial resolution of the flow directions used for the meshing.

.. warning::
    
    ``%Y%m%d%H%M`` is a unique key, the ``prcp_directory`` (and all subdirectories) can not contains files with similar dates.
    
Potential evapotranspiration
''''''''''''''''''''''''''''

The potential evapotranspiration files must be store for each each time step of the simulation. For one time step, `smash` will recursively search in the ``pet_directory``, a file with the following name structure: ``*<%Y%m%d%H%M>*.<pet_format>``.
An example of file name in tif format for the date 2014-09-15 00:00: ``pet_201409150000.tif``. The spatial resolution must be identical to the spatial resolution of the flow directions used for the meshing.

.. warning::
    
    ``%Y%m%d%H%M`` is a unique key, the ``pet_directory`` (and all subdirectories) can not contains files with similar dates.
    
In case of ``daily_interannual_pet``, `smash` will recursively search in the ``pet_directory``, a file with the following name structure: ``*<%m%d>*.<pet_format>``.
An example of file name in tif format for the date 09-15: ``dia_pet_0915.tif``. This file will be desaggregated to the corresponding time step ``dt``.

.. _user_guide.mesh_argument:

Mesh argument
*************

Mesh composition
''''''''''''''''

``mesh`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary mesh Fortran arrays). 

.. note::
    
    Each key and associated values that can be passed into the ``mesh`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.model_initialization.mesh>`.
    
.. ipython:: python

    mesh.keys()
    
To get into the details:

- ``dx``: the calculation spatial step in m,

.. ipython:: python
    
    mesh["dx"]

- ``xmin``: the minimum value of the domain extension in x (it depends on the flow directions projection)

.. ipython:: python
    
    mesh["xmin"]

- ``ymax``: the maximum value of the domain extension in y (it depends on the flow directions projection)

.. ipython:: python
    
    mesh["ymax"]

- ``nrow``: the number of rows,

.. ipython:: python
    
    mesh["nrow"]

- ``ncol``: the number of columns,

.. ipython:: python
    
    mesh["ncol"]

- ``ng``: the number of gauges,

.. ipython:: python
    
    mesh["ng"]
    
- ``nac``: the number of cells that contribute to any gauge discharge,

.. ipython:: python
    
    mesh["nac"]
    
- ``area``: the catchments area in m²,

.. ipython:: python 
    
    mesh["area"]
    
- ``code``: the gauges code, 

.. ipython:: python
    
    mesh["code"]
    
.. warning::
    
    This argument is tricky to use because any NumPy uint8 array wrapped must be filled with ASCII values.
    One way to retrieve the value from uint8 array is:
    
    .. ipython:: python
        
        mesh["code"].tobytes("F").decode().split()
        
- ``gauge_pos``: the gauges position in the grid (it must follow **Fortran indexing**),

.. ipython:: python
    
    mesh["gauge_pos"]
    
- ``flwdir``: the flow directions,

.. ipython:: python
    
    plt.imshow(mesh["flwdir"]);
    plt.colorbar(label="Flow direction (D8)");
    @savefig flwdir_rc_Cance_user_guide.png
    plt.title("Real case - Cance - Flow direction");
    
- ``dirained_area``: the drained area in number of cells,

.. ipython:: python
    
    plt.imshow(mesh["drained_area"]);
    plt.colorbar(label="Drained area (nb cells)");
    @savefig da_rc_Cance_user_guide.png
    plt.title("Real case - Cance - Drained area");
    
- ``flwdst``: the flow distances from the main outlet in m,

.. ipython:: python
    
    plt.imshow(mesh["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig flwdst_rc_Cance_user_guide.png
    plt.title("Real case - Cance - Flow distance");
    
- ``active_cell``: the cells that contribute to any gauge discharge (mask),

.. ipython:: python
    
    plt.imshow(mesh["active_cell"]);
    plt.colorbar(label="Logical active cell (0: False, 1: True)");
    @savefig active_cell_rc_Cance_user_guide.png
    plt.title("Real case - Cance - Active cell");
    
- ``path``: the calculation path.

.. ipython:: python

    mesh["path"]

Obviously, the data set included in the ``mesh`` dictionary is not generated by hand. The method :meth:`smash.generate_mesh` allows from a flow directions file, the gauge coordinates and the area to generate this same data set.

Generate a mesh automatically
'''''''''''''''''''''''''''''

The method required the path to the flow directions ``tif`` file. Once can load it directly with,

.. ipython:: python

    flwdir = smash.load_dataset("flwdir")
    
    flwdir
    
This path leads to a flow directions ``tif`` file of the whole France at 1km² spatial resolution and Lambert93 projection (*EPSG:2154*)

Get the gauge coordinates, area and code (this data is considered to be known by the user at the time the meshing is generated):

.. ipython:: python
    
    x = [840_261, 826_553, 828_269]
    
    y = [6_457_807, 6_467_115, 6_469_198]
    
    area = [381.7 * 1e6, 107 * 1e6, 25.3 * 1e6]
    
    code = ["V3524010", "V3515010", "V3517010"]
    
The ``x`` and ``y`` coordinates of the gauge must be in the same projection of the flow directions used for the meshing, here Lambert93 (*EPSG:2154*).

Call the :meth:`smash.generate_mesh` method:

.. ipython:: python

    mesh2 = smash.generate_mesh(
        path=flwdir,
        x=x,
        y=y,
        area=area,
        code=code,
    )
    
This ``mesh`` created is a dictionnary which is identical to the ``mesh`` loaded with the :meth:`smash.load_dataset` method.

.. ipython:: python

    mesh2["dx"], mesh2["nrow"], mesh2["ncol"], mesh2["ng"], mesh2["gauge_pos"]
    
As a remainder, the ``mesh`` can be saved to HDF5 using the :meth:`smash.save_mesh` method and reload with the :meth:`smash.read_mesh` method.

.. ipython:: python

    smash.save_mesh(mesh2, "mesh_Cance.hdf5")
    
    mesh3 = smash.read_mesh("mesh_Cance.hdf5")
    
    mesh3["dx"], mesh3["nrow"], mesh3["ncol"], mesh3["ng"], mesh3["gauge_pos"]
    
Finally, create the :class:`.Model` object using the ``setup`` and ``mesh`` loaded.

.. ipython:: python

    model = smash.Model(setup, mesh)
    
    model
    
-------------
Viewing Model 
-------------

Similar to the :ref:`user_guide.practice_case`, it is possible to visualize what the :class:`.Model` contains through the 6 attributes: :attr:`.Model.setup`, :attr:`.Model.mesh`, :attr:`.Model.input_data`, 
:attr:`.Model.parameters`, :attr:`.Model.states`, :attr:`.Model.output`. As we have already detailed in the :ref:`user_guide.practice_case` the access to any data, we will only visualize the observed discharges and the spatialized atmospheric forcings here.

Input Data - Observed discharge
*******************************

3 gauges were placed in the meshing. For the sake of clarity, only the most downstream gauge discharge ``V3524010`` is plotted.

.. ipython:: python

    code = model.mesh.code.tobytes("F").decode().split()
    
    plt.plot(model.input_data.qobs[0,:]);
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge ($m^3/s$)");
    @savefig qobs_rc_Cance_user_guide.png
    plt.title(code[0]);
    
Input Data - Atmospheric forcings
*********************************

Precipitation and potential evapotranspiration files were read for each time step. For the sake of clarity, only one precipiation grid at time step 1200 is plotted.

.. ipython:: python

    plt.imshow(model.input_data.prcp[..., 1200]);
    plt.title("Precipitation at time step 1200");
    @savefig prcp_rc_Cance_user_guide.png
    plt.colorbar(label="Precipitation ($mm/h$)");
    
    
It is possible to mask the precipitation grid to only visualize the precipitation on active cells using NumPy method ``np.where``.

.. ipython:: python

    ma_prcp = np.where(
        model.mesh.active_cell == 0,
        np.nan,
        model.input_data.prcp[..., 1200]
    )
    
    plt.imshow(ma_prcp);
    plt.title("Masked precipitation at time step 1200");
    @savefig ma_prcp_rc_Cance_user_guide.png
    plt.colorbar(label="Precipitation ($mm/h$)");

---
Run 
---

.. note::
    
    We consider in the whole section, a model structure composed of 3 reservoirs for production, transfer and routing and a non-conservative exchange. 

Forward run
***********

Make a forward run using the :meth:`.Model.run` method.

.. ipython:: python

    model.run()
    
Here, unlike the :ref:`user_guide.practice_case`, we have not specified ``inplace=True``. By default, this argument is assigned to False, i.e. when the :meth:`.Model.run` method is called, the model object is not modified in-place but in a copy which can be returned.
So if we display the representation of the model, the last update will still be ``Initialization`` and no simulated discharge is available.

.. ipython:: python
    
    model
    
    model.output.qsim
    
This argument is useful to keep the original :class:`.Model` and store the results of the forward run in an other instance.

.. ipython:: python

    model_forward = model.run()
    
    model
    
    model_forward
    
We can visualize the simulated discharges after a forward run for the most downstream gauge.

.. ipython:: python

    plt.plot(model_forward.input_data.qobs[0,:], label="Observed discharge");
    plt.plot(model_forward.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    @savefig qsim_fw_rc_user_guide.png
    plt.legend();


Optimization
************

Let us briefly formulate here the general hydrological model calibration inverse problem. Let :math:`J \left( \theta \right)` be a cost function measuring the misfit between simulated and
observed quantities, such as discharge. Note that :math:`J` depends on the sought parameter set :math:`\theta` throught the hydrological model :math:`\mathcal{M}`. An optimal estimate of 
:math:`\hat{\theta}` of model parameter set is obtained from the condition:

.. math::
    
    \hat{\theta} = \underset{\theta}{\mathrm{argmin}} \; J\left( \theta \right)
    
Several calibration strategies are available in `smash`. They are based on different optimization algorithms and are for example adapted to inverse problems of various complexity, including high dimensional ones.
For the purposes of the user guide, we will only perform a spatially uniform and distributed optimization on the most downstream gauge (TODO ref).

Spatially uniform optimization
''''''''''''''''''''''''''''''

We consider here for optimization:

- a global minimization algorithm :math:`\mathrm{SBS}`,
- a single :math:`\mathrm{NSE}` objective function from discharge time series at the most downstream gauge ``V3524010``,
- a spatially uniform parameter set :math:`\theta = \left( \mathrm{cp, cft, lr, exc} \right)^T` with :math:`\mathrm{cp}` being the maximum capacity of the production reservoir, :math:`\mathrm{cft}` being the maximum capacity of the transfer reservoir, :math:`\mathrm{lr}` being the linear routing parameter and :math:`\mathrm{exc}` being the non-conservative exchange parameter.

Call the :meth:`.Model.optimize` method, fill in the arguments ``algorithm``, ``jobs_fun``, ``control_vector`` and for the sake of computation time, set the maximum number of iterations in the ``options`` argument to 2. 

.. ipython:: python

    model_su = model.optimize(
        algorithm="sbs",
        jobs_fun="nse",
        control_vector=["cp", "cft", "lr", "exc"],
        options={"maxiter": 2},
    )

While the optimization routine is in progress, some information are provided.

.. code-block:: text

    </> Optimize Model J
        Algorithm: 'sbs'
        Jobs function: 'nse'
        Nx: 1
        Np: 4 [ cp cft exc lr ] 
        Ns: 0 [  ] 
        Ng: 1 [ V3524010 ] 
        wg: 1 [ 1.000000 ] 
     
        At iterate      0    nfg =     1    J =  0.677410    ddx = 0.64
        At iterate      1    nfg =    30    J =  0.130159    ddx = 0.64
        At iterate      2    nfg =    59    J =  0.044362    ddx = 0.32
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT
        
This information is reminiscent of what we have entered in optimization options:

- ``Algorithm``: the minimization algorithm,
- ``Jobs_fun``: the objective function,
- ``Nx``: the dimension of the problem (1 means that we perform a spatially uniform optimization),
- ``Np``: the number of parameters to optimize and their name,
- ``Ns``: the number of initial states to optimize and their name,
- ``Ng``: the number of gauges to optimize and their code/name,
- ``wg``: the weight assigned to each optimized gauge.

.. note::

    The size of the control vector is defined by :math:`Nx \left(Np + Ns \right)`
    
Then, for each iteration, we can retrieve:

- ``nfg``: the total number of function and gradient evaluations (there is no gradient evaluations in the minimization algorithm :math:`\mathrm{SBS}`),
- ``J``: the value of the cost function,
- ``ddx``: the convergence criterion specific to the minimization algorithm :math:`\mathrm{SBS}` (the algorithm converges when ``ddx`` is lower than 0.01).

The last line informs about the reason why the optimization ended. Here, since we have forced 2 iterations maximum, the algorithm stopped because the number of iterations was exceeded.

Once the optimization is complete. We can visualize the simulated discharge,

.. ipython:: python

    plt.plot(model_su.input_data.qobs[0,:], label="Observed discharge");
    plt.plot(model_su.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    @savefig qsim_su_rc_user_guide.png
    plt.legend();
    
The cost function value :math:`J` (should be equal to the last iteration ``J``),

.. ipython:: python

    model_su.output.cost
    
The optimized parameters :math:`\hat{\theta}` (for the sake of clarity and because we performed a spatially uniform optimization, we will only display the parameter set values for one cell wihtin the catchment active cells, which is the most downstream gauge position here),

.. ipython:: python

    (
     model_su.parameters.cp[20, 27],
     model_su.parameters.cft[20, 27],
     model_su.parameters.lr[20, 27],
     model_su.parameters.exc[20, 27],
    )

Spatially distributed optimization
''''''''''''''''''''''''''''''''''

We consider here for optimization:

- a gradient descent minimization algorithm :math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}`,
- a single :math:`\mathrm{NSE}` objective function from discharge time series at the most downstream gauge ``V3524010``,
- a spatially distributed parameter set :math:`\theta = \left( \mathrm{cp, cft, lr, exc} \right)^T` with :math:`\mathrm{cp}` being the maximum capacity of the production reservoir, :math:`\mathrm{cft}` being the maximum capacity of the transfer reservoir, :math:`\mathrm{lr}` being the linear routing parameter and :math:`\mathrm{exc}` being the non-conservative exchange parameter.
- a prior set of parameters :math:`\bar{\theta}^*` generated from the previous spatially uniform global optimization.

Call the :meth:`.Model.optimize` method, fill in the arguments ``algorithm``, ``jobs_fun``, ``control_vector`` and for the sake of computation time, set the maximum number of iterations in the ``options`` argument to 10.

As we run this optimization from the previously generated uniform parameter set, we apply the :meth:`.Model.optimize` method from the ``model_su`` instance which had stored the previous optimized parameters.

.. ipython:: python

    model_sd = model_su.optimize(
        algorithm="l-bfgs-b",
        jobs_fun="nse",
        control_vector=["cp", "cft", "lr", "exc"],
        options={"maxiter": 10},
    )
    
While the optimization routine is in progress, some information are provided.

.. code-block:: text
    
    </> Optimize Model J
        Algorithm: 'l-bfgs-b'
        Jobs function: 'nse'
        Jreg function: 'prior'
        wJreg: .000000
        Nx: 383
        Np: 4 [ cp cft exc lr ] 
        Ns: 0 [  ] 
        Ng: 1 [ V3524010 ] 
        wg: 1 [ 1.000000 ] 
     
        At iterate      0    nfg =     1    J =  0.044362    |proj g| =  0.000000
        At iterate      1    nfg =     2    J =  0.044120    |proj g| =  0.000144
        At iterate      2    nfg =     3    J =  0.039256    |proj g| =  0.000075
        At iterate      3    nfg =     4    J =  0.038600    |proj g| =  0.000088
        At iterate      4    nfg =     5    J =  0.035663    |proj g| =  0.000019
        At iterate      5    nfg =     7    J =  0.034913    |proj g| =  0.000011
        At iterate      6    nfg =     8    J =  0.033650    |proj g| =  0.000010
        At iterate      7    nfg =     9    J =  0.032127    |proj g| =  0.000013
        At iterate      8    nfg =    10    J =  0.031194    |proj g| =  0.000011
        At iterate      9    nfg =    11    J =  0.028002    |proj g| =  0.000085
        At iterate     10    nfg =    12    J =  0.027122    |proj g| =  0.000024
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT
        
        
The information are broadly similar to the spatially uniform optimization, except for

- ``Jreg_function``: the regularization function,
- ``wJreg``: the weight assigned to the regularization term,

.. note::
    
    We did not specified any regularization options. Therefore, the ``wJreg`` term is set to 0 and no regularization is applied to the optimization.
    
Then, for each iteration, we can retrieve same information with ``nfg`` (there are gradients evaluations for the :math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}` algorithm) and ``J``.
``|proj g|`` is the infinity norm of the projected gradient.

.. note::
    
    The cost function :math:`J` at 0\ :sup:`th` iteration is equal to the cost function at the end of the spatially uniform optimization. This means that we used the previous optimized parameters as new prior.

The algorithm also stopped because the number of iterations was exceeded.

We can once again visualize, the simulated discharges (``su``: spatially uniform, ``sd``: spatially distributed)

.. ipython:: python

    plt.plot(model_sd.input_data.qobs[0,:], label="Observed discharge");
    plt.plot(model_su.output.qsim[0,:], label="Simulated discharge - su");
    plt.plot(model_sd.output.qsim[0,:], label="Simulated discharge - sd");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    @savefig qsim_sd_rc_user_guide.png
    plt.legend();
    
.. note::
    
    The difference between the two simulated discharges is very slight. Indeed, the spatially uniform optimization already leads to rather good performances with a cost function :math:`J` equal to 0.04.
    Spatially distributed optimization only improved the performances by approximately 0.02.
    
The cost function value :math:`J`,

.. ipython:: python

    model_sd.output.cost
    
The optimized parameters :math:`\hat{\theta}`,
    
.. ipython:: python

    ma = (model_sd.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, model_sd.parameters.cp)
    ma_cft = np.where(ma, np.nan, model_sd.parameters.cft)
    ma_lr = np.where(ma, np.nan, model_sd.parameters.lr)
    ma_exc = np.where(ma, np.nan, model_sd.parameters.exc)
    
    f, ax = plt.subplots(2, 2)
    
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    
    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");
    
    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");
    
    map_exc = ax[1,1].imshow(ma_exc);
    @savefig theta_sd_rc_user_guide.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/h)");

------------    
Getting data
------------

Finally and as a remainder of the :ref:`user_guide.practice_case`, it is possible to save any :class:`.Model` object to HDF5. Here, we will save both optimized instances.

.. ipython:: python

    smash.save_model(model_su, "su_optimize_Cance.hdf5")
    smash.save_model(model_sd, "sd_optimize_Cance.hdf5")
    
And reload them as follows

.. ipython:: python

    model_su_reloaded = smash.read_model("su_optimize_Cance.hdf5")
    model_sd_reloaded = smash.read_model("sd_optimize_Cance.hdf5")
    
    model_su, model_su_reloaded
    
    model_sd, model_sd_reloaded
