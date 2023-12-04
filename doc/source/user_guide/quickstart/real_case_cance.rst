.. _user_guide.quickstart.real_case_cance:

=================
Real case - Cance
=================

A real case is considered: ``the Cance river catchment at Sarras``, a right bank tributary of the Rhône river. 

.. image:: ../../_static/real_case_cance_catchment.png
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
    from smash.factory import load_dataset, generate_mesh
    from smash.io import save_setup, read_setup, save_mesh, read_mesh 
    from smash.io import save_model, read_model
    import numpy as np
    import matplotlib.pyplot as plt

---------------------   
Model object creation
---------------------

Creating a :class:`.Model` requires two input arguments: ``setup`` and ``mesh``. For this case, it is possible to directly load the both input dictionnaries using the :meth:`smash.factory.load_dataset` method, lastly imported. 

.. ipython:: python

    setup, mesh = load_dataset("Cance")
    
.. _user_guide.quickstart.real_case_cance.setup_argument:

Setup argument
**************
    
``setup`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary Fortran arrays). 

.. ipython:: python

    setup
    
To get into the details:

- ``hydrological_model``: the model structure,

- ``dt``: the calculation time step in s,

- ``start_time``: the beginning of the simulation,

- ``end_time``: the end of the simulation,

- ``read_qobs``: whether or not to read observed discharges files,

- ``qobs_directory``: the path to the observed discharges files (this path is automatically generated when you load the data),

- ``read_prcp``: whether or not to read precipitation files,

- ``prcp_format``: the precipitation files format (``tif`` format is the only available at the moment),

- ``prcp_conversion_factor``: the precipitation conversion factor (the precipitation value will be **multiplied** by the conversion factor),

- ``prcp_directory``: the path to the precipitation files (this path is automatically generated when you load the data),

- ``read_pet``: whether or not to read potential evapotranspiration files,

- ``pet_format``: the potential evapotranspiration files format (``tif`` format is the only available at the moment),

- ``pet_conversion_factor``: the potential evapotranspiration conversion factor (the potential evapotranspiration value will be **multiplied** by the conversion factor),

- ``daily_interannual_pet``: whether or not to read potential evapotranspiration files as daily interannual value desaggregated to the corresponding time step ``dt``,

- ``pet_directory``: the path to the potential evapotranspiration files (this path is automatically generated when you load the data),

- ``read_descriptor``: whether or not to read catchment descriptors files,

- ``descriptor_name``: the names of the descriptors (the name must correspond to the name of the file without the extension such as ``slope.tif``),

- ``descriptor_directory``: the path to the catchment descriptors files (this path is automatically generated when you load the data),

.. note::
    
    - See the User Guide section: :ref:`user_guide.others.model_structure` for more information about model structure
    - See the User Guide section: :ref:`user_guide.others.model_input_data_convention` for more information about model input data convention
    
    Each key and associated values that can be passed into the ``setup`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.others.model_initialization.setup>`.
    
Compared to the :ref:`user_guide.quickstart.practice_case`, more options have been filled in the ``setup`` dictionary.

.. _user_guide.quickstart.real_case_cance.mesh_argument:

Mesh argument
*************

Mesh composition
''''''''''''''''

``mesh`` is a dictionary that allows to initialize :class:`.Model` (i.e. allocate the necessary Fortran arrays). 

.. note::
    
    Each key and associated values that can be passed into the ``mesh`` dictionary are detailed in the User Guide section: :ref:`Model initialization <user_guide.others.model_initialization.mesh>`.
    
.. ipython:: python

    mesh.keys()
    
To get into the details:

- ``xres``: the computation horizontal space step in m,

- ``yres``: the computation vertical space step in m,

.. ipython:: python
    
    mesh["xres"], mesh["yres"]

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
        
- ``gauge_pos``: the gauges position in the grid,

.. ipython:: python
    
    mesh["gauge_pos"]
    
- ``flwdir``: the flow directions,

.. ipython:: python
    
    plt.imshow(mesh["flwdir"]);
    plt.colorbar(label="Flow direction (D8)");
    @savefig user_guide.quickstart.real_case_cance.flwdir.png
    plt.title("Real case - Cance - Flow direction");
    
- ``flwacc``: the flow accumulation in number of cells,

.. ipython:: python
    
    plt.imshow(mesh["flwacc"]);
    plt.colorbar(label="Flow accumulation (nb cells)");
    @savefig user_guide.quickstart.real_case_cance.flwacc.png
    plt.title("Real case - Cance - Flow accumulation");
    
- ``flwdst``: the flow distances from the main outlet in m,

.. ipython:: python
    
    plt.imshow(mesh["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig user_guide.quickstart.real_case_cance.flwdst.png
    plt.title("Real case - Cance - Flow distance");
    
- ``active_cell``: the cells that contribute to any gauge discharge (mask),

.. ipython:: python
    
    plt.imshow(mesh["active_cell"]);
    plt.colorbar(label="Logical active cell (0: False, 1: True)");
    @savefig user_guide.quickstart.real_case_cance.active_cell.png
    plt.title("Real case - Cance - Active cell");
    

Obviously, the data set included in the ``mesh`` dictionary is not generated by hand. The method :meth:`smash.generate_mesh` allows from a flow directions file, the gauge coordinates and the area to generate this same data set. More details can be found in the User Guide section: :ref:`user_guide.in_depth.automatic_meshing`.

Generate a mesh automatically
'''''''''''''''''''''''''''''

The method required the path to the flow directions ``tif`` file. One can load it directly with,

.. ipython:: python

    flwdir = load_dataset("flwdir")
    
    flwdir
    
This path leads to a flow directions ``tif`` file of the whole France at 1km spatial resolution and Lambert93 projection (*EPSG:2154*)

Get the gauge coordinates, area and code (this data is considered to be known by the user at the time the mesh is generated):

.. ipython:: python
    
    x = [840_261, 826_553, 828_269]
    
    y = [6_457_807, 6_467_115, 6_469_198]
    
    area = [381.7 * 1e6, 107 * 1e6, 25.3 * 1e6]
    
    code = ["V3524010", "V3515010", "V3517010"]
    
The ``x`` and ``y`` coordinates of the gauge must be in the same projection of the flow directions used for the meshing, here Lambert93 (*EPSG:2154*). The ``area`` must be in **m²**.

Call the :meth:`smash.generate_mesh` method:

.. ipython:: python

    mesh2 = generate_mesh(
        flwdir_path=flwdir,
        x=x,
        y=y,
        area=area,
        code=code,
    )
    
This ``mesh2`` created is a dictionnary which is identical to the ``mesh`` loaded with the :meth:`smash.load_dataset` method.

.. ipython:: python

    mesh2["xres"], mesh2["yres"], mesh2["nrow"], mesh2["ncol"], mesh2["ng"], mesh2["gauge_pos"]
    
As a remainder, the ``mesh`` can be saved to HDF5 using the :meth:`smash.save_mesh` method and reload with the :meth:`smash.read_mesh` method.

.. ipython:: python

    save_mesh(mesh2, "mesh_Cance.hdf5")
    
    mesh3 = read_mesh("mesh_Cance.hdf5")
    
    mesh3["xres"], mesh3["yres"], mesh3["nrow"], mesh3["ncol"], mesh3["ng"], mesh3["gauge_pos"]
    
Finally, create the :class:`.Model` object using the ``setup`` and ``mesh`` loaded.

.. ipython:: python

    model = smash.Model(setup, mesh)
    
    model
    
-------------
Viewing Model 
-------------

Similar to the :ref:`user_guide.quickstart.practice_case`, it is possible to visualize what the :class:`.Model` contains through the 6 attributes: :attr:`.Model.setup`, :attr:`.Model.mesh`, 
:attr:`.Model.parameters`, :attr:`.Model.states`, :attr:`.Model.output`. As we have already detailed in the :ref:`user_guide.quickstart.practice_case` the access to any data, we will only visualize the observed discharges and the spatialized atmospheric forcings here.

Input Data - Observed discharge
*******************************

3 gauges were placed in the meshing. For the sake of clarity, only the most downstream gauge discharge ``V3524010`` is plotted.

.. ipython:: python
    
    plt.plot(model.response_data.q[0,:]);
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge ($m^3/s$)");
    @savefig user_guide.quickstart.real_case_cance.qobs.png
    plt.title(model.mesh.code[0]);
    
Input Data - Atmospheric forcings
*********************************

Precipitation and potential evapotranspiration files were read for each time step. For the sake of clarity, only one precipiation grid at time step 1200 is plotted.

.. ipython:: python

    plt.imshow(model.atmos_data.prcp[..., 1200]);
    plt.title("Precipitation at time step 1200");
    @savefig user_guide.quickstart.real_case_cance.prcp.png
    plt.colorbar(label="Precipitation ($mm/h$)");
    
    
It is possible to mask the precipitation grid to only visualize the precipitation on active cells using numpy method ``np.where``.

.. ipython:: python

    ma_prcp = np.where(
        model.mesh.active_cell == 0,
        np.nan,
        model.atmos_data.prcp[..., 1200]
    )
    
    plt.imshow(ma_prcp);
    plt.title("Masked precipitation at time step 1200");
    @savefig user_guide.quickstart.real_case_cance.ma_prcp.png
    plt.colorbar(label="Precipitation ($mm/h$)");

---
Run 
--- 

Forward run
***********

Make a forward run using the :meth:`.Model.forward_run()` method.

.. ipython:: python

    model.forward_run();
    
Here, unlike the :ref:`user_guide.quickstart.practice_case`, we have not specified ``inplace=True``. By default, this argument is assigned to False, i.e. when the :meth:`.Model.run` method is called, the model object is not modified in-place but in a copy which can be returned.
So if we display the representation of the model, the last update will still be ``Initialization`` and no simulated discharge is available.

.. ipython:: python
    
    model
    
    model.response.q
    
This argument is useful to keep the original :class:`.Model` and store the results of the forward run in an other instance.
    
We can visualize the simulated discharges after a forward run for the most downstream gauge.

.. ipython:: python

    plt.plot(model.response_data.q[0,:], label="Observed discharge");
    plt.plot(model.response.q[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model.mesh.code[0]);
    @savefig user_guide.quickstart.real_case_cance.qsim_forward.png
    plt.legend();

.. _quickstart.cance.optimization:

Optimization
************

Let us briefly formulate here the general hydrological model calibration inverse problem. Let :math:`J \left( \theta \right)` be a cost function measuring the misfit between simulated and
observed quantities, such as discharge. Note that :math:`J` depends on the sought parameter set :math:`\theta` throught the hydrological model :math:`\mathcal{M}`. An optimal estimate of 
:math:`\hat{\theta}` of model parameter set is obtained from the condition:

.. math::
    
    \hat{\theta} = \underset{\theta}{\mathrm{argmin}} \; J\left( \theta \right)
    
Several calibration strategies are available in `smash`. They are based on different optimization algorithms and are for example adapted to inverse problems of various complexity, including high dimensional ones.
For the purposes of the User Guide, we will only perform a spatially uniform and distributed optimization on the most downstream gauge.

Spatially uniform optimization
''''''''''''''''''''''''''''''

We consider here for optimization (which is the default setup with ``gr-a`` structure):

- a global minimization algorithm :math:`\mathrm{SBS}`,
- a single :math:`\mathrm{NSE}` objective function from discharge time series at the most downstream gauge ``V3524010``,
- a spatially uniform parameter set :math:`\theta = \left( \mathrm{cp, cft, lr, exc} \right)^T` with :math:`\mathrm{cp}` being the maximum capacity of the production reservoir, :math:`\mathrm{cft}` being the maximum capacity of the transfer reservoir, :math:`\mathrm{lr}` being the linear routing parameter and :math:`\mathrm{exc}` being the non-conservative exchange parameter.

Call the :meth:`.Model.optimize` method and for the sake of computation time, set the maximum number of iterations in the ``options`` argument to 2. 

.. ipython:: python
    :suppress:

    res = model.optimize(optimize_options={"termination_crit":{"maxiter": 2}}, 
        return_options={"cost": True, "iter_cost": True});
    
    model_su = model
    
.. ipython:: python
    :verbatim:

    res = model.optimize(optimize_options={"termination_crit":{"maxiter": 2}}, 
        return_options={"cost": True, "iter_cost": True});

While the optimization routine is in progress, some information are provided.

.. code-block:: text

    </> Optimize
        At iterate      0    nfg =     1    J =  0.677404    ddx = 0.64
        At iterate      1    nfg =    30    J =  0.130012    ddx = 0.64
        At iterate      2    nfg =    59    J =  0.043658    ddx = 0.32
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT
        
This information remainds the ptimization options:

- ``Mapping``: the optimization mapping of parameters,
- ``Algorithm``: the minimization algorithm,
- ``Jobs_fun``: the objective function(s),
- ``wJobs``: the weight assigned to each objective function,
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

    plt.plot(model.response_data.q[0,:], label="Observed discharge");
    plt.plot(model.response.q[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model.mesh.code[0]);
    @savefig user_guide.quickstart.real_case_cance.qsim_su.png
    plt.legend();

The cost function value :math:`J` (should be equal to the last iteration ``J``),

.. ipython:: python

    res.cost
    res.iter_cost

    
The optimized parameters :math:`\hat{\theta}` (for the sake of clarity and because we performed a spatially uniform optimization, we will only display the parameter set values for one cell within the catchment active cells, which is the most downstream gauge position here),

.. ipython:: python

    cp = model.get_rr_parameters("cp")
    ct = model.get_rr_parameters("ct")
    llr = model.get_rr_parameters("llr")
    kexc = model.get_rr_parameters("kexc")

    ind = tuple(model.mesh.gauge_pos[0,:])
    
    ind
    
    (
    cp[ind],
    ct[ind],
    llr[ind],
    kexc[ind],
    )

As a remainder of the :ref:`user_guide.quickstart.practice_case`, it is possible to save any :class:`.Model` object to HDF5. Here, we will save the uniform optimized instances for a future displaying.

.. ipython:: python

    save_model(model, "model.hdf5")
    model_su = read_model("model.hdf5")


    
Spatially distributed optimization
''''''''''''''''''''''''''''''''''

We consider here for optimization:

- a gradient descent minimization algorithm :math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}`,
- a single :math:`\mathrm{NSE}` objective function from discharge time series at the most downstream gauge ``V3524010``,
- a spatially distributed parameter set :math:`\theta = \left( \mathrm{cp, cft, lr, exc} \right)^T` with :math:`\mathrm{cp}` being the maximum capacity of the production reservoir, :math:`\mathrm{cft}` being the maximum capacity of the transfer reservoir, :math:`\mathrm{lr}` being the linear routing parameter and :math:`\mathrm{exc}` being the non-conservative exchange parameter.
- a prior set of parameters :math:`\bar{\theta}^*` generated from the previous spatially uniform global optimization.

Call the :meth:`.Model.optimize` method, fill in the arguments ``mapping`` with "distributed" and for the sake of computation time, set the maximum number of iterations in the ``options`` argument to 15.

As we run this optimization from the previously generated uniform parameter set, we apply the :meth:`.Model.optimize` method from the ``model`` instance which had stored the previous optimized parameters.

.. ipython:: python
    :suppress:
    
    res = model.optimize(
            mapping="distributed",
            optimize_options={"termination_crit":{"maxiter": 15}},
            return_options={"cost": True, "iter_cost": True},
        )
    

.. ipython:: python
    :verbatim:
    
    res = model.optimize(
            mapping="distributed",
            optimize_options={"termination_crit":{"maxiter": 15}},
            return_options={"cost": True, "iter_cost": True},
        )
    

While the optimization routine is in progress, some information are provided.

.. code-block:: text
    
    </> Optimize
        At iterate      0    nfg =     1    J =      0.052158    |proj g| =      0.003706
        At iterate      1    nfg =     3    J =      0.046520    |proj g| =      0.034254
        At iterate      2    nfg =     4    J =      0.045327    |proj g| =      0.018078
        At iterate      3    nfg =     6    J =      0.044052    |proj g| =      0.013705
        At iterate      4    nfg =     7    J =      0.039764    |proj g| =      0.031789
        At iterate      5    nfg =     8    J =      0.037352    |proj g| =      0.020424
        At iterate      6    nfg =    10    J =      0.032467    |proj g| =      0.028793
        At iterate      7    nfg =    11    J =      0.030155    |proj g| =      0.065370
        At iterate      8    nfg =    12    J =      0.024909    |proj g| =      0.007977
        At iterate      9    nfg =    13    J =      0.023737    |proj g| =      0.017989
        At iterate     10    nfg =    14    J =      0.022914    |proj g| =      0.007681
        At iterate     11    nfg =    15    J =      0.022013    |proj g| =      0.008621
        At iterate     12    nfg =    17    J =      0.020395    |proj g| =      0.011236
        At iterate     13    nfg =    18    J =      0.018925    |proj g| =      0.027226
        At iterate     14    nfg =    19    J =      0.018672    |proj g| =      0.033382
        At iterate     15    nfg =    20    J =      0.018326    |proj g| =      0.003987
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
    
    plt.plot(model.response_data.q[0,:], label="Observed discharge");
    plt.plot(model_su.response.q[0,:], label="Simulated discharge - su");
    plt.plot(model.response.q[0,:], label="Simulated discharge - sd");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model.mesh.code[0]);
    @savefig user_guide.quickstart.real_case_cance.qsim_sd.png
    plt.legend();
    
.. note::
    
    The difference between the two simulated discharges is very slight. Indeed, the spatially uniform optimization already leads to rather good performances with a cost function :math:`J` equal to 0.04.
    Spatially distributed optimization only improved the performances by approximately 0.02.
    
The cost function value :math:`J`,

.. ipython:: python

    res.cost
    
The optimized parameters :math:`\hat{\theta}`,
    
.. ipython:: python

    ma = (model.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, cp)
    ma_ct = np.where(ma, np.nan, ct)
    ma_llr = np.where(ma, np.nan, llr)
    ma_kexc = np.where(ma, np.nan, kexc)
    
    f, ax = plt.subplots(2, 2)
    
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    
    map_ct = ax[0,1].imshow(ma_ct);
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    
    map_llr = ax[1,0].imshow(ma_llr);
    f.colorbar(map_llr, ax=ax[1,0], label="llr (min)");
    
    map_kexc = ax[1,1].imshow(ma_kexc);
    @savefig user_guide.quickstart.real_case_cance.theta.png
    f.colorbar(map_kexc, ax=ax[1,1], label="kexc (mm/h)");
