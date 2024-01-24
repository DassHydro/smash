.. _user_guide.classical_uses.regionalization:

Regionalization
***************

The parameters :math:`\boldsymbol{\theta}` and initial states :math:`\boldsymbol{h_0}` can be written as a mapping :math:`\phi` of descriptors :math:`\boldsymbol{\mathcal{D}}` (slope, drainage density, soil water storage...) and :math:`\boldsymbol{\rho}` a control vector: :math:`\left(\boldsymbol{\theta}(x),\boldsymbol{h}_{0}(x)\right)=\phi\left(\boldsymbol{\mathcal{D}}(x,t),\boldsymbol{\rho}\right)`.

First, a shape is assumed for the mapping (here polynomial or neural network).
Then we search to optimize the control vector of the mapping: :math:`\boldsymbol{\hat{\rho}}=\underset{\mathrm{\boldsymbol{\rho}}}{\text{argmin}}J`, with :math:`J` the cost function. See the section :ref:`Mapping Math / Num Documentation <math_num_documentation.mapping>` for more details.

==================
Polynomial mapping
==================

Here, we aim to employ some physiographic descriptors to find the pre-regionalization mapping using polynomial functions. 
Six descriptors are considered in this example, which are:

.. image:: ../../_static/physio_descriptors.png
    :align: center

| 
| First, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python

    import smash
    import matplotlib.pyplot as plt
    import numpy as np
    from smash.factory import load_dataset
    from smash.factory import Net


---------------------
Model object creation
---------------------

To perform the calibrations, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python
            
    setup, mesh = load_dataset("Lez")
    model = smash.Model(setup, mesh)
    
----------------------------
Visualization of descriptors
----------------------------
We can plot the descriptors, as the beginning :

.. code-block:: none

    desc_name = model.setup.descriptor_name
    fig, axes = plt.subplots(1, len(desc_name), figsize=(12,4), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_title(desc_name[i])
        im = ax.imshow(model.physio_data.descriptor[..., i])
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
        cbar.ax.tick_params()
    fig.suptitle("Physiographic descriptors");

.. ipython:: python

    # Reset figsize to the Matplotlib default
    plt.figure(figsize=plt.rcParamsDefault['figure.figsize']);

-----------------------------
Finding a uniform first guess
-----------------------------

Similar to the :ref:`fully-distributed optimization <user_guide.in_depth.optimize.fully_distributed>` method, 
providing a uniform first guess is recommended for this method. 
In this case, we use the :math:`\mathrm{SBS}` algorithm to find such a first guess:

.. ipython:: python

    model.optimize(
        mapping="uniform",
        optimizer="sbs",
        optimize_options={"termination_crit":{"maxiter": 2}},
        );

.. hint::

    You may want to refer to the :ref:`Bayesian estimation <user_guide.in_depth.optimize.bayes_estimate>` section 
    for information on how to improve the first guess using a Bayesian estimation approach.

----------------------------------------------------------
Optimizing hyperparameters for pre-regionalization mapping
----------------------------------------------------------

There are two types of polynomial mapping that can be employed for pre-regionalization:

- ``hyper-linear``: a linear mapping where the hyperparameters to be estimated are the coefficients.
- ``hyper-polynomial``: a polynomial mapping where the hyperparameters to be estimated are the coefficients and the degree.

As an example, the hyper-polynomial mapping can be combined with the variational calibration algorithm 
:math:`\mathrm{L}\text{-}\mathrm{BFGS}\text{-}\mathrm{B}` as shown below:

.. ipython:: python
    :suppress:

    res = model.optimize(
        mapping="multi-linear",
        optimizer="lbfgsb",
        optimize_options={"termination_crit":{"maxiter": 30}},
        return_options={"cost": True,},
    )

.. ipython:: python
    :verbatim:

    res = model.optimize(
        mapping="multi-polynomial",
        optimizer="lbfgsb",
        optimize_options={"termination_crit":{"maxiter": 30}},
        return_options={"cost": True,},
    )

Some information are also provided during the optimization:

.. code-block:: text

    </> Optimize
    At iterate      0    nfg =     1    J =      0.180574    |proj g| =      0.174191
    At iterate      1    nfg =     3    J =      0.178274    |proj g| =      0.057199
    At iterate      2    nfg =     4    J =      0.177763    |proj g| =      0.050301
    At iterate      3    nfg =     5    J =      0.175845    |proj g| =      0.022164
    At iterate      4    nfg =     6    J =      0.175363    |proj g| =      0.021531
    At iterate      5    nfg =     7    J =      0.172832    |proj g| =      0.029747
    At iterate      6    nfg =     8    J =      0.170395    |proj g| =      0.036075
    At iterate      7    nfg =     9    J =      0.152341    |proj g| =      0.308658
    At iterate      8    nfg =    11    J =      0.146294    |proj g| =      0.221512
    ...
    At iterate     25    nfg =    29    J =      0.134164    |proj g| =      0.009976
    At iterate     26    nfg =    30    J =      0.133877    |proj g| =      0.012379
    At iterate     27    nfg =    31    J =      0.133571    |proj g| =      0.005878
    At iterate     28    nfg =    32    J =      0.133450    |proj g| =      0.021392
    At iterate     29    nfg =    33    J =      0.133341    |proj g| =      0.009055
    At iterate     30    nfg =    34    J =      0.133130    |proj g| =      0.024293
    STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                            

------------------------
Visualization of results
------------------------

Now we can visualize the simulated discharge:

.. ipython:: python

    qobs = model.response_data.q[0,:].copy()
    qobs = np.where(qobs<0, np.nan, qobs)  # to deal with missing data
    qsim = model.response.q[0,:]
    plt.plot(qobs, label="Observed discharge");
    plt.plot(qsim, label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model.mesh.code[0]);
    @savefig user_guide.classical_uses.optimize.pre_regio_poly.qsim.png
    plt.legend();


The cost value:

.. ipython:: python

    res.cost

And finally, the spatially distributed model parameters constrained by physiographic descriptors:

.. ipython:: python

    cp = model.get_rr_parameters("cp").copy()
    ct = model.get_rr_parameters("ct").copy()
    llr = model.get_rr_parameters("llr").copy()
    kexc = model.get_rr_parameters("kexc").copy()

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
    @savefig user_guide.in_depth.optimize.pre_regio_ann.theta.png
    f.colorbar(map_kexc, ax=ax[1,1], label="kexc (mm/d)");


======================
Neural network mapping
======================
We can also find the pre-regionalization mapping using an artificial neural network. 

Let's reinitialize the model:

.. ipython:: python

    model = smash.Model(setup, mesh)
    desc_name = model.setup.descriptor_name
    n_desc = model.setup.nd


To define a custom neural network, you may need to have information about the physiographic descriptors and hydrological parameters. 
This information will be used to determine the input and output layers of the network, including the number of descriptors, 
the control vector, and the boundary condition (if you want to scale the network output to the boundary condition). 
The default values of these parameters can be obtained as follows:

.. ipython:: python

    bounds_param = model.get_rr_parameters_bounds()
    bounds_param.pop("ci")
    control_vector = list(bounds_param.keys())
    bounds = list(bounds_param.values())

Next, we need to initialize the Net object:

.. ipython:: python

    net = Net()

Then, we can define a graph for our custom neural network by specifying the number of layers, type of activation function, 
and output scaling. For example, we can define a neural network with 2 hidden dense layers followed by ``ReLU`` activation functions 
and a final layer followed by a ``sigmoid`` function. To scale the network output to the boundary condition, we apply a ``MinMaxScale`` function:

.. ipython:: python

    net.add(
        layer="dense", 
        options={
            "input_shape": (n_desc,), 
            "neurons": 48,
        },
    )
    net.add(layer="activation", options={"name": "relu"})

.. ipython:: python

    net.add(
        layer="dense", 
        options={
            "neurons": 16,
        },
    )
    net.add(layer="activation", options={"name": "relu"})

.. ipython:: python

    net.add(
        layer="dense", 
        options={
            "neurons": len(control_vector),
        },
    )
    net.add(layer="activation", options={"name": "sigmoid"})
    net.add(layer="scale", options={"bounds": bounds})

.. ipython:: python

    net  # display a summary of the network

---------------------------
Training the neural network
---------------------------

Now, we can train the neural network with the custom graph using the :meth:`smash.Model.optimize` method. 
We define the optimize options for the custom neural network.

.. ipython:: python

    optimize_options = {
        "bounds": bounds_param,
        "net": net,
        "random_state": 23,
        "learning_rate": 0.004,
        "termination_crit": dict(epochs=100, early_stopping=20),
    }

- ``bounds``
    imposes a min/max condition on parameters,

.. ipython:: python

    bounds

- ``net``
    specifies the custom neural network, if you do not specify the neural network (``net`` argument) in the :meth:`smash.Model.optimize` method, a default network will be used to learn the descriptors-to-parameters mapping. 
    
- ``random_state``
    initialize the weight of the neural network,

- ``learning_rate``
    updates the weight during the training,
    
- ``termination_crit``
    is the maximum number of iterations

- ``early_stopping``
    stops the optimization if the loss function does not decrease below the current optimal value for early_stopping consecutive epochs


.. ipython:: python
        :suppress:
    
        ann = model.optimize(
            mapping="ANN",
            optimizer="Adam",
            optimize_options=optimize_options,
            common_options={"verbose": True},
            return_options={"net": True, "cost":True,},
        )

.. ipython:: python
        :verbatim:
    
        ann = model.optimize(
            mapping="ANN",
            optimizer="Adam",
            optimize_options=optimize_options,
            common_options={"verbose": True},
            return_options={"net": True, "cost":True,},
        )

Some information are also provided during the training process:

.. code-block:: text

    Training: 100%|█████████████████████████████████████| 100/100 [00:07<00:00, 13.04it/s]
    </> Reading precipitation: 100%|████████████████████| 1440/1440 [00:00<00:00, 4775.83it/s]
    </> Reading daily interannual pet: 100%|█████████████| 366/366 [00:00<00:00, 12425.46it/s]
    </> Disaggregating daily interannual pet: 100%|███| 1440/1440 [00:00<00:00, 127209.88it/s]
        At iterate      0    nfg =     1    J =      0.643190    ddx = 0.64
        At iterate      1    nfg =    30    J =      0.097397    ddx = 0.64
        At iterate      2    nfg =    59    J =      0.052158    ddx = 0.32
        At iterate      3    nfg =    88    J =      0.043086    ddx = 0.08
        At iterate      4    nfg =   118    J =      0.040684    ddx = 0.02
        At iterate      5    nfg =   152    J =      0.040604    ddx = 0.01
        CONVERGENCE: DDX < 0.01                                                                                                         

.. note::

    By default, ``nse`` is used to define the objective function if you do not specify the ``jobs_fun`` argument 
    in :meth:`smash.Model.optimize`.

------------------------
Visualization of results
------------------------

To visualize the descent of the cost function, we use the ``net`` object and create a plot of the cost function value versus 
the number of iterations. Here's an example:

.. ipython:: python

    y = ann.net.history["loss_train"]
    x = range(1, len(y) + 1)
    plt.figure()
    plt.plot(x, y);
    plt.xlabel("Epoch");
    plt.ylabel("$1-NSE$");
    plt.grid(alpha=.7, ls="--");
    plt.title("Cost function descent");


The simulated discharge:

.. ipython:: python

    qo = model.response_data.q[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model.response.q[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model.mesh.code[0]);
    @savefig user_guide.classical_uses.optimize.pre_regio_ann.qsim.png
    plt.legend();

The cost value:

.. ipython:: python

    ann.cost

And finally, the spatially distributed model parameters constrained by physiographic descriptors:

.. ipython:: python

    cp = model.get_rr_parameters("cp").copy()
    ct = model.get_rr_parameters("ct").copy()
    llr = model.get_rr_parameters("llr").copy()
    kexc = model.get_rr_parameters("kexc").copy()
    
    ma = (model.mesh.active_cell == 0)
    ma_cp = np.where(ma, np.nan, cp)
    ma_cft = np.where(ma, np.nan, ct)
    ma_lr = np.where(ma, np.nan, llr)
    ma_exc = np.where(ma, np.nan, kexc)
    f, ax = plt.subplots(2, 2)
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");
    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");
    map_exc = ax[1,1].imshow(ma_exc);
    @savefig user_guide.in_depth.optimize.pre_regio_ann.theta.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");

.. ipython:: python
    :suppress:

    plt.close('all')

