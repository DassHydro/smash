.. _user_guide.classical_uses.lez_regionalization:

=====================
Lez - Regionalization
=====================

This guide on `smash` will be carried out on the French catchment, **the Lez at Lattes** and aims to perform an optimization of the
hydrological model considering regional mapping of physical descriptors onto model conceptual parameters.
The parameters :math:`\boldsymbol{\theta}` can be written as a mapping :math:`\phi` of descriptors :math:`\boldsymbol{\mathcal{D}}`
(slope, drainage density, soil water storage, etc) and :math:`\boldsymbol{\rho}` a control vector:
:math:`\boldsymbol{\theta}(x)=\phi\left(\boldsymbol{\mathcal{D}}(x),\boldsymbol{\rho}\right)`.
See the :ref:`math_num_documentation.mapping` section for more details.

First, a shape is assumed for the mapping (here **multi-polynomial** or **neural network**).
Then the control vector of the mapping needs to be optimized: :math:`\boldsymbol{\hat{\rho}}=\underset{\mathrm{\boldsymbol{\rho}}}{\text{argmin}}\;J`,
with :math:`J` the cost function.

.. image:: ../../_static/lez.png
    :width: 400
    :align: center

Required data
-------------

.. hint::

    It is the same dataset as the :ref:`user_guide.quickstart.lez_split_sample_test` study, so possibly no need to re-download it.

You need first to download all the required data.

.. button-link:: https://smash.recover.inrae.fr/dataset/Lez-dataset.tar
    :color: primary
    :shadow:
    :align: center

    **Download**

If the download was successful, a file named ``Lez-dataset.tar`` should be available. We can switch to the directory where this file has been 
downloaded and extract it using the following command:

.. code-block:: shell

    tar xf Lez-dataset.tar

Now a folder called ``Lez-dataset`` should be accessible and contain the following files and folders:

- ``France_flwdir.tif``
    A GeoTiff file containing the flow direction data,
- ``gauge_attributes.csv``
    A csv file containing the gauge attributes (gauge coordinates, drained area and code),
- ``prcp``
    A directory containing precipitation data in GeoTiff format with the following directory structure: ``%Y/%m`` 
    (``2012/08``),
- ``pet``
    A directory containing daily interannual potential evapotranspiration data in GeoTiff format,
- ``qobs``
    A directory containing the observed discharge data in csv format,
- ``descriptor``
    A directory containing physiographic descriptors in GeoTiff format.

Six physical descriptors are considered in this example, which are:

.. image:: ../../_static/physio_descriptors.png
    :align: center

.. TODO: Add descriptor explanation

We can open a Python interface. The current working directory will be assumed to be the directory where
the ``Lez-dataset`` is located.

Open a Python interface:

.. code-block:: shell

    python3

.. ipython:: python
    :suppress:

    import os
    os.system("python3 generate_dataset.py -d Lez")

Imports
-------

We will first import everything we need in this tutorial.

.. ipython:: python

    import smash
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

Model creation
--------------

Model setup creation
********************

.. ipython:: python

    setup = {
        "start_time": "2012-08-01",
        "end_time": "2013-07-31",
        "dt": 86_400, # daily time step
        "hydrological_module": "gr4", 
        "routing_module": "lr",
        "read_prcp": True, 
        "prcp_directory": "./Lez-dataset/prcp", 
        "read_pet": True,  
        "pet_directory": "./Lez-dataset/pet",
        "read_qobs": True,
        "qobs_directory": "./Lez-dataset/qobs",
        "read_descriptor": True,
        "descriptor_directory": "./Lez-dataset/descriptor",
        "descriptor_name": [
            "slope",
            "drainage_density",
            "karst",
            "woodland",
            "urban",
            "soil_water_storage"
        ]
    }

Model mesh creation
*******************

.. ipython:: python

    gauge_attributes = pd.read_csv("./Lez-dataset/gauge_attributes.csv")

    mesh = smash.factory.generate_mesh(
        flwdir_path="./Lez-dataset/France_flwdir.tif",
        x=list(gauge_attributes["x"]),
        y=list(gauge_attributes["y"]),
        area=list(gauge_attributes["area"] * 1e6), # Convert km² to m²
        code=list(gauge_attributes["code"]),
    )

Then, we can initialize the `smash.Model` object

.. ipython:: python

    model = smash.Model(setup, mesh)

Model simulation
----------------

Multiple polynomial
*******************

To optimize the rainfall-runoff model using a multiple polynomial mapping of descriptors to conceptual model parameters,
the value ``multi-polynomial`` simply needs to be passed to the mapping argument. We add another option to limit the number of iterations
by stopping the optimizer after ``50`` iterations.

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model_mp = smash.optimize(
        model,
        mapping="multi-polynomial",
        optimize_options={
            "termination_crit": dict(maxiter=50),
        },
        common_options={"ncpu": ncpu},
    )

.. ipython:: python
    :verbatim:

    model_mp = smash.optimize(
        model,
        mapping="multi-polynomial",
        optimize_options={
            "termination_crit": dict(maxiter=50),
        },
    )

We have therefore optimized the set of rainfall-runoff parameters using a multiple polynomial regression constrained by
physiographic descriptors. Here, most of the options used are the default ones, i.e. a minimization of one minus the Nash-Sutcliffe
efficiency on the most downstream gauge of the domain. The resulting rainfall-runoff parameter maps can be viewed.

.. ipython:: python

    f, ax = plt.subplots(2, 2)

    map_cp = ax[0,0].imshow(model_mp.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_mp.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_kexc = ax[1,0].imshow(model_mp.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_llr = ax[1,1].imshow(model_mp.get_rr_parameters("llr"));
    @savefig user_guide.classical_uses.lez_regionalization.mp_theta.png
    f.colorbar(map_llr, ax=ax[1,1], label="llr (min)");

As well as performances at upstream gauges

.. ipython:: python
    
    metrics = ["nse", "kge"]
    upstream_perf = pd.DataFrame(index=model.mesh.code[1:], columns=metrics)

    for m in metrics:
        upstream_perf[m] = np.round(smash.metrics(model_mp, metric=m)[1:], 2)

    upstream_perf

.. note::
    The two upstream gauges are the two last gauges of the list. This is why we use ``[1:]`` in the lists in order to take all the gauges
    except the first, which is the downstream gauge on which the model has been calibrated.

Artificial neural network
*************************

We can optimize the rainfall-runoff model using a neural network (NN) based mapping of descriptors to conceptual model parameters.
It is possible to define your own network to implement this optimization, but here we willl use the default neural network.
Similar to multiple polynomial mapping, all you have to do is to pass the value, ``ann`` to the ``mapping`` argument.
We also pass other options specific to the use of a NN:

- ``optimize_options``
    - ``random_state``: a random seed used to initialize neural network weights.
    - ``learning_rate``: the learning rate used for weights updates during training.
    - ``termination_crit``: the number of training ``epochs`` for the neural network and a positive number to stop training when the loss function does not decrease below the current optimal value for  ``early_stopping`` consecutive ``epochs``

- ``return_options``
    - ``net``: return the optimized neural network

.. To speed up documentation generation
.. ipython:: python
    :suppress:

    ncpu = min(5, max(1, os.cpu_count() - 1))
    model_ann, opt_ann = smash.optimize(
        model,
        mapping="ann",
        optimize_options={
            "random_state": 23,
            "learning_rate": 0.004,
            "termination_crit": dict(epochs=100, early_stopping=20),
        },
        return_options={"net": True},
        common_options={"ncpu": ncpu},
    )

.. ipython:: python
    :verbatim:

    model_ann, opt_ann = smash.optimize(
        model,
        mapping="ann",
        optimize_options={
            "random_state": 23,
            "learning_rate": 0.004,
            "termination_crit": dict(epochs=100, early_stopping=20),
        },
        return_options={"net": True},
    )
.. note::
    As we used the `smash.optimize` method (here an :ref:`ADAM algorithm <math_num_documentation.optimization_algorithm>` by default when choosing a NN based mapping) and asked for optional return values, this function will return two values, the optimized model
    ``model_ann`` and the optional returns ``opt_ann``.

Since we have returned the optimized neural network, we can visualize what it contains

.. ipython:: python

    opt_ann.net

The information displayed tells us that the default neural network is composed of 2 hidden dense layers followed by ``ReLU`` activation functions 
and a final layer followed by a ``Sigmoid`` function. To scale the network output to the boundary condition, a ``MinMaxScale`` function is applied. 
Other information is available in the `smash.factory.Net` object, including the value of the cost function at each iteration.

.. ipython:: python

    plt.plot(opt_ann.net.history["loss_train"]);
    plt.xlabel("Epoch");
    plt.ylabel("$1-NSE$");
    plt.grid(alpha=.7, ls="--");
    @savefig user_guide.classical_uses.lez_regionalization.ann_J.png
    plt.title("Cost function descent");

Finally, we can visualize parameters and performances

.. ipython:: python

    f, ax = plt.subplots(2, 2)

    map_cp = ax[0,0].imshow(model_ann.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_ann.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_kexc = ax[1,0].imshow(model_ann.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_llr = ax[1,1].imshow(model_ann.get_rr_parameters("llr"));
    @savefig user_guide.classical_uses.lez_regionalization.ann_theta.png
    f.colorbar(map_llr, ax=ax[1,1], label="llr (min)");

.. ipython:: python
    
    metrics = ["nse", "kge"]
    upstream_perf = pd.DataFrame(index=model.mesh.code[1:], columns=metrics)

    for m in metrics:
        upstream_perf[m] = np.round(smash.metrics(model_ann, metric=m)[1:], 2)

    upstream_perf

.. ipython:: python
    :suppress:

    plt.close('all')
