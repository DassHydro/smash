.. _user_guide.optimize.pre_regio_poly:

============================================
Pre-regionalization using polynomial mapping
============================================

Here, we aim to employ some physiographic descriptors to find the pre-regionalization mapping using polynomial functions. 
Six descriptors are considered in this example, which are:

- Slope
- Drainage density
- Karst
- Woodland
- Urban
- Soil water storage

First, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt

---------------------
Model object creation
---------------------

To perform the calibrations, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Lez`` dataset.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)

----------------------------
Visualization of descriptors
----------------------------

This method requires input descriptors, which were provided during the creation of the Model object. 
We can visualize these descriptors and verify if they were successfully loaded:

.. ipython:: python

    model.input_data.descriptor.shape

.. ipython:: python

    desc_name = model.setup.descriptor_name
    fig, axes = plt.subplots(1, len(desc_name), figsize=(12,4), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_title(desc_name[i]);
        im = ax.imshow(model.input_data.descriptor[..., i]);
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal");
        cbar.ax.tick_params();
    @savefig desc_optimize_poly_user_guide.png
    fig.suptitle("Physiographic descriptors");

.. ipython:: python

    # Reset figsize to the Matplotlib default
    plt.figure(figsize=plt.rcParamsDefault['figure.figsize']);

-----------------------------
Finding a uniform first guess
-----------------------------

Similar to the :ref:`fully-distributed optimization <user_guide.optimize.bayes_estimate>` method, 
providing a uniform first guess is recommended for this method. 
In this case, we use the :meth:`smash.Model.bayes_estimate` method to find a suitable first guess:

.. ipython:: python

    model_ufg = model.bayes_estimate(k=np.linspace(-2, 12, 50), n=200, random_state=0)

In the example above, we generated a set of 200 random spatially uniform model parameters, and 
used the L-curve approach to find an optimal regularization parameter within a search range of :math:`[-2, 12]`.

.. hint::

    Check out the :ref:`Bayesian estimation <user_guide.optimize.bayes_estimate>` section for an example of 
    how to improve the first guess using the Bayesian estimation approach.

----------------------------------------------------------
Optimizing hyperparameters for pre-regionalization mapping
----------------------------------------------------------

There are two types of polynomial mapping that can be employed for pre-regionalization:

- ``hyper-linear``: a linear mapping where the hyperparameters to be estimated are the coefficients.
- ``hyper-polynomial``: a polynomial mapping where the hyperparameters to be estimated are the coefficients and the degree.

As an example, the hyper-polynomial mapping can be combined with the variational calibration algorithm ``L-BFGS-B`` as shown below:

.. ipython:: python
    :suppress:

    model_hp = model_ufg.optimize(
            mapping="hyper-polynomial", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python
    :verbatim:

    model_hp = model_ufg.optimize(
            mapping="hyper-polynomial", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

Some information are also provided during the optimization:

.. code-block:: text

    </> Optimize Model
        Mapping: 'hyper-polynomial' k(x) = a0 + a1 * D1 ** b1 + ... + an * Dn ** bn
        Algorithm: 'l-bfgs-b'
        Jobs function: [ nse ]
        wJobs: [ 1.0 ]
        Jreg function: 'prior'
        wJreg: 0.000000
        Nx: 1
        Np: 52 [ cp cft exc lr ]
        Ns: 0 [  ]
        Ng: 1 [ Y3204040 ]
        wg: 1 [ 1.0 ]

        At iterate      0    nfg =     1    J =  0.450582    |proj g| =  0.000000
        At iterate      1    nfg =     2    J =  0.384291    |proj g| =  0.206485
        At iterate      2    nfg =     4    J =  0.292124    |proj g| =  0.341891
        At iterate      3    nfg =     6    J =  0.256965    |proj g| =  0.359353
        At iterate      4    nfg =     7    J =  0.195863    |proj g| =  0.151185
        At iterate      5    nfg =     9    J =  0.189970    |proj g| =  0.042805
        At iterate      6    nfg =    10    J =  0.188656    |proj g| =  0.057944
        At iterate      7    nfg =    11    J =  0.188190    |proj g| =  0.032824
        At iterate      8    nfg =    12    J =  0.187963    |proj g| =  0.018767
        At iterate      9    nfg =    14    J =  0.185999    |proj g| =  0.058310
        At iterate     10    nfg =    15    J =  0.176684    |proj g| =  0.205041
        At iterate     11    nfg =    16    J =  0.165673    |proj g| =  0.547557
    ...
        At iterate     28    nfg =    37    J =  0.134749    |proj g| =  0.003044
        At iterate     29    nfg =    38    J =  0.134725    |proj g| =  0.001460
        At iterate     30    nfg =    39    J =  0.134716    |proj g| =  0.004003
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT

------------------------
Visualization of results
------------------------

Now we can visualize the simulated discharge:

.. ipython:: python

    qo = model_hp.input_data.qobs[0,:].copy()
    qo = np.where(qo<0, np.nan, qo)  # to deal with missing data
    plt.plot(qo, label="Observed discharge");
    plt.plot(model_hp.output.qsim[0,:], label="Simulated discharge");
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.title(model_hp.mesh.code[0]);
    @savefig qsim_optimize_pre-regio_hp_user_guide.png
    plt.legend();

The cost value:

.. ipython:: python

    model_hp.output.cost

And finally, the spatially distributed model parameters constrained by physiographic descriptors:

.. ipython:: python

    ma = (model_hp.mesh.active_cell == 0)

    ma_cp = np.where(ma, np.nan, model_hp.parameters.cp)
    ma_cft = np.where(ma, np.nan, model_hp.parameters.cft)
    ma_lr = np.where(ma, np.nan, model_hp.parameters.lr)
    ma_exc = np.where(ma, np.nan, model_hp.parameters.exc)
    
    f, ax = plt.subplots(2, 2)
    
    map_cp = ax[0,0].imshow(ma_cp);
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    
    map_cft = ax[0,1].imshow(ma_cft);
    f.colorbar(map_cft, ax=ax[0,1], label="cft (mm)");
    
    map_lr = ax[1,0].imshow(ma_lr);
    f.colorbar(map_lr, ax=ax[1,0], label="lr (min)");
    
    map_exc = ax[1,1].imshow(ma_exc);
    @savefig theta_sd_optimize_pre-regio_hp_user_guide.png
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/h)");