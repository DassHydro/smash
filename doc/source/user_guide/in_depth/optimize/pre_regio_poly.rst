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
In this case, we use the SBS algorithm to find such a first guess:

.. ipython:: python

    model_su = model.optimize(mapping="uniform", algorithm="sbs", options={"maxiter": 2});

.. hint::

    You may want to refer to the :ref:`Bayesian estimation <user_guide.optimize.bayes_estimate>` section 
    for information on how to improve the first guess using a Bayesian estimation approach.

----------------------------------------------------------
Optimizing hyperparameters for pre-regionalization mapping
----------------------------------------------------------

There are two types of polynomial mapping that can be employed for pre-regionalization:

- ``hyper-linear``: a linear mapping where the hyperparameters to be estimated are the coefficients.
- ``hyper-polynomial``: a polynomial mapping where the hyperparameters to be estimated are the coefficients and the degree.

As an example, the hyper-polynomial mapping can be combined with the variational calibration algorithm ``L-BFGS-B`` as shown below:

.. ipython:: python
    :suppress:

    model_hp = model_su.optimize(
            mapping="hyper-polynomial", 
            algorithm="l-bfgs-b", 
            options={"maxiter": 30}
        )

.. ipython:: python
    :verbatim:

    model_hp = model_su.optimize(
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

        At iterate      0    nfg =     1    J =  0.176090    |proj g| =  0.000000
        At iterate      1    nfg =     3    J =  0.174870    |proj g| =  0.160574
        At iterate      2    nfg =     4    J =  0.173283    |proj g| =  0.059085
        At iterate      3    nfg =     5    J =  0.172243    |proj g| =  0.043317
        At iterate      4    nfg =     6    J =  0.171181    |proj g| =  0.045926
        At iterate      5    nfg =     7    J =  0.170460    |proj g| =  0.023084
        At iterate      6    nfg =     8    J =  0.169568    |proj g| =  0.025826
        At iterate      7    nfg =     9    J =  0.168186    |proj g| =  0.046616
        At iterate      8    nfg =    10    J =  0.165931    |proj g| =  0.069842
        At iterate      9    nfg =    11    J =  0.160961    |proj g| =  0.077036
        At iterate     10    nfg =    13    J =  0.152905    |proj g| =  0.209049
        At iterate     11    nfg =    14    J =  0.148905    |proj g| =  0.056703
    ...
        At iterate     28    nfg =    34    J =  0.133811    |proj g| =  0.008069
        At iterate     29    nfg =    35    J =  0.133753    |proj g| =  0.003290
        At iterate     30    nfg =    36    J =  0.133749    |proj g| =  0.001082
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
    f.colorbar(map_exc, ax=ax[1,1], label="exc (mm/d)");