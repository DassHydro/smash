.. _user_guide.others.gr6:

=============
Genie Rural 6
=============

In this tutorial, we will look at the GR6 conceptual model dedicated to low-flow simulations presented in :cite:p:`michel2003` and :cite:p:`pushpalatha`.
The GR6 model is based on an augmentation of the GR5 model by adding an exponential reservoir to better capture recession phenomena.
:cite:p:`michel2003` points out that the expression for the flow leaving the reservoir is often badly treated 
in a continuous manner using expression :

.. math::
    
    \begin{eqnarray}
        q(x, t) = \exp\left(\frac{h_e(x, t)}{t_e(x)}\right)
    \end{eqnarray}

with :math:`t_e` controls the water level recession (:math:`mm`) and :math:`h_e` (:math:`mm`) is the level of the reservoir.
:cite:p:`michel2003` suggests considering the continuity equation :math:`dh_e = -q ds` over a given period, between two time steps :math:`t^*` and :math:`t`.

.. math::

    & \int_{t^*}^{t} \exp\left(\frac{-h_e(x, s)}{t_e(x)}\right) dh_e = -\int_{t^*}^{t} ds 
    
    & - t_e(x) \left( \exp \left( \frac{-h_e(x, t)}{t_e(x)} \right) - \exp \left( \frac{-h_e(x, t^*)}{t_e(x)} \right) \right) = t - t^*
        
We can express the loss of water between :math:`t^*` and  :math:`t` as :math:`h_e(x, t) = h_e(x, t^*) - q_{e}(x, t)`.

.. math::

    & \exp \left( \frac{-h_e(x, t^*) + q_{e}(x,t)}{t_e(x)} \right) - \exp \left( \frac{-h_e(x, t^*)}{t_e(x)} \right) = \frac{t^* - t}{t_e(x)}
    
    & \exp \left( \frac{q_{e}}{t_e(x)} \right) = 1 + \frac{t^* - t}{t_e(x)} \exp \left( \frac{h_e(x, t^*)}{t_e(x)} \right)


If we choose the initial condition carefully, we can express the loss of water (mm) as :

.. math::

    q_{e}(x,t) = t_e(x) \ln \left( 1 + \exp \left( \frac{h_e(x, t^*)}{t_e(x)} \right) \right).

In :cite:p:`michel2003`, they suggest some treatments of the asymptotic behaviors of the quantity :math:`q_{e}` and take over in the GR6 module from airGR package https://hydrogr.github.io/airGR as follow :

.. math::
    :nowrap:

    \begin{eqnarray}

        q_{e}(x, t) =
        \begin{cases}
            
            t_e(x) \ln \left( 1 + \exp \left( \frac{h_e(x, t^*)}{t_e(x)} \right) \right) &\text{if} \; -7 \lt \frac{h_e(x, t^*)}{t_e(x)} \lt 7 \\

            t_e(x) * \exp \left( \frac{h_e(x, t^*)}{t_e(x)} \right) &\text{if} \; \frac{h_e(x, t^*)}{t_e(x)} \lt -7 \\

            h_e(x, t^*) + \frac{ t_e(x) }{ \exp \left( \frac{h_e(x, t^*)}{t_e(x)} \right) } \; &\text{otherwise}.

        \end{cases}

    \end{eqnarray}

One can find details :ref:`hydrological module gr6 <math_num_documentation.forward_structure.hydrological_module.gr6>`.


Imports
-------

.. ipython:: python

    import smash
    from smash.io import save_model, read_model
    from smash.factory import load_dataset
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


Run and optimize
----------------

The Cance is modelled successively with GR5 and GR6. Then we optimize a first uniform guest with the lbfgs-b algorithm to obtain a distributed mapping.

.. ipython:: python
    :suppress:
    
    setup, mesh = load_dataset("Cance")
    models = ['gr5', 'gr6']
    for m in models:
        setup['hydrological_module']=m
        model = smash.Model(setup, mesh)        
        model.forward_run();
        save_model(model, "model_{}.hdf5".format(m))
        res = model.optimize(
            optimize_options={"termination_crit":{"maxiter": 2}},
            return_options={"cost": True, "iter_cost": True}
            );
        res = model.optimize(
                mapping="distributed",
                optimize_options={"termination_crit":{"maxiter": 3}},
                return_options={"cost": True, "iter_cost": True},
            )
        save_model(model, "model_sd_{}.hdf5".format(m))

.. code-block:: text
    
    </> Reading precipitation: 100%|████████████████████████████████████████████████████████████████████████████████████| 1440/1440 [00:02<00:00, 529.85it/s]
    </> Reading daily interannual pet: 100%|█████████████████████████████████████████████████████████████████████████████| 366/366 [00:00<00:00, 2126.80it/s]
    </> Disaggregating daily interannual pet: 100%|██████████████████████████████████████████████████████████████████| 1440/1440 [00:00<00:00, 130435.11it/s]
    </> Computing mean atmospheric data
    </> Adjusting GR interception capacity
    </> Forward Run
    </> Optimize
        At iterate      0    nfg =     1    J =      0.695010    ddx = 0.64
        At iterate      1    nfg =    47    J =      0.069483    ddx = 0.64
        At iterate      2    nfg =    94    J =      0.039312    ddx = 0.16
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

    </> Optimize
        At iterate      0    nfg =     1    J =      0.039311    |proj g| =      0.001498
        At iterate      1    nfg =     3    J =      0.037930    |proj g| =      0.015125
        At iterate      2    nfg =     4    J =      0.037024    |proj g| =      0.010977
        At iterate      3    nfg =     5    J =      0.036018    |proj g| =      0.029318
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

    </> Reading precipitation: 100%|████████████████████████████████████████████████████████████████████████████████████| 1440/1440 [00:02<00:00, 534.58it/s]
    </> Reading daily interannual pet: 100%|█████████████████████████████████████████████████████████████████████████████| 366/366 [00:00<00:00, 2127.16it/s]
    </> Disaggregating daily interannual pet: 100%|██████████████████████████████████████████████████████████████████| 1440/1440 [00:00<00:00, 134116.40it/s]
    </> Computing mean atmospheric data
    </> Adjusting GR interception capacity
    </> Forward Run
    </> Optimize
        At iterate      0    nfg =     1    J =      0.805968    ddx = 0.64
        At iterate      1    nfg =    68    J =      0.101309    ddx = 0.64
        At iterate      2    nfg =   136    J =      0.037820    ddx = 0.16
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

    </> Optimize
        At iterate      0    nfg =     1    J =      0.037820    |proj g| =      0.039416
        At iterate      1    nfg =     4    J =      0.035688    |proj g| =      0.057545
        At iterate      2    nfg =     5    J =      0.034707    |proj g| =      0.030183
        At iterate      3    nfg =     6    J =      0.033398    |proj g| =      0.034014
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT


Post-processing
---------------

This part is only dedicated to the post-processing. First, for sake of clarity, we load the last models.

.. ipython:: python

    model_gr5 = read_model("model_gr5.hdf5")
    model_gr6 = read_model("model_gr6.hdf5")
    model_sd_gr5 = read_model("model_sd_gr5.hdf5")
    model_sd_gr6 = read_model("model_sd_gr6.hdf5")

Discharge run
*************

We compare the output discharges of GR6 and GR5 models.

.. ipython:: python

    plt.plot(model_gr5.response_data.q[0,:], label="Observed discharge");
    plt.plot(model_gr5.response.q[0,:], label="GR5");
    plt.plot(model_gr6.response.q[0,:], label="GR6", color='m');
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.legend();
    @savefig user_guide.others.gr6.forward.png
    plt.title("Forward run at station {}".format(model_gr5.mesh.code[0]));

We can show NSE and KGE errors on the distributed solution.

.. ipython:: python

    metrics = ["nse", "kge"]
    perf_gr5 = smash.evaluation(model_gr5, metric=metrics)
    perf_gr6 = smash.evaluation(model_gr6, metric=metrics)

    perf = pd.DataFrame(index=["GR5", "GR6"], columns=metrics)
    for m in metrics:
        perf.loc["GR5", m] = smash.evaluation(model_gr5, metric=m)[0]
    for m in metrics:
        perf.loc["GR6", m] = smash.evaluation(model_gr6, metric=m)[0]
    perf 

By direct computation, GR5 is better than GR6 comparing the NSE and KGE. 

Optimized discharge
*******************

We compare the optimized discharge of GR6 and GR5 models, the metrics.

.. ipython:: python

    plt.figure()
    plt.plot(model_gr5.response_data.q[0,:], label="Observed discharge");
    plt.plot(model_sd_gr5.response.q[0,:], label="Optimized GR5 - sd");
    plt.plot(model_sd_gr6.response.q[0,:], label="Optimized GR6 - sd", color='m');
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.legend();
    @savefig user_guide.others.gr6.optimize.png
    plt.title("Distributed optimization");

.. ipython:: python

    metrics = ["nse", "kge"]
    perf_gr5 = smash.evaluation(model_gr5, metric=metrics)
    perf_gr6 = smash.evaluation(model_gr6, metric=metrics)

    perf = pd.DataFrame(index=["GR5", "GR6"], columns=metrics)
    for m in metrics:
        perf.loc["GR5", m] = smash.evaluation(model_sd_gr5, metric=m)[0]
    for m in metrics:
        perf.loc["GR6", m] = smash.evaluation(model_sd_gr6, metric=m)[0]
    perf 

The GR6 model doesn't perform the result on the case of Cance : the run of GR5 is better
for KGE metric and similar for NSE metric.  

Parameters mapping
******************

Finally, we can draw the parameter maps.

.. ipython:: python

    f, ax =  plt.subplots(2, 3, figsize=(15, 8))
    map_cp = ax[0,0].imshow(model_sd_gr5.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)"); 
    map_ct = ax[0,1].imshow(model_sd_gr5.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    ax[0,2].remove()
    map_kexc = ax[1,0].imshow(model_sd_gr5.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_aexc = ax[1,1].imshow(model_sd_gr5.get_rr_parameters("aexc"));
    f.colorbar(map_aexc, ax=ax[1,1], label="aexc");
    map_llr = ax[1,2].imshow(model_sd_gr5.get_rr_parameters("llr"));
    f.colorbar(map_llr, ax=ax[1,2], label="llr (min)");
    @savefig user_guide.others.gr6.gr5_parameters.png
    f.suptitle('GR5', fontsize=20)
    
    f, ax = plt.subplots(2, 3, figsize=(15, 8))
    map_cp = ax[0,0].imshow(model_sd_gr6.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_sd_gr6.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_te = ax[0,2].imshow(model_sd_gr6.get_rr_parameters("te"));
    f.colorbar(map_te, ax=ax[0,2], label="te (mm)");
    map_kexc = ax[1,0].imshow(model_sd_gr6.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_aexc = ax[1,1].imshow(model_sd_gr6.get_rr_parameters("aexc"));
    f.colorbar(map_aexc, ax=ax[1,1], label="aexc");
    map_llr = ax[1,2].imshow(model_sd_gr6.get_rr_parameters("llr"));
    f.colorbar(map_llr, ax=ax[1,2], label="llr (min)");
    @savefig user_guide.others.gr6.gr6_parameters.png
    f.suptitle('GR6', fontsize=20)
