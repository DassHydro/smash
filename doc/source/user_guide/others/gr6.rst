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
        q_{re}(x, t) = \exp\left(\frac{h_e(x, t)}{c_e(x)}\right)
    \end{eqnarray}

with :math:`c_e` capacity of the reservoir and the level is :math:`h_e`. We assume that :math:`c_e` is always positive and :math:`h_e` always negative.
:cite:p:`michel2003` suggests considering the continuity equation :math:`dh_e = -q_{re}ds` over a given period, between two time steps :math:`t^*` and :math:`t`.

.. math::

    & \int_{t^*}^{t} \exp\left(\frac{-h_e(x, s)}{c_e(x)}\right) dh_e = -\int_{t^*}^{t} ds 
    
    & - c_e(x) \left( \exp \left( \frac{-h_e(x, t)}{c_e(x)} \right) - \exp \left( \frac{-h_e(x, t^*)}{c_e(x)} \right) \right) = t - t^*
        
We can express the loss of water between :math:`t^*` and  :math:`t` as :math:`h_e(x, t) = h_e(x, t^*) - h(x,t)`.

.. math::

    & \exp \left( \frac{-h_e(x, t^*) + h(x,t)}{c_e(x)} \right) - \exp \left( \frac{-h_e(x, t^*)}{c_e(x)} \right) = \frac{t^* - t}{c_e(x)}
    
    & \exp \left( \frac{h}{c_e(x)} \right) = 1 + \frac{t^* - t}{c_e(x)} \exp \left( \frac{h_e(x, t^*)}{c_e(x)} \right)


If we choose the initial condition carefully, we can express the loss of water as :math:`h(x,t) = c_e(x) \ln \left( 1 + \exp \left( \frac{h_e(x, t^*)}{c_e(x)} \right) \right)`.
Note that :math:`-h` is always negative and :math:`h_e=0` is the maximum.

.. math::

    q_{re}(x, t) = \max \left( 0, h_e(x, t^*)c_e(x) - \ln\left(1 + \exp \left( \frac{h_e(x, t^*)}{c_e(x)} \right) \right) c_e(x) \right)

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
    models = ['gr5','gr6']
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
                optimize_options={"termination_crit":{"maxiter": 15}},
                return_options={"cost": True, "iter_cost": True},
            )
        save_model(model, "model_sd_{}.hdf5".format(m))

.. code-block:: text
    
    </> Reading precipitation: 100%|████████████████████| 1440/1440 [00:00<00:00, 4591.17it/s]
    </> Reading daily interannual pet: 100%|█████████████| 366/366 [00:00<00:00, 12317.18it/s]
    </> Disaggregating daily interannual pet: 100%|███| 1440/1440 [00:00<00:00, 129417.77it/s]
        At iterate      0    nfg =     1    J =      0.643190    ddx = 0.64
        At iterate      1    nfg =    47    J =      0.097397    ddx = 0.32
        At iterate      2    nfg =    94    J =      0.048932    ddx = 0.08
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

        At iterate      0    nfg =     1    J =      0.048932    |proj g| =      0.079021
        At iterate      1    nfg =     4    J =      0.045529    |proj g| =      0.041243
        At iterate      2    nfg =     5    J =      0.043874    |proj g| =      0.017285
        At iterate      3    nfg =     6    J =      0.043481    |proj g| =      0.014298
        At iterate      4    nfg =     8    J =      0.042662    |proj g| =      0.015985
        At iterate      5    nfg =     9    J =      0.041588    |proj g| =      0.009900
        At iterate      6    nfg =    10    J =      0.040576    |proj g| =      0.030573
        At iterate      7    nfg =    11    J =      0.039574    |proj g| =      0.019104
        At iterate      8    nfg =    12    J =      0.038581    |proj g| =      0.029094
        At iterate      9    nfg =    13    J =      0.037825    |proj g| =      0.065368
        At iterate     10    nfg =    14    J =      0.036845    |proj g| =      0.017084
        At iterate     11    nfg =    15    J =      0.035724    |proj g| =      0.038028
        At iterate     12    nfg =    16    J =      0.034238    |proj g| =      0.036330
        At iterate     13    nfg =    17    J =      0.031792    |proj g| =      0.058857
        At iterate     14    nfg =    18    J =      0.029198    |proj g| =      0.024985
        At iterate     15    nfg =    19    J =      0.027338    |proj g| =      0.027682
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

    </> Reading precipitation: 100%|████████████████████| 1440/1440 [00:00<00:00, 4682.68it/s]
    </> Reading daily interannual pet: 100%|█████████████| 366/366 [00:00<00:00, 12533.80it/s]
    </> Disaggregating daily interannual pet: 100%|███| 1440/1440 [00:00<00:00, 130570.46it/s]
        At iterate      0    nfg =     1    J =      0.503706    ddx = 0.64
        At iterate      1    nfg =    68    J =      0.211769    ddx = 0.64
        At iterate      2    nfg =   134    J =      0.053807    ddx = 0.64
        STOP: TOTAL NO. OF ITERATION EXCEEDS LIMIT                                                                                      

        At iterate      0    nfg =     1    J =      0.053808    |proj g| =      0.099369
        At iterate      1    nfg =     5    J =      0.046207    |proj g| =      0.154723
        At iterate      2    nfg =     6    J =      0.043945    |proj g| =      0.115107
        At iterate      3    nfg =     7    J =      0.040459    |proj g| =      0.029903
        At iterate      4    nfg =     8    J =      0.039962    |proj g| =      0.031887
        At iterate      5    nfg =     9    J =      0.038804    |proj g| =      0.044005
        At iterate      6    nfg =    10    J =      0.037164    |proj g| =      0.043613
        At iterate      7    nfg =    11    J =      0.035314    |proj g| =      0.046803
        At iterate      8    nfg =    12    J =      0.034041    |proj g| =      0.036070
        At iterate      9    nfg =    13    J =      0.031954    |proj g| =      0.042313
        At iterate     10    nfg =    14    J =      0.030495    |proj g| =      0.045343
        At iterate     11    nfg =    15    J =      0.029325    |proj g| =      0.039053
        At iterate     12    nfg =    16    J =      0.028579    |proj g| =      0.038041
        At iterate     13    nfg =    17    J =      0.028112    |proj g| =      0.030944
        At iterate     14    nfg =    18    J =      0.027698    |proj g| =      0.014870
        At iterate     15    nfg =    19    J =      0.027394    |proj g| =      0.028152
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

We compare the output discharges of GR6 and GR5 models... ADD COMMENTS

.. ipython:: python

    plt.plot(model_gr5.response_data.q[0,:], label="Observed discharge");
    plt.plot(model_gr5.response.q[0,:], label="GR5");
    plt.plot(model_gr6.response.q[0,:], label="GR6", color='m');
    plt.grid(alpha=.7, ls="--");
    plt.xlabel("Time step");
    plt.ylabel("Discharge $(m^3/s)$");
    plt.legend();
    @savefig user_guide.others.gr6.forward.png
    plt.title("Run at station {}".format(model_gr5.mesh.code[0]));

We can show NSE and KGE errors on the distributed solution.

.. ipython:: python

    metrics = ["nse", "kge"]
    perf = pd.DataFrame(index=["GR5", "GR6"], columns=metrics)
    for m in metrics:
        perf.loc["GR5", m] = smash.metrics(model_gr5, metric=m)[0]
    for m in metrics:
        perf.loc["GR6", m] = smash.metrics(model_gr6, metric=m)[0]
    perf 


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
    perf = pd.DataFrame(index=["GR5", "GR6"], columns=metrics)
    for m in metrics:
        perf.loc["GR5", m] = smash.metrics(model_sd_gr5, metric=m)[0]
    for m in metrics:
        perf.loc["GR6", m] = smash.metrics(model_sd_gr6, metric=m)[0]
    perf 

The GR6 model doesn't perform the result on the case of Cance : the run of GR5 is better
and the values for metrics of optimized discharges are similar.  

Parameters mapping
******************

Finally, we can draw the parameter map. The capacity of the reservoir exponential is in the range of values exhibit by :cite:p:`pushpalatha`, between 5 mm and 10 mm.

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
    f.colorbar(map_aexc, ax=ax[1,1], label="aexc (mm/d)");
    map_llr = ax[1,2].imshow(model_sd_gr5.get_rr_parameters("llr"));
    f.colorbar(map_llr, ax=ax[1,2], label="llr (min)");
    @savefig user_guide.others.gr6.gr5_parameters.png
    f.suptitle('GR5', fontsize=20)
    
    f, ax = plt.subplots(2, 3, figsize=(15, 8))
    map_cp = ax[0,0].imshow(model_sd_gr6.get_rr_parameters("cp"));
    f.colorbar(map_cp, ax=ax[0,0], label="cp (mm)");
    map_ct = ax[0,1].imshow(model_sd_gr6.get_rr_parameters("ct"));
    f.colorbar(map_ct, ax=ax[0,1], label="ct (mm)");
    map_ce = ax[0,2].imshow(model_sd_gr6.get_rr_parameters("ce"));
    f.colorbar(map_ce, ax=ax[0,2], label="ce (mm)");
    map_kexc = ax[1,0].imshow(model_sd_gr6.get_rr_parameters("kexc"));
    f.colorbar(map_kexc, ax=ax[1,0], label="kexc (mm/d)");
    map_aexc = ax[1,1].imshow(model_sd_gr6.get_rr_parameters("aexc"));
    f.colorbar(map_aexc, ax=ax[1,1], label="aexc (mm/d)");
    map_llr = ax[1,2].imshow(model_sd_gr6.get_rr_parameters("llr"));
    f.colorbar(map_llr, ax=ax[1,2], label="llr (min)");
    @savefig user_guide.others.gr6.gr6_parameters.png
    f.suptitle('GR6', fontsize=20)
