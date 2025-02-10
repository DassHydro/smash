.. _user_guide.in_depth.multicriteria_calibration:

==========================
Multi-criteria Calibration
==========================

This section provides a detailed guide on optimizing a model using multiple calibration metrics.
The multiple calibration metrics are based on hydrological signatures :ref:`Math / Num Documentation <math_num_documentation.hydrological_signature>`.

First, open a Python interface:

.. code-block:: none

    python3

Imports
*******

We will first import the necessary libraries for this tutorial.

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import pandas as pd

Model object creation
*********************

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.lez` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)
    
Multiple metrics calibration using signatures
*********************************************

The `smash` optimization algorithms enable the use of a cost function composed of multiple calibration metrics.
First, the observation at a gauge :math:`g` is defined as the sum of several metrics as:

.. math::

    J_{obs, g} = \sum w_c j_c

with :math:`j_c` and :math:`w_c` being a specific metric and the associated weight, respectively.
For more details, see the :ref:`Math / Num Documentation <math_num_documentation.cost_function>` section.

Note that this multi-criteria approach is possible for the optimization methods :meth:`smash.Model.optimize` and :meth:`smash.Model.bayesian_optimize`. 
For simplicity, in this example, we use :meth:`smash.Model.optimize` for single gauge optimization with a uniform mapping.

First, we establish a reference benchmark using the default cost options with a single-objective function.

.. ipython:: python

    # We set the max number of iterations to 5 for example
    optmize_options = {"termination_crit": {"maxiter": 5}}

    model_1 = smash.optimize(model, optimize_options=optmize_options)

The default evaluation metric :math:`j_c` is the Nash-Sutcliffe efficiency (NSE).

In addition to NSE, we now perform a multi-criteria optimization using two other metrics: 
the relative absolute error based on the continuous runoff coefficient (Crc) and the relative absolute error of the peak flow (Epf) for multi-criteria calibration.

.. ipython:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Crc", "Epf"],
        "wjobs_cmpt": [0.6, 0.1, 0.3],
    }
    model_2 = smash.optimize(model, cost_options=cost_options, optimize_options=optmize_options)

where the weights of the objective functions :math:`w_c` based on NSE, Crc, and Epf are set to 0.6, 0.1, and 0.3 respectively. 
If these weights are not provided by the user, they are equal by default and their sum equals 1, hence the cost value is computed as the mean of the objective functions.

.. code-block:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Crc", "Epf"],
        "wjobs_cmpt": "mean",  # default value using alias 'mean'
    }

For multiple metrics based on flood-event signatures, these metrics are computed using flood event signatures computed from an automatic segmentation algorithm (see the tutorial on :ref:`segmentation algorithm <user_guide.classical_uses.hydrograph_segmentation>`).
The parameters of this algorithm, which utilizes rainfall and discharge signals, can be adjusted.
For example, consider a calibration using a multi-criteria cost function based on NSE and the flood flow (Eff) metric, with respective weights of 0.6 and 0.4, where the segmentation criterion is set to exceed a peak threshold of 0.9.

.. ipython:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Eff"],
        "event_seg": {"peak_quant": 0.9},
        "wjobs_cmpt": [0.6, 0.4],
    }

    model_3 = smash.optimize(model,
        cost_options=cost_options,
        optimize_options=optmize_options,
    )

Now, we compute the Nash-Sutcliffe error for the calibrated gauge of each model.

.. ipython:: python
          
    models = [model_1, model_2, model_3]
    nse = []
    for m in models:
        nse.append(smash.evaluation(m, metric='nse')[0][0])

Then, we compute the observed and simulated signatures for each model.

.. ipython:: python

    models = [model_1, model_2, model_3]
    signatures_obs = []
    signatures_sim = []
    for m in models:
        signatures_obs.append(smash.signatures(m, sign=['Crc', 'Epf', 'Eff']))
        signatures_sim.append(smash.signatures(m, sign=['Crc', 'Epf', 'Eff'], domain='sim'))

For simplicity, we arange the signatures by type.

.. ipython:: python

    crc_obs = []
    epf_obs = []
    eff_obs = []
    for sign in signatures_obs:
        crc_obs.append(sign.cont.iloc[0]['Crc'])
        epf_obs.append(sign.event.iloc[0]['Epf'])
        eff_obs.append(sign.event.iloc[0]['Eff'])

    crc_sim = []
    epf_sim = []
    eff_sim = []
    for sign in signatures_sim:
        crc_sim.append(sign.cont.iloc[0]['Crc'])
        epf_sim.append(sign.event.iloc[0]['Epf'])
        eff_sim.append(sign.event.iloc[0]['Eff'])

We compute the relative absolute error for each signatures.

.. ipython:: python

    RAE_Crc = [abs(sim / obs - 1) for (sim, obs) in zip(crc_sim, crc_obs)]
    RAE_Epf = [abs(sim / obs - 1) for (sim, obs) in zip(epf_sim, epf_obs)]
    RAE_Eff = [abs(sim / obs - 1) for (sim, obs) in zip(eff_sim, eff_obs)]

Finally, we group the metric information together as follows:

.. ipython:: python
        
    metric_info = {
        'NSE': nse,
        'RAE_Crc': RAE_Crc,
        'RAE_Epf': RAE_Epf,
        'RAE_Eff': RAE_Eff,
    }

    index = ["model_1 (NSE)", "model_2 (NSE, Crc, Epf)", "model_3 (NSE, Eff)"]

    df = pd.DataFrame(metric_info, index=index)
    df

.. ipython:: python
    :suppress:

    plt.close('all')
