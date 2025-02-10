.. _user_guide.in_depth.multicriteria_calibration:

==========================
Multi-criteria Calibration
==========================

This section provides a detailed guide on optimization of a model using multiple calibration metrics.
The multiple calibration metrics are based on the hydrological signatures :ref:`Math / Num Documentation <math_num_documentation.hydrological_signature>`.

First, open a Python interface:

.. code-block:: none

    python3



First, import smash and python librairies needed in this tutorial.

Imports
*******

We will import everything we need in this tutorial.

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import pandas as pd

Model object creation
*********************

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.Lez` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Lez")
    
    model = smash.Model(setup, mesh)
    
Multiple metrics calibration using signatures
*********************************************

The smash optimization algorithms enable to use a cost function composed of multiple calibration metrics.
First, the observation at a single gauge :math:`J_{obs, g}` is defined as the sum of several metrics as:

.. math::

    J_{obs, g} = \sum w_c j_c

with :math:`j_c` and :math:`w_c` a specific metric and the weight associated respectively.

Then objective function :math:`J_{obs}` agregates observations of all considered gauges:

.. math::

    J_{obs} = \sum w_g J_{obs, g}

where :math:`J_{obs, g}` and :math:`w_g` are respectively the observation term and weight associated to each gauge.
For more details, see the :ref:`Math / Num Documentation <math_num_documentation.cost_function>` section.

Note that this multi-criteria approach is possible for the optimization methods :meth:`smash.Model.optimize` and :meth:`smash.Model.bayesian_optimize`. 
For simplicity, in this example, we use :meth:`smash.Model.optimize` with a uniform mapping.

First, let's perform an optimization using the default options, which include a NSE cost function, over the three gauges as defined in the setup.

.. ipython:: python

    model1 = smash.optimize(model);

The default evaluation metric :math:`j_c` is the Nash-Sutcliffe efficiency ``nse``.
Second, in addition to NSE two other metrics are considered, the continuous Crc and the flood-event Erc runoff coefficients for multi-criteria calibration:

.. ipython:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Crc", "Erc"],
        "wjobs_cmpt": [0.6, 0.1, 0.3],
    }
    model2 = smash.optimize(model, cost_options = cost_options);

where the weights of the objective functions :math:`w_c` are based on ``nse``, ``Crc``, ``Erc`` are set to 0.6, 0.1 and 0.3 respectively. 
If these weights are not given by user, they are equal by default and their sum equals 1, hence the cost value is computed as the mean of the objective functions.

.. code-block:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Crc", "Erc"],
        "wjobs_cmpt": "mean",
    }

For multiple metrics based on flood-event signatures, those metrics are computed using flood event detected automatically with the segmentation algorithm (:ref:segmentation <user_guide.classical_uses.hydrograph_segmentation>). The parameters of this segmentation algorithm, which utilizes rainfall and discharge signals, can be adjusted.
For example, let us perform a calibration on the three gauges of model parameters using a multi-criteria cost function based on continuous NSE and peak flow Epf signature, with weights 0.6 and 0.4 respectively and segmentation criterion of exceeding peak threshold of 0.9.

.. ipython:: python

    cost_options = {
        "jobs_cmpt": ["nse", "Epf"],
        "event_seg": {"peak_quant": 0.9},
        "wjobs_cmpt": [0.6, 0.4],
    }
    model3 = smash.optimize(model,
        cost_options=cost_options,
    )

Let's compute the Nash-Sutcliffe error for the first gauge of each model.

.. ipython:: python
          
    models = [model1, model2, model3]
    nse = []
    for m in models:
        nse.append(1. - smash.evaluation(m, metric='nse')[0][0])

Let's compute the signatures for each model.

.. ipython:: python

    models = [model1, model2, model3]
    signatures_obs = []
    signatures_sim = []
    for m in models:
        signatures_obs.append(smash.signatures(m, sign=['Crc', 'Erc', 'Epf']))
        signatures_sim.append(smash.signatures(m, sign=['Crc', 'Erc', 'Epf'], domain='sim'))

For simplicity, we arange the signatures by type.

.. ipython:: python

    crc_obs = []
    erc_obs = []
    epf_obs = []
    for sign in signatures_obs:
        crc_obs.append(sign.cont.iloc[0]['Crc'])
        erc_obs.append(sign.event.iloc[0]['Erc'])
        epf_obs.append(sign.event.iloc[0]['Epf'])

    crc_sim = []
    erc_sim = []
    epf_sim = []
    for sign in signatures_sim:
        crc_sim.append(sign.cont.iloc[0]['Crc'])
        erc_sim.append(sign.event.iloc[0]['Erc'])
        epf_sim.append(sign.event.iloc[0]['Epf'])

We compute the relative error for each signatures.

.. ipython:: python

    RE_Crc = [sim / obs - 1 for (sim, obs) in zip(crc_sim, crc_obs)]
    RE_Erc = [sim / obs - 1 for (sim, obs) in zip(erc_sim, erc_obs)]
    RE_Epf = [sim / obs - 1 for (sim, obs) in zip(epf_sim, epf_obs)]

Finally, we group the metric informations together:

.. ipython:: python
        
    metric_info = {
        '1 - NSE': nse,
        'RE_Crc': RE_Crc,
        'RE_Erc': RE_Erc,
        'RE_Epf':RE_Epf,
    }

    index = ["model1 (NSE)", "model2 (NSE, Crc, Erc)", "model3 (Epf)"]

    df = pd.DataFrame(metric_info, index=index)
    df

.. ipython:: python
    :suppress:

    plt.close('all')
