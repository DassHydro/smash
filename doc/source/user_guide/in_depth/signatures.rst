.. _user_guide.signatures:

.. role:: bolditalic
    :class: bolditalic

==========
Signatures
==========

Several signatures describing and quantifying properties of discharge time series are introduced 
in view to analyze and calibrate hydrological models [WH+15]_. 
These signatures permit to describe various aspects of the rainfall-runoff behavior such as: 
flow distribution (based for instance on flow percentiles), 
flow dynamics (based for instance on base-flow separation [NT+15]_, [LM+79]_), 
flow timing, etc.. A so-called continuous signature is a signature that can be computed on the whole studied period.
Flood event signatures on the other hand focus on the behavior of the high flows 
that are observed in the flood events. 
These flood event signatures are calculated via a proposed segmentation algorithm as depicted in ??? (TODO).

.. list-table:: List of all studied signatures
   :widths: 10 25 55 10
   :header-rows: 1

   * - Notation
     - Signature
     - Description
     - Unit
   * - Crc
     - Continuous runoff coefficients
     - Coefficient relating the amount of runoff to the amount of precipitation received
     - --
   * - Crchf
     - 
     - Coefficient relating the amount of high-flow to the amount of precipitation received
     - --
   * - Crclf
     - 
     - Coefficient relating the amount of low-flow to the amount of precipitation received
     - --
   * - Crch2r
     - 
     - Coefficient relating the amount of high-flow to the amount of runoff
     - --
   * - Cfp2
     - Flow percentiles
     - 0.02-quantiles from flow duration curve
     - mm
   * - Cfp10
     -
     - 0.1-quantiles from flow duration curve
     - mm
   * - Cfp50
     -
     - 0.5-quantiles from flow duration curve
     - mm
   * - Cfp90
     -
     - 0.9-quantiles from flow duration curve
     - mm
   * - Eff
     - Flood flow
     - Amount of quickflow in flood event
     - mm
   * - Ebf
     - Base flow
     - Amount of baseflow in flood event
     - mm
   * - Erc
     - Flood event runoff coefficients
     - Coefficient relating the amount of runoff to the amount of precipitation received
     - --
   * - Erchf
     - 
     - Coefficient relating the amount of high-flow to the amount of precipitation received
     - --
   * - Erclf
     - 
     - Coefficient relating the amount of low-flow to the amount of precipitation received
     - --
   * - Erch2r
     - 
     - Coefficient relating the amount of high-flow to the amount of runoff
     - --
   * - Elt
     - Lag time
     - Difference time between the peak runoff and the peak rainfall
     - h
   * - Epf
     - Peak flow
     - Peak runoff in flood event
     - mm

.. [WH+15]

  Westerberg, I. K., and Hilary K. McMillan. "Uncertainty in hydrological signatures." Hydrology and Earth System Sciences 19.9 (2015): 3951-3968.

.. [NT+15]

  Nathan, Rory J., and Thomas A. McMahon. "Evaluation of automated techniques for base flow and recession analyses." Water resources research 26.7 (1990): 1465-1473.

.. [LM+79]

  Lyne, V., and M. Hollick. "Stochastic time-variable rainfall-runoff modelling." Institute of Engineers Australia National Conference. Vol. 79. No. 10. Barton, Australia: Institute of Engineers Australia, 1979.

This section aims to go into detail on how to compute and vizualize some hydrological signatures as well as 
the sensitivity of the model parameters to these signatures.

First, open a Python interface:

.. code-block:: none

    python3

-------
Imports
-------

.. ipython:: python
    
    import smash
    import pandas as pd

---------------------
Model object creation
---------------------

To compute the signatures, you must create a :class:`smash.Model` object. 
For this case, we will use the ``Cance`` dataset used in the User Guide section: :ref:`user_guide.real_case_cance`.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

.. _user_guide.signatures.computation:

----------------------
Signatures computation
----------------------

To start with, we need to run a direct (or optimized) simulation. Then the signatures computation result is available using the :meth:`Model.signatures() <smash.Model.signatures>` method. 
The argument ``event_seg`` (only related to flood event signatures) could be defined for tuning the parameters of segmentation algorithm.

.. ipython:: python

    model.run(inplace=True);

    res = model.signatures(event_seg={"peak_quant": 0.99});

The signatures computation result is represented as a :class:`smash.SignResult` object containning 2 attributes which are 2 different dictionaries:

- ``cont`` : Continuous signatures computation result,

- ``event``: Flood event signatures computation result.

Each dictionary has 2 keys which are 2 different pandas.DataFrame:

- ``obs``: Observation result,

- ``sim``: Simulation result.

For example, to display the simulated flood event signatures computation result.

.. ipython:: python

    res.event["sim"]

Now, we visualize, for instance, the simulated and observed continuous runoff coefficients in the boxplots below.

.. ipython:: python

    df_obs = res.cont["obs"]
    df_sim = res.cont["sim"]
    
    df = pd.concat([df_obs, df_sim], ignore_index=True)
    df["domain"] = ["obs"]*len(df_obs) + ["sim"]*len(df_sim)
    
    @savefig sign_comp.png
    boxplot = df.boxplot(column=["Crc", "Crchf", "Crclf", "Crch2r"], by="domain")

.. _user_guide.signatures.sensitivity:

----------------------
Signatures sensitivity
----------------------

We are interested in investigating the variance-based sensitivities of the input model parameters to the output signatures. 
Several Sobol indices which are the first- and total-order sensitivities, are estimated using ``SALib`` Python library.

The estimated sensitivities are available using the :meth:`Model.signatures_sensitivity() <smash.Model.signatures_sensitivity>` method.

.. ipython:: python

    res_sens = model.signatures_sensitivity(n=32, random_state=99);

.. note::

    In real-world applications, the value of ``n`` can be much larger to attain more accurate results. 

The signatures sensitivity result is represented as a :class:`smash.SignSensResult` object containning 3 attributes which are 2 different dictionaries and 1 pandas.DataFrame:

- ``cont`` : Continuous signatures sensitivity result,

- ``event``: Flood event signatures sensitivity result,

- ``sample``: Generated samples used to estimate Sobol indices represented in a pandas.dataframe.

Each dictionary has 2 keys which are 2 different sub-dictionaries:

- ``total_si``: Result of total-order sensitivities,

- ``first_si``: Result of first-order sensitivities.

Each sub-dictionary has ``n_param`` keys (where ``n_param`` is the number of the model parameters), 
which are the dataframes containing the sensitivities of the associated model parameter to all studied signatures.

For example, to display the first-order sensitivities of the production parameter ``cp`` to all continuous signatures.

.. ipython:: python

    res_sens.cont["first_si"]["cp"]

Finally, we visualize, for instance, the total-order sensitivities of the model parameters to the lag time ``Elt`` and the peak flow ``Epf``.

.. ipython:: python

    df_cp = res_sens.event["total_si"]["cp"]
    df_cft = res_sens.event["total_si"]["cft"]
    df_exc = res_sens.event["total_si"]["exc"]
    df_lr = res_sens.event["total_si"]["lr"]

    df_sens = pd.concat([df_cp, df_cft, df_exc, df_lr], ignore_index=True)
    df_sens["parameter"] = ["cp"]*len(df_cp) + ["cft"]*len(df_cft) + ["exc"]*len(df_exc) + ["lr"]*len(df_lr)

    boxplot_sens = df_sens.boxplot(column=["Elt", "Epf"], by="parameter")
    @savefig sign_sens.png
    boxplot_sens[0].set_ylabel("Total-order sensitivity");
