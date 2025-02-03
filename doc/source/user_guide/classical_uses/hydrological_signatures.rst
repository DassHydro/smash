.. _user_guide.classical_uses.hydrological_signatures:

=======================
Hydrological Signatures
=======================

This section aims to go into detail on how to compute and vizualize some hydrological signatures
(a list of studied signatures and their formula is given in :ref:`Math / Num Documentation <math_num_documentation.hydrological_signature>`).

First, open a Python interface:

.. code-block:: none

    python3


Imports
*******

We will first import everything we need in this tutorial

.. ipython:: python

    import smash
    import pandas as pd
    import matplotlib.pyplot as plt


Model object creation
*********************

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.demo_data.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

Signatures computation
**********************

To start we need to do a forward_run (or  optimization) simulation. Then the sinatures computation is available using the `smash.signatures` function which has three argument (model, domain and event_seg).

.. ipython:: python

    model.forward_run()

    sig_obs = smash.signatures(model, domain="obs")
    sig_sim = smash.signatures(model, domain="sim")

The signatures coputation result object contains 2 attributes which are 2 different dictionaries:

- ``cont`` : Continuous signatures computation result,

- ``event``: Flood event signatures computation result.

For example, we can display the simulated continuous and flood event signatures as follow.

.. ipython:: python

    sig_sim.cont

    sig_sim.event

Now, we visualize, for instance, the simulated and observed flood event runoff coefficients in the boxplots below.

.. ipython:: python

    df_obs = sig_obs.event
    df_sim = sig_sim.event

    df = pd.concat([df_obs, df_sim], ignore_index=True)
    df["domain"] = ["obs"]*len(df_obs) + ["sim"]*len(df_sim)

    @savefig user_guide.classical_uses.hydrological_signatures_boxplot.png
    boxplot = df.boxplot(column=["Erc", "Erchf", "Erclf", "Erch2r"], by="domain")

We can also find the relative/absolute errors for any desired signatures. The computation of the relative error for Erc (Flood event runoff coefficient) is illustated below.

.. ipython:: python

    RErc = sig_sim.event["Erc"] / sig_obs.event["Erc"] - 1
    RErc


.. To compute the signatures, you need to create a :class:`smash.Model` object. 
.. For this case, we will use the ``Cance`` dataset used in the User Guide section: :ref:`user_guide.quickstart.real_case_cance`.

.. Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset()` method and create the :class:`smash.Model` object.

.. .. ipython:: python

..     setup, mesh = smash.load_dataset("Cance")
    
..     model = smash.Model(setup, mesh)

.. .. _user_guide.in_depth.signatures.computation:

.. ----------------------
.. Signatures computation
.. ----------------------

.. To start with, we need to run a direct (or optimized) simulation. Then the signatures computation result is available using the :meth:`Model.signatures() <smash.Model.signatures>` method. 
.. The argument ``event_seg`` (only related to flood event signatures) could be defined for tuning the parameters of the segmentation algorithm. 

.. .. ipython:: python

..     model.run(inplace=True);

..     res = model.signatures(event_seg={"peak_quant": 0.99});

.. .. hint::
  
..     See the :meth:`Model.event_segmentation() <smash.Model.event_segmentation>` method, detailed in :ref:`User Guide <user_guide.in_depth.event_segmentation>`, for tuning the segmentation parameters.

.. The signatures computation result is represented as a :class:`smash.SignResult` object containning 2 attributes which are 2 different dictionaries:

.. - ``cont`` : Continuous signatures computation result,

.. - ``event``: Flood event signatures computation result.

.. Each dictionary has 2 keys which are 2 different pandas.DataFrame:

.. - ``obs``: Observation result,

.. - ``sim``: Simulation result.

.. For example, to display the simulated continuous signatures computation result.

.. .. ipython:: python

..     res.cont["sim"]

.. Now, we visualize, for instance, the simulated and observed flood event runoff coefficients in the boxplots below.

.. .. ipython:: python

..     df_obs = res.event["obs"]
..     df_sim = res.event["sim"]
    
..     df = pd.concat([df_obs, df_sim], ignore_index=True)
..     df["domain"] = ["obs"]*len(df_obs) + ["sim"]*len(df_sim)
    
..     @savefig user_guide.in_depth.signatures.sign_comp.png
..     boxplot = df.boxplot(column=["Erc", "Erchf", "Erclf", "Erch2r"], by="domain")

.. .. _user_guide.in_depth.signatures.sensitivity:

.. ----------------------
.. Signatures sensitivity
.. ----------------------

.. We are interested in investigating the variance-based sensitivities of the input model parameters to the output signatures. 
.. Several Sobol indices which are the first- and total-order sensitivities, are estimated using `SALib <https://salib.readthedocs.io>`__ Python library.
 
.. The ``problem`` argument can be defined if you prefer to change the default boundary constraints of the Model parameters. 
.. You can use the :meth:`Model.get_bound_constraints() <smash.Model.get_bound_constraints>` method to get the names of the Model parameters (depending on the defined Model structure) 
.. and its boundary constraints.

.. .. ipython:: python

..     model.get_bound_constraints()

.. Then you can redefine the problem to estimate the sensitivities of 3 parameters ``cp``, ``cft``, ``lr`` with the modified bounds (by fixing ``exc`` with its default value):

.. .. ipython:: python

..     problem = {
..         "num_vars": 3, 
..         "names": ["cp", "cft", "lr"], 
..         "bounds": [[1,1000], [1,800], [1,500]]
..     }

.. The estimated sensitivities of the Model parameters to the signatures are available using the :meth:`Model.signatures_sensitivity() <smash.Model.signatures_sensitivity>` method.

.. .. ipython:: python

..     res_sens = model.signatures_sensitivity(problem, n=16, event_seg={"peak_quant": 0.99}, random_state=99);

.. .. note::

..     In real-world applications, the value of ``n`` can be much larger to attain more accurate results.

.. .. hint::
  
..     See the :meth:`Model.event_segmentation() <smash.Model.event_segmentation>` method, detailed in :ref:`User Guide <user_guide.in_depth.event_segmentation>`, for tuning the segmentation parameters. 

.. The signatures sensitivity result is represented as a :class:`smash.SignSensResult` object containning 3 attributes which are 2 different dictionaries and 1 pandas.DataFrame:

.. - ``cont`` : Continuous signatures sensitivity result,

.. - ``event``: Flood event signatures sensitivity result,

.. - ``sample``: Generated samples used to estimate Sobol indices represented in a pandas.dataframe.

.. Each dictionary has 2 keys which are 2 different sub-dictionaries:

.. - ``total_si``: Result of total-order sensitivities,

.. - ``first_si``: Result of first-order sensitivities.

.. Each sub-dictionary has ``n_param`` keys (where ``n_param`` is the number of the model parameters), 
.. which are the dataframes containing the sensitivities of the associated model parameter to all studied signatures.

.. For example, to display the first-order sensitivities of the production parameter ``cp`` to all continuous signatures.

.. .. ipython:: python

..     res_sens.cont["first_si"]["cp"]

.. Finally, we visualize, for instance, the total-order sensitivities of the model parameters to the lag time ``Elt`` and the peak flow ``Epf``.

.. .. ipython:: python

..     df_cp = res_sens.event["total_si"]["cp"]
..     df_cft = res_sens.event["total_si"]["cft"]
..     df_lr = res_sens.event["total_si"]["lr"]

..     df_sens = pd.concat([df_cp, df_cft, df_lr], ignore_index=True)
..     df_sens["parameter"] = ["cp"]*len(df_cp) + ["cft"]*len(df_cft) + ["lr"]*len(df_lr)

..     boxplot_sens = df_sens.boxplot(column=["Elt", "Epf"], by="parameter")
..     @savefig user_guide.in_depth.signatures.sign_sens.png
..     boxplot_sens[0].set_ylabel("Total-order sensitivity");

.. .. ipython:: python
..     :suppress:

..     plt.close('all')