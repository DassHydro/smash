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


