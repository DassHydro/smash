.. _user_guide.classical_uses.hydrological_signatures:

=======================
Hydrological Signatures
=======================

This section provides a detailed explanation of how to compute and visualize various hydrological signatures,  
which are described in the :ref:`Math / Num Documentation <math_num_documentation.hydrological_signature>` section.

First, open a Python interface:

.. code-block:: none

    python3

Imports
-------

We will first import the necessary libraries for this tutorial.

.. ipython:: python

    import smash
    import pandas as pd
    import matplotlib.pyplot as plt

Model object creation
---------------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

Signatures computation
----------------------

We can compute the observed signatures as follows. By default, all signatures are computed, but we choose several of them.
For example, here we choose to compute the continuous runoff coefficient (``'Crc'``), the flow percentile at 90% (``'Cfp90'``),
the flood event runoff coefficient (``'Erc'``), and the peak flow (``'Epf'``). 

.. ipython:: python

    sig_obs = smash.signatures(model, sign=["Crc", "Cfp90", "Erc", "Epf"], domain="obs")

To compute the simulated signatures, a simulation (either forward run or optimization) has to be performed to generate the simulated discharge.
We compute the same signatures as the observed ones for the simulated discharge.

.. ipython:: python

    model.forward_run()

    sig_sim = smash.signatures(model, sign=["Crc", "Cfp90", "Erc", "Epf"], domain="sim")

The signatures computation result object contains two attributes which are two different dictionaries:

- ``cont`` : Continuous signatures computation result,

- ``event``: Flood event signatures computation result.

.. ipython:: python

    sig_sim.cont

    sig_sim.event

For flood event signatures computation, more options can be specified such as the threshold for flood event detection, the maximum duration of the flood event, etc.
The segmentation algorithm used to detect the flood events can be adjusted by setting the ``event_seg`` parameter in the `smash.signatures` function.
This parameter is a dictionary with keys that are the parameters used for the segmentation algorithm (refer to the tutorial on :ref:`hydrograph segmentation <user_guide.classical_uses.hydrograph_segmentation>` for more details).
For instance, we can reduce the quantile threshold for flood event detection to 0.99.

.. ipython:: python

    sig_obs_2 = smash.signatures(model, sign=["Erc", "Epf"], domain="obs", event_seg={"peak_quant": 0.99})
    sig_obs_2.event

    sig_sim_2 = smash.signatures(model, sign=["Erc", "Epf"], domain="sim", event_seg={"peak_quant": 0.99})
    sig_sim_2.event

Signatures visualization
------------------------

Now, we visualize, for instance, the simulated and observed peak flow in the boxplot below.

.. ipython:: python

    df_obs = sig_obs_2.event
    df_sim = sig_sim_2.event

    df = pd.concat([df_obs.assign(domain="Observed"), df_sim.assign(domain="Simulated")], ignore_index=True)

    @savefig user_guide.classical_uses.hydrological_signatures_boxplot.png
    boxplot = df.boxplot(column=["Epf"], by="domain")

We can also compute the relative bias for any desired signature.
For example, the computation and visualization of the relative bias for the two selected flood event signatures are shown below.

.. ipython:: python

    ERR_Erc = sig_sim_2.event["Erc"] / sig_obs_2.event["Erc"] - 1
    ERR_Epf = sig_sim_2.event["Epf"] / sig_obs_2.event["Epf"] - 1

    df_err = pd.DataFrame({"Relative bias (Erc)": ERR_Erc, "Relative bias (Epf)": ERR_Epf})

    @savefig user_guide.classical_uses.hydrological_signatures_relative_bias_boxplot.png
    boxplot_err = df_err.boxplot()

.. ipython:: python
    :suppress:

    plt.close('all')