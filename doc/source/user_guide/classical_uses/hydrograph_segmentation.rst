.. _user_guide.classical_uses.hydrograph_segmentation:

=======================
Hydrograph Segmentation
=======================

This section provides a detailed guide on using and visualizing hydrograph segmentation
based on the algorithm depicted in the :ref:`Math / Num Documentation <math_num_documentation.hydrograph_segmentation>` section.

First, open a Python interface:

.. code-block:: none

    python3

Imports
-------

We will first import the necessary libraries for this tutorial.

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import pandas as pd

Model object creation
---------------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

Flood event information and visualization
-----------------------------------------

Once the :class:`smash.Model` object is created, the flood events information are available using the `smash.hydrograph_segmentation` function.

.. ipython:: python

    event_seg = smash.hydrograph_segmentation(model);
    event_seg
    event_seg.keys()

The result is represented by a `pandas.DataFrame` with 7 columns.

- ``code`` : The catchment code,
- ``start`` : The beginning of event under ``YYYY-MM-DD HH:MM:SS`` format,
- ``end`` : The end of event under ``YYYY-MM-DD HH:MM:SS`` format,
- ``multipeak`` : Boolean which indicates whether the event has multiple peak,
- ``maxrainfall`` : The moment when the maximum precipation is observed under ``YYYY-MM-DD HH:MM:SS`` format,
- ``flood`` : The moment when the maximum discharge is observed under ``YYYY-MM-DD HH:MM:SS`` format,
- ``season`` : The season in which the event occurs.

Then the segmented event, for instance of catchment ``V3524010``, is shown in the hydrograph below.

.. ipython:: python

    dti = pd.date_range(start=model.setup.start_time, end=model.setup.end_time, freq="h")[1:]

    qobs = model.response_data.q[0, :]

    mean_prcp = model.atmos_data.mean_prcp[0, :]

    starts = pd.to_datetime(event_seg["start"])
    ends = pd.to_datetime(event_seg["end"])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)

    ax1.bar(dti, mean_prcp, color="lightslategrey", label="Rainfall");
    ax1.axvspan(starts[0], ends[0], alpha=.1, color="red", label="Event segmentation");
    ax1.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax1.grid(alpha=.7, ls="--")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("$mm$");
    ax1.invert_yaxis()

    ax2.plot(dti, qobs, label="Observed discharge");
    ax2.axvspan(starts[0], ends[0], alpha=.1, color="red");
    ax2.grid(alpha=.7, ls="--")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylabel("$m^3/s$");
    ax2.set_xlim(ax1.get_xlim());

    fig.legend();
    @savefig user_guide.in_depth.hydrograph_segmentation.event_seg.png
    fig.suptitle("V3524010");

Customized flood event segmentation
-----------------------------------

Several options are available to customize the flood event segmentation algorithm.

Quantile option
***************

In the example above, an event seems to be missing. However, we can adjust the ``peak_quant`` parameter of the segmentation algorithm to detect more events.
By default, ``peak_quant`` is set to 0.995, meaning that only peaks exceeding the 0.995 quantile of the discharge are selected by the algorithm.
To detect more events, we can choose a smaller value, such as 0.99:

.. ipython:: python

    event_seg_2 = smash.hydrograph_segmentation(model, peak_quant=0.99);
    event_seg_2

We can once again visualize the segmented events of catchment ``V3524010`` on the hydrograph, where a second event is now selected.

.. ipython:: python

    starts = pd.to_datetime(event_seg_2["start"])
    ends = pd.to_datetime(event_seg_2["end"])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)

    ax1.bar(dti, mean_prcp, color="lightslategrey", label="Rainfall");
    ax1.axvspan(starts[0], ends[0], alpha=.1, color="red", label="Event segmentation");
    ax1.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax1.grid(alpha=.7, ls="--")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("$mm$");
    ax1.invert_yaxis()

    ax2.plot(dti, qobs, label="Observed discharge");
    ax2.axvspan(starts[0], ends[0], alpha=.1, color="red");
    ax2.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax2.grid(alpha=.7, ls="--")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylabel("$m^3/s$");
    ax2.set_xlim(ax1.get_xlim());

    fig.legend();
    @savefig user_guide.in_depth.event_segmentation.event_seg_2.png
    fig.suptitle("V3524010");

Max duration option
*******************

The ``max_duration`` parameter sets the expected maximum duration of an event (in hours), which helps define the event end. 
The default value is 240 hours, but it can be adjusted as needed. For example, setting ``max_duration=120`` limits event durations to a maximum of 120 hours:

.. ipython:: python

    event_seg_3 = smash.hydrograph_segmentation(model, max_duration=120);
    event_seg_3

Visualizing segmented events of catchment ``V3524010``:
 
.. ipython:: python

    starts = pd.to_datetime(event_seg_3["start"])
    ends = pd.to_datetime(event_seg_3["end"])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)

    ax1.bar(dti, mean_prcp, color="lightslategrey", label="Rainfall");
    ax1.axvspan(starts[0], ends[0], alpha=.1, color="red", label="Event segmentation");
    ax1.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax1.grid(alpha=.7, ls="--")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("$mm$");
    ax1.invert_yaxis()

    ax2.plot(dti, qobs, label="Observed discharge");
    ax2.axvspan(starts[0], ends[0], alpha=.1, color="red");
    ax2.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax2.grid(alpha=.7, ls="--")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylabel("$m^3/s$");
    ax2.set_xlim(ax1.get_xlim());

    fig.legend();
    @savefig user_guide.in_depth.event_segmentation.event_seg_3.png
    fig.suptitle("V3524010");

Discharge type option
*********************

The ``by`` parameter allows us to choose whether the segmentation should be based on observed or simulated discharge data.
By default, ``by='obs'`` uses observed discharge for event segmentation. However, if we want to use simulated discharge data from a simulation for hydrograph segmentation, 
we can set ``by='sim'`` to segment based on that data.
In this case, it is important to ensure that a simulation (either forward run or optimization) has been performed to generate the simulated discharge.

.. ipython:: python

    model.forward_run()
    qsim = model.response.q[0, :]

    event_seg_4 = smash.hydrograph_segmentation(model, by='sim');
    event_seg_4

Visualizing hydrograph segmented by simulated discharge of catchment ``V3524010``:

.. ipython:: python

    starts = pd.to_datetime(event_seg_4["start"])
    ends = pd.to_datetime(event_seg_4["end"])

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0)

    ax1.bar(dti, mean_prcp, color="lightslategrey", label="Rainfall");
    ax1.axvspan(starts[0], ends[0], alpha=.1, color="red", label="Event segmentation");
    ax1.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax1.grid(alpha=.7, ls="--")
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("$mm$");
    ax1.invert_yaxis()

    ax2.plot(dti, qsim, label="Simulated discharge");
    ax2.axvspan(starts[0], ends[0], alpha=.1, color="red");
    ax2.axvspan(starts[1], ends[1], alpha=.1, color="red");
    ax2.grid(alpha=.7, ls="--")
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.set_ylabel("$m^3/s$");
    ax2.set_xlim(ax1.get_xlim());

    fig.legend();
    @savefig user_guide.in_depth.event_segmentation.event_seg_4.png
    fig.suptitle("V3524010");

.. ipython:: python
    :suppress:

    plt.close('all')
