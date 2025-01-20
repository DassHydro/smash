.. _user_guide.classical_uses.hydrograph_segmentation:

=======================
Hydrograph Segmentation
=======================

This section aims to go into detail on how to use and visualize the hydrograph segmentation from 
a proposed algorithm as depicted in section :ref:`Math / Num Documentation <math_num_documentation.hydrograph_segmentation>`.

First, open a Python interface:

.. code-block:: none

    python3

-------
Imports
-------

.. ipython:: python
    
    import smash
    import matplotlib.pyplot as plt
    import pandas as pd

---------------------
Model object creation
---------------------

To obtain flood events segmentation, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Cance`` dataset used in the User Guide section: :ref:`user_guide.demo_data.cance`.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

-----------------------------------------
Flood event information and visualization
-----------------------------------------

Once the :class:`smash.Model` object is created, the flood events information are available using the `smash.hydrograph_segmentation` function.

.. ipython:: python

    event_seg = smash.hydrograph_segmentation(model);
    event_seg
    event_seg.keys()

The result is represented by a pandas.DataFrame with 7 columns.

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
        print(starts)
        print(ends)

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

In this case, an event seems to be missing, note that ``multipeak`` attribut indicates several events.
We suggest to adjust ``peak_quant`` of the segmentation algorithm to detect the other flood events. 
Indeed by default ``peak_quant`` fixed to 0.995, that means that only the discharges which exceed the 0.99 quantile are selected by the algorithm.
For example, we chose a smaller value.

.. ipython:: python

    event_seg_2 = smash.hydrograph_segmentation(model, peak_quant=0.99);
    event_seg_2

We can once again visualize, the segmented events of catchment ``V3524010`` on the hydrograph.

.. ipython:: python

        starts = pd.to_datetime(event_seg_2["start"])
        ends = pd.to_datetime(event_seg_2["end"])

A second event is selected.

We can once again visualize, the segmented events of catchment ``V3524010`` on the hydrograph.

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

If we are intested in specific time duration of the invent, we can control it with the ``max_duration`` option.
By default ``max_duration`` is fixed to 240 hours.

.. ipython:: python

        event_seg_3 = smash.hydrograph_segmentation(model, peak_quant=0.99, max_duration=20);
        event_seg_3

We catch the events of 20 hours.
 
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

In the same way, after a run, we can study the simulated rainfall with turning ``by`` option to ``sim`` instead of the default value ``obs``.

.. ipython:: python

    model.forward_run()
    qobs = model.response.q[0, :]

    event_seg_4 = smash.hydrograph_segmentation(model, peak_quant=0.99, max_duration=20, by='sim');
    event_seg_4

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

        ax2.plot(dti, qobs, label="Observed discharge");
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