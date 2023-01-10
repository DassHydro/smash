.. _user_guide.event_segmentation:

==================
Event segmentation
==================

Segmentation algorithm aims to capture important events occuring over the studied period on each catchment. 
We propose an algorithm for capturing flood events with the aid of the rainfall gradient and rainfall energy as depicted in ??? (TODO).

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

To obtain flood events segmentation, you must create a :class:`smash.Model` object. 
For this case, we will use the ``Cance`` dataset used in the User Guide section: :ref:`user_guide.real_case_cance`.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

-----------------------
Flood event information
-----------------------

Once the :class:`smash.Model` object is created, the flood events information are available using the :meth:`Model.event_segmentation() <smash.Model.event_segmentation>` method.

.. ipython:: python

    event_seg = model.event_segmentation();
    event_seg

The result is represented by a pandas.DataFrame with 6 columns.

- ``code`` : The catchment code,
- ``start`` : The beginning of event under ``YYYY-MM-DD HH:MM:SS`` format,
- ``end`` : The end of event under ``YYYY-MM-DD HH:MM:SS`` format,
- ``maxrainfall`` : The moment that the maximum precipation is observed under ``YYYY-MM-DD HH:MM:SS`` format,
- ``flood`` : The moment that the maximum discharge is observed under ``YYYY-MM-DD HH:MM:SS`` format,
- ``season`` : The season that event occurrs.

-----------------------
Hydrograph segmentation
-----------------------

Finally, the segmented event, for instance of catchement ``V3524010``, can be visualized in the hydrograph below.

.. ipython:: python

        dti = pd.date_range(start=model.setup.start_time, end=model.setup.end_time, freq="H")[1:]

        dte = pd.date_range(start=event_seg["start"][0], end=event_seg["end"][0], freq="H")

        qo = model.input_data.qobs[0, :]

        prcp = model.input_data.mean_prcp[0, :]

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0)

        ax1.bar(dti, prcp, color="lightslategrey", label="Rainfall");
        ax1.grid(alpha=.7, ls="--")
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylabel("$mm$");
        ax1.invert_yaxis()

        ax2.plot(dti, qo, label="Observed discharge");
        ax2.grid(alpha=.7, ls="--")
        ax2.tick_params(axis='x', labelrotation=20)
        ax2.set_ylabel("$mm$");

        plt.fill_between(dte, np.min(qo), np.max(qo), alpha=0.1, color="red", label="Event segmentation");

        fig.legend();
        @savefig event_seg.png
        fig.suptitle("V3524010");
