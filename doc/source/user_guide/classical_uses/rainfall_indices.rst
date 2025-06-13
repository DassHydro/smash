.. _user_guide.classical_uses.rainfall_indices:

================
Rainfall Indices
================

This section aims to go into detail on how to compute and visualize precipitation indices.

First, open a Python interface:

.. code-block:: none

    python3
    
Imports
-------

We will first import the necessary libraries for this tutorial.

.. ipython:: python
    
    import smash
    import numpy as np
    import matplotlib.pyplot as plt
    
Model object creation
---------------------

Now, we need to create a :class:`smash.Model` object.
For this case, we will use the :ref:`user_guide.data_and_format_description.cance` dataset as an example.

Load the ``setup`` and ``mesh`` dictionaries using the `smash.factory.load_dataset` function and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.factory.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

Precipitation indices computation
---------------------------------

Once the :class:`smash.Model` is created. The precipitation indices, computed for each gauge and each time step, are available using the `smash.precipitation_indices` function.
They are defined in the next section :ref:`Precipitation indices description <user_guide.classical_uses.precipitation_indices_description>`.

.. ipython:: python

    res = smash.precipitation_indices(model);

The precipitation indices results are represented as a :class:`smash.PrecipitationIndices` object containning 4 attributes which are 4 different indices:

- ``std`` : The precipitation spatial standard deviation,

- ``d1`` : The first scaled moment :cite:p:`zocatelli_2011`,

- ``d2`` : The second scaled moment :cite:p:`zocatelli_2011`,

- ``vg`` : The vertical gap :cite:p:`emmanuel_2015` .

Each attributes (i.e. precipitation indices) of the :class:`smash.PrecipitationIndices` object is a numpy.ndarray of shape (number of gauge, number of time step).

.. ipython:: python

    res.std
    
    res.std.shape

.. note::

    NaN value means that there is no precipitation at this specific gauge and time step and therefore no precipitation indices.
    

.. _user_guide.classical_uses.precipitation_indices_description:

Precipitation indices description
---------------------------------

Precipitation spatial standard deviation (std)
**********************************************

Simply the standard deviation.

Scaled moments (d1 and d2)
**************************

The spatial scaled moments are described in :cite:t:`zocatelli_2011` in the *section 2*:

    **Spatial moments of catchment rainfall: definitions**

    *The first scaled moment* :math:`\delta 1` *describes the distance of the centroid of catchment rainfall with respect to the average value of the flow distance (i.e. the catchment centroid).*
    *Values of* :math:`\delta 1` *close to 1 reflect a rainfall distribution either concentrated close to the position of the catchment centroid or spatially homogeneous, with values less than one indicating
    that rainfall is distributed near the basin outlet, and values greater than one indicating that rainfall is distributed towards the catchment headwaters.*

    *The second scaled moment* :math:`\delta 2` *describes the dispersion of the rainfall-weighted flow distances about their mean value with respect to the dispersion of the flow distances.*
    *Values of* :math:`\delta 2` *close to 1 reflect a uniform-like rainfall distribution, with values less than 1 indicating that rainfall is characterised by a unimodal distribution along the flow distance.*
    *Values greater than 1 are generally rare, and indicate cases of multimodal rainfall distributions.*


Vertical gap (VG)
*****************

The vertical gap is described in :cite:p:`emmanuel_2015` in the *section 5.2*:

    **The proposed indexes** 

    *VG values close to zero indicate a rainfall distribution over the catchment revealing weak spatial variability. The higher the VG value,
    the more concentrated the rainfall over a small part of the catchment.*

Precipitation indices visualization
-----------------------------------

Most of the precipitation indices computations are based on flow distances. As a reminder and to facilitate the understanding of the indices values with respect to the catchment outlet and headwaters,
the flow distances of the catchment are plotted below.

.. ipython:: python
    
    flwdst = np.where(model.mesh.active_cell==0, np.nan, model.mesh.flwdst)
    
    plt.imshow(flwdst);
    plt.colorbar(label="Flow distance (m)");
    @savefig user_guide.in_depth.prcp_indices.flwdst.png
    plt.title("Cance - Flow distance");

Let's have nicer callable variables.

.. ipython:: python 

    std = res.std
    d1 = res.d1
    d2 = res.d2
    vg = res.vg

    prcp = model.atmos_data.prcp

Precipitation spatial standard deviation (std)
**********************************************
    
Let's start by finding out where the minimum and maximum are located for the first gauge.
The methods numpy.nanargmin and numpy.nanargmax ignore NaN's values.

.. ipython:: python

    ind_min = np.nanargmin(std[0, :])
    ind_max = np.nanargmax(std[0, :])
    
    ind_min, ind_max

The associated values at those time steps are:

.. ipython:: python

    std_min = std[0, ind_min]
    std_max = std[0, ind_max]
    
    std_min, std_max

We can also visualize the precipitations at those time steps, masking the non active cells.

.. ipython:: python

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, prcp[:, :, ind_min])
    prcp_max = np.where(ma, np.nan, prcp[:, :, ind_max])

    fig, ax = plt.subplots(1, 2, tight_layout=True)

    map_min = ax[0].imshow(prcp_min);
    fig.colorbar(map_min, ax=ax[0], fraction=0.05);
    ax[0].set_title("Minimum - std");

    map_max = ax[1].imshow(prcp_max);
    fig.colorbar(map_max, ax=ax[1], fraction=0.05, label="Precipitation (mm)");
    @savefig user_guide.in_depth.prcp_indices.std.png
    ax[1].set_title("Maximum - std");
    
Scaled moments (d1 and d2)
**************************

Again we find out where the minimum and maximum are located and give the associated values.

.. ipython:: python

    ind_min = np.nanargmin(d1[0, :])
    ind_max = np.nanargmax(d1[0, :])
    ind_min, ind_max

    d1_min = d1[0, ind_min]
    d1_max = d1[0, ind_max]
    d1_min, d1_max

We also interested in the precipitations when the scaled moment is closed to 1.

.. ipython:: python

    ind_one = np.nanargmin(np.abs(d1[0, :] - 1))
    ind_one

    d1_one = d1[0, ind_one]
    d1_one

Then, we can visualize the precipitations at those time steps.

.. ipython:: python

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, prcp[:, :, ind_min])
    prcp_max = np.where(ma, np.nan, prcp[:, :, ind_max])
    prcp_one = np.where(ma, np.nan, prcp[:, :, ind_one])
    
    fig, ax = plt.subplots(2, 2, tight_layout=True)

    map_min = ax[0, 0].imshow(prcp_min);
    fig.colorbar(map_min, ax=ax[0, 0]);
    ax[0, 0].set_title("Minimum - d1");

    map_max = ax[0, 1].imshow(prcp_max);
    fig.colorbar(map_max, ax=ax[0, 1]);   
    ax[0, 1].set_title("Maximum - d1");
    
    map_one = ax[1, 0].imshow(prcp_one);
    fig.colorbar(map_one, ax=ax[1, 0], label="Precipitation (mm)");
    ax[1, 0].set_title("Close to one - d1");
    
    @savefig user_guide.in_depth.prcp_indices.d1.png
    ax[1, 1].axis('off');


Applying the same principle to the d2 moment:

.. ipython:: python

    ind_min = np.nanargmin(d2[0, :])
    ind_max = np.nanargmax(d2[0, :])
    ind_one = np.nanargmin(np.abs(d2[0, :] - 1))
    
    ind_min, ind_max, ind_one

    d2_min = d2[0, ind_min]
    d2_max = d2[0, ind_max]
    d2_one = d2[0, ind_one]

    d2_min, d2_max, d2_one

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, prcp[:, :, ind_min])
    prcp_max = np.where(ma, np.nan, prcp[:, :, ind_max])
    prcp_one = np.where(ma, np.nan, prcp[:, :, ind_one])
    
    f, ax = plt.subplots(2, 2, tight_layout=True)

    map_min = ax[0, 0].imshow(prcp_min);
    f.colorbar(map_min, ax=ax[0, 0]);
    ax[0, 0].set_title("Minimum - d2");

    map_max = ax[0, 1].imshow(prcp_max);
    f.colorbar(map_max, ax=ax[0, 1]);   
    ax[0, 1].set_title("Maximum - d2");
    
    map_one = ax[1, 0].imshow(prcp_one);
    f.colorbar(map_one, ax=ax[1, 0], label="Precipitation (mm)");
    ax[1, 0].set_title("Close to one - d2");
    
    @savefig user_guide.in_depth.prcp_indices.d2.png 
    ax[1, 1].axis('off');

Vertical gap (VG)
*****************

Finally, applying the same principle to the vertical gap:

.. ipython:: python

    ind_min = np.nanargmin(vg[0, :])
    ind_max = np.nanargmax(vg[0, :])
    
    ind_min, ind_max
    
    vg_min = res.vg[0, ind_min]
    vg_max = res.vg[0, ind_max]
    
    vg_min, vg_max

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, prcp[:,:,ind_min])
    prcp_max = np.where(ma, np.nan, prcp[:,:,ind_max])
    
    fig, ax = plt.subplots(1, 2, tight_layout=True)

    map_min = ax[0].imshow(prcp_min);
    fig.colorbar(map_min, ax=ax[0], fraction=0.05);
    ax[0].set_title("Minimum - vg");

    map_max = ax[1].imshow(prcp_max);
    fig.colorbar(map_max, ax=ax[1], fraction=0.05, label="Precipitation (mm)");
    @savefig user_guide.in_depth.prcp_indices.vg.png
    ax[1].set_title("Maximum - vg");
    
.. ipython:: python
    :suppress:

    plt.close('all')