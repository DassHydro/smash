.. _user_guide.in_depth.prcp_indices:

=====================
Precipitation indices
=====================

This section aims to go into detail on how to compute and vizualize precipitation indices.

First, open a Python interface:

.. code-block:: none

    python3
    
-------
Imports
-------

.. ipython:: python
    
    import smash
    import numpy as np
    import matplotlib.pyplot as plt
    
---------------------
Model object creation
---------------------

To compute precipitation indices, you need to create a :class:`smash.Model` object. 
For this case, we will use the ``Cance`` dataset used in the User Guide section: :ref:`user_guide.quickstart.real_case_cance`.

Load the ``setup`` and ``mesh`` dictionaries using the :meth:`smash.load_dataset` method and create the :class:`smash.Model` object.

.. ipython:: python

    setup, mesh = smash.load_dataset("Cance")
    
    model = smash.Model(setup, mesh)

---------------------------------
Precipitation indices computation
---------------------------------

Once the :class:`smash.Model` is created. The precipitation indices, computed for each gauge and each time step, are available using the :meth:`Model.prcp_indices() <smash.Model.prcp_indices>` method.

.. ipython:: python

    res = model.prcp_indices();
    
    res

.. note::

    NaN value means that there is no precipitation at this specific gauge and time step and therefore no precipitation indices.
    
The precipitation indices results are represented as a :class:`smash.PrcpIndicesResult` object containning 4 attributes which are 4 different indices:

- ``std`` : The precipitation spatial standard deviation,

- ``d1`` : The first scaled moment :cite:p:`zocatelli_2011`,

- ``d2`` : The second scaled moment :cite:p:`zocatelli_2011`,

- ``vg`` : The vertical gap :cite:p:`emmanuel_2015` .

Each attributes (i.e. precipitation indices) of the :class:`smash.PrcpIndicesResult` object is a numpy.ndarray of shape (number of gauge, number of time step)

.. ipython:: python

    res.std
    
    res.std.shape

---------------------------------
Precipitation indices description
---------------------------------

Precipitation spatial standard deviation (std)
''''''''''''''''''''''''''''''''''''''''''''''

Simply the standard deviation.

Scaled moments (d1 and d2)
''''''''''''''''''''''''''

The spatial scaled moments are described in :cite:p:`zocatelli_2011` in the section *2 Spatial moments of catchment rainfall: definitions*.

*The first scaled moment* :math:`\delta 1` *describes the distance of the centroid of catchment rainfall with respect to the average value of the flow distance (i.e. the catchment centroid). 
Values of* :math:`\delta 1` *close to 1 reflect a rainfall distribution either concentrated close to the position of the catchment centroid or spatially homogeneous, with values less than one indicating
that rainfall is distributed near the basin outlet, and values greater than one indicating that rainfall is distributed towards the catchment headwaters.*

*The second scaled moment* :math:`\delta 2` *describes the dispersion of the rainfall-weighted flow distances about their mean value with respect to the dispersion of the flow distances.
Values of* :math:`\delta 2` *close to 1 reflect a uniform-like rainfall distribution, with values less than 1 indicating that rainfall is characterised by a unimodal distribution along the flow distance.
Values greater than 1 are generally rare, and indicate cases of multimodal rainfall distributions.*


Vertical gap (vg)
'''''''''''''''''

The vertical gap is described in :cite:p:`emmanuel_2015` in the section *5.2 The proposed indexes*. 

*VG values close to zero indicate a rainfall distribution over the catchment revealing weak spatial variability. The higher the VG value,
the more concentrated the rainfall over a small part of the catchment.*

-----------------------------------
Precipitation indices visualization
-----------------------------------

Most of the precipitation indices are calculated based on flow distances. As a reminder and to facilitate the understanding of the indices values with respect to the catchment outlet and headwaters,
the flow distances of the catchment are plotted below.

.. ipython:: python
    
    flwdst = np.where(model.mesh.active_cell==0, np.nan, model.mesh.flwdst)
    
    plt.imshow(flwdst);
    plt.colorbar(label="Flow distance (m)");
    @savefig user_guide.in_depth.prcp_indices.flwdst.png
    plt.title("Cance - Flow distance");
    

Precipitation spatial standard deviation (std)
''''''''''''''''''''''''''''''''''''''''''''''

First getting the indexes (i.e. the time step) where occured the minimum and maximum. We use the methods numpy.nanargmin and numpy.nanargmax to find the indexes ignoring NaN's.

.. ipython:: python

    ind_min = np.nanargmin(res.std[0,:])
    ind_max = np.nanargmax(res.std[0,:])
    
    ind_min, ind_max
    
Then, we can visualize the precipitation grids at this time steps masking the non active cells.

.. ipython:: python

    f, ax = plt.subplots(1, 2, tight_layout=True)

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_min])
    prcp_max = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_max])
    
    map_min = ax[0].imshow(prcp_min);
    f.colorbar(map_min, ax=ax[0], fraction=0.05);
    ax[0].set_title("Minimum std");

    map_max = ax[1].imshow(prcp_max);
    f.colorbar(map_max, ax=ax[1], fraction=0.05, label="Precipitation (mm)");
    @savefig user_guide.in_depth.prcp_indices.std.png
    ax[1].set_title("Maximum std");
    
And the associated indices values

.. ipython:: python

    std_min = res.std[0, ind_min]
    std_max = res.std[0, ind_max]
    
    std_min, std_max
    
Scaled moments (d1 and d2)
''''''''''''''''''''''''''

The same applies to scaled moments, except that we will also visualize the precipitation maps where the scaled moments are closed to 1.

.. ipython:: python

    ind_min = np.nanargmin(res.d1[0,:])
    ind_max = np.nanargmax(res.d1[0,:])
    ind_one = np.nanargmin(np.abs(res.d1[0,:] - 1))
    
    ind_min, ind_max, ind_one
    
Then, we can visualize the precipitation grids at this times step masking the non active cells.

.. ipython:: python

    f, ax = plt.subplots(2, 2, tight_layout=True)

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_min])
    prcp_max = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_max])
    prcp_one = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_one])
    
    map_min = ax[0,0].imshow(prcp_min);
    f.colorbar(map_min, ax=ax[0,0]);
    ax[0,0].set_title("Minimum d1");

    map_max = ax[0,1].imshow(prcp_max);
    f.colorbar(map_max, ax=ax[0,1]);   
    ax[0,1].set_title("Maximum d1");
    
    map_one = ax[1,0].imshow(prcp_one);
    f.colorbar(map_one, ax=ax[1,0], label="Precipitation (mm)");
    ax[1,0].set_title("Close to one d1");
    
    @savefig user_guide.in_depth.prcp_indices.d1.png
    ax[1,1].axis('off');
    
And the associated indices values

.. ipython:: python

    d1_min = res.d1[0, ind_min]
    d1_one = res.d1[0, ind_one]
    d1_max = res.d1[0, ind_max]
    
    d1_min, d1_one, d1_max

Applying the same for d2

.. ipython:: python

    ind_min = np.nanargmin(res.d2[0,:])
    ind_max = np.nanargmax(res.d2[0,:])
    ind_one = np.nanargmin(np.abs(res.d2[0,:] - 1))
    
    ind_min, ind_max, ind_one
    
Then, we can visualize the precipitation grids at this time steps masking the non active cells.

.. ipython:: python

    f, ax = plt.subplots(2, 2, tight_layout=True)

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_min])
    prcp_max = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_max])
    prcp_one = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_one])
    
    map_min = ax[0,0].imshow(prcp_min);
    f.colorbar(map_min, ax=ax[0,0]);
    ax[0,0].set_title("Minimum d2");

    map_max = ax[0,1].imshow(prcp_max);
    f.colorbar(map_max, ax=ax[0,1]);   
    ax[0,1].set_title("Maximum d2");
    
    map_one = ax[1,0].imshow(prcp_one);
    f.colorbar(map_one, ax=ax[1,0], label="Precipitation (mm)");
    ax[1,0].set_title("Close to one d2");
    
    @savefig user_guide.in_depth.prcp_indices.d2.png 
    ax[1,1].axis('off');
    
And the associated indices values

.. ipython:: python

    d2_min = res.d2[0, ind_min]
    d2_one = res.d2[0, ind_one]
    d2_max = res.d2[0, ind_max]
    
    d2_min, d2_one, d2_max

Vertical gap (vg)
'''''''''''''''''

Finally, the same applies to the vertical gap.

.. ipython:: python

    ind_min = np.nanargmin(res.vg[0,:])
    ind_max = np.nanargmax(res.vg[0,:])
    
    ind_min, ind_max
    
Then, we can visualize the precipitation grids at this time steps masking the non active cells.

.. ipython:: python

    f, ax = plt.subplots(1, 2, tight_layout=True)

    ma = (model.mesh.active_cell == 0)
    
    prcp_min = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_min])
    prcp_max = np.where(ma, np.nan, model.input_data.prcp[:,:,ind_max])
    
    map_min = ax[0].imshow(prcp_min);
    f.colorbar(map_min, ax=ax[0], fraction=0.05);
    ax[0].set_title("Minimum vg");

    map_max = ax[1].imshow(prcp_max);
    f.colorbar(map_max, ax=ax[1], fraction=0.05, label="Precipitation (mm)");
    @savefig user_guide.in_depth.prcp_indices.vg.png
    ax[1].set_title("Maximum vg");
    
And the associated indices values

.. ipython:: python

    vg_min = res.vg[0, ind_min]
    vg_max = res.vg[0, ind_max]
    
    vg_min, vg_max

