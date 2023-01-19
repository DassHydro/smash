.. _user_guide.automatic_meshing:

=================
Automatic meshing
=================

This section aims to go into detail on how to generate a mesh automatically.

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

-----------------
Single gauge mesh
-----------------

The mesh on a single gauge is applied on the catchment of ``L'Ardèche`` (TODO add fig).

The minimum data to fill in are the coordinates of the outlet and the area in m².

.. ipython:: python

    (x, y) = 823_629, 6_358_543
    area = 2264 * 1e6

Then we need to provide the path to the flow directions raster file. Here we will directly load the 1km France flow directions from the
:meth:`smash.load_dataset` method.

.. warning::

    The flow directions file and the (``x``, ``y``) coordinates must be in the same CRS. Here we use the Lambert93 cartesien projection (*EPSG:2154*).


.. ipython:: python

    flwdir = smash.load_dataset("flwdir")

We can now generate the mesh using the :meth:`smash.generate_mesh` method.

.. ipython:: python

    mesh = smash.generate_mesh(
            path=flwdir,
            x=x,
            y=y,
            area=area,
    )

Check output
************

Once the mesh is generated, the user can visualize the output to confirm that the meshing has been done correctly.

The simpliest way to check if the meshing is correct (or at least, not completely wrong) is to check the modeled ``x``, ``y`` and ``area`` compared to
the observed data.

.. ipython:: python

    # GAUGE POS STORE (ROW, COL) INDICES
    # ROW
    y_mod = mesh["ymax"] - mesh["gauge_pos"][0, 0] * mesh["dx"]

    # COL
    x_mod = mesh["xmin"] + mesh["gauge_pos"][0, 1] * mesh["dx"]

    x_mod, x

    y_mod, y

    distance = np.sqrt((x - x_mod) ** 2 + (y - y_mod) ** 2)

    distance
   

The distance between the observed outlet and the modeled outlet is approximately 831 meters. Concerning the area.

.. ipython:: python

    area_mod = mesh["area"][0]

    area_mod, area

    rel_err = (area - area_mod) / area

    rel_err

The relative error between observed area and modeled area is approximately 0.3%.

Then, we can visualize any map such as the flow distances.

.. ipython:: python

    plt.imshow(mesh["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig flwdst_indepth_single_gauge_mesh.png
    plt.title("Single gauge - Flow distance");

Missmatching data
*****************

It can sometimes happen that the meshing observed data (``x``, ``y``, and ``area``) is not consistent with the flow directions. 
We will assume in the following case that we have a shift of the catchment outlet real coordinates.

.. ipython:: python

    x_off = x + 2_000

.. ipython:: python

    mesh_off = smash.generate_mesh(
            path=flwdir,
            x=x_off,
            y=y,
            area=area,
    )

    area_mod = mesh_off["area"][0]

    area_mod, area

    rel_err = (area - area_mod) / area

    rel_err

    plt.imshow(mesh_off["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig missmatch_flwdst_indepth_single_gauge_mesh.png
    plt.title("Missmatch single gauge - Flow distance");

As shown by the relative error on the areas (98%) and the flow distances, we did not generate the expected meshing for the catchment. 

Max depth search
****************

One way to circumvent this type of problem is to allow the meshing algorithm to search for cells further away from the real coordinates of the outlet in order to retrieve a relative error on the coherent areas.
This can be done by entering the maximum acceptable distance ``max_depth`` in the function. 
This value is by default set to 1, i.e. we take the cell minimizing the error between observed and modeled area for a circle of 1 around the observed outlet.
Setting this value to n :math:`\forall n \in \mathbb{N}^*`, allows to look at a circle of n around the outlet. 
This argument is useful to find the catchment you want to model if there are small inconsistencies between the flow directions and the observed data.
You have to be careful with this argument. If the value of ``max_depth`` is too large, it is possible that the algorithm finds a point minimizing the relative error on the areas but for a different catchment.

Let's try a ``max_depth`` set to 2.

.. ipython:: python

    mesh_off = smash.generate_mesh(
            path=flwdir,
            x=x_off,
            y=y,
            area=area,
            max_depth=2,
    )

    area_mod = mesh_off["area"][0]

    area_mod, area

    rel_err = (area - area_mod) / area

    rel_err

    plt.imshow(mesh_off["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig missmatch_maxdepth_flwdst_indepth_single_gauge_mesh.png
    plt.title("Max depth single gauge - Flow distance");

We allowed the algorithm to look for an outlet further around the real outlet and we found the initial mesh.

--------------------------
Nested multiple gauge mesh
--------------------------

The mesh on nested multiple gauge is still applied on the catchment of ``L'Ardèche`` for 4 gauges (TODO add fig).

Instead of being float, ``x``, ``y`` and ``area`` are lists of float.

.. ipython:: python

    x = np.array([786875, 770778, 810350, 823714])
    y = np.array([6370217, 6373832, 6367508, 6358435])
    area = np.array([496, 103, 1958, 2264]) * 1e6

.. note::

    ``x``, ``y`` and ``area`` are NumPy arrays but could've been lists, tuples or sets (any list-like object) but working with NumPy arrays makes the operations easier.

Then call the :meth:`smash.generate_mesh` method.

.. ipython:: python

    mesh = smash.generate_mesh(
            path=flwdir,
            x=x,
            y=y,
            area=area,
    )

Check output
************

Same as the single gauge, we can confirm that the mesh has been correctly done by checking distances and areas.

.. ipython:: python

    y_mod = mesh["ymax"] - mesh["gauge_pos"][:, 0] * mesh["dx"]

    x_mod = mesh["xmin"] + mesh["gauge_pos"][:, 1] * mesh["dx"]

    distance = np.sqrt((x - x_mod) ** 2 + (y - y_mod)** 2)

    rel_err = (area - mesh["area"]) / area

    distance

    rel_err

As well as visualize the flow distances map, which will be the same as the single gauge case because the flow distances are only calculated for the most
downstream gauge in case of nested gauges.

.. ipython:: python

    plt.imshow(mesh["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig flwdst_indepth_multiple_gauge_mesh.png
    plt.title("Nested multiple gauge - Flow distance");

Gauges location
***************

One way to visualize where the 4 gauges are located.

.. ipython:: python

    canvas = np.zeros(shape=mesh["flwdir"].shape)

    canvas = np.where(mesh["active_cell"] == 0, np.nan, canvas)

    for pos in mesh["gauge_pos"]:
        canvas[tuple(pos)] = 1

    plt.imshow(canvas, cmap="Set1_r");
    @savefig gauge_pos_indepth_multiple_gauge_mesh.png.png
    plt.title("Nested multiple gauge - Gauges location");

Gauges code
***********

When working with a single gauge, it is not usefull to give a code for the gauge. When working with multiple gauge, especially in case of optimization, we need
to know how to call the gauges. By default, if no code are given to the :meth:`smash.generate_mesh` method, the codes are the following.

.. ipython:: python

    mesh["code"]

The gauges code are always sorted in the same way than the gauges location.

The default codes are generally not enought explicit and the user can provide 
it's own gauges code by directly change the ``mesh`` dictionary value or filling in the argument ``code`` in the :meth:`smash.generate_mesh`

.. ipython:: python

    code = np.array(["V5045030", "V5046610", "V5054010", "V5064010"])

    code

    mesh["code"] = code.copy()

    mesh = smash.generate_mesh(
            path=flwdir,
            x=x,
            y=y,
            area=area,
            code=code,
    )

    mesh["code"]

.. warning::

    When setting gauges code directly in the ``mesh`` dictionary, the value must be a NumPy array. Otherwise, similar to the arguments ``x``, ``y`` and 
    ``area``, the codes can be any list-like object (NumPy array, list, tuple or set).

------------------------------
Non-nested multiple gauge mesh
------------------------------

The mesh on non-nested multiple gauge is still applied on the catchment of ``L'Ardèche`` for 4 gauges and on the catchment of ``Le Gardon`` for 3 gauges (TODO add fig).

There is no difference in the way the method :meth:`smash.generate_mesh` is called between nested and non-nested gauges.

So we fill in ``x``, ``y`` and the ``area``.

.. ipython:: python

    x = np.array(
        [786875, 770778, 810350, 823714, 786351, 778264, 792628]
    )
    y = np.array(
        [6370217, 6373832, 6367508, 6358435, 6336298, 6349858, 6324727]
    )
    area = np.array(
        [496, 103, 1958, 2264, 315, 115, 1093]
    ) * 1e6

Then call the :meth:`smash.generate_mesh` method.

.. ipython:: python

    mesh = smash.generate_mesh(
            path=flwdir,
            x=x,
            y=y,
            area=area,
    )

    plt.imshow(mesh["flwdst"]);
    plt.colorbar(label="Flow distance (m)");
    @savefig flwdst_indepth_nn_multiple_gauge_mesh.png
    plt.title("Non-nested multiple gauge - Flow distance");

The mesh has been generated for two groups of catchments which are non-nested.

.. note::

    The flow distances are always calculated on the most downstream gauge. In case of non-nested groups of catchments. The flow distance are calculated
    for each group on the most downstream gauge.

Finally, visualize the gauge positions for this mesh.

.. ipython:: python

    canvas = np.zeros(shape=mesh["flwdir"].shape)

    canvas = np.where(mesh["active_cell"] == 0, np.nan, canvas)

    for pos in mesh["gauge_pos"]:
        canvas[tuple(pos)] = 1

    plt.imshow(canvas, cmap="Set1_r");
    @savefig gauge_pos_indepth_nn_multiple_gauge_mesh.png.png
    plt.title("Non-nested multiple gauge - Gauges location");
    

