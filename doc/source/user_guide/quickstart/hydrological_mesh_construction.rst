.. _user_guide.quickstart.hydrological_mesh_construction:

==============================
Hydrological Mesh Construction
==============================

This section explains the hydrological mesh construction and details how to generate and visualize this mesh.
The tutorial uses data downloaded from the :ref:`Cance dataset <user_guide.data_and_format_description.cance>` section.

.. ipython:: python
    :suppress:

    import os
    os.system("python3 generate_dataset.py -d Cance")

Let's start by opening a Python interface. For this tutorial, assume that the example dataset has been downloaded and is located at the following path: ``'./Cance-dataset/'``.

.. code-block:: none

    python3

Imports
-------

Let's begin by importing the necessary libraries for this tutorial.

.. ipython:: python

    import smash
    import pandas as pd
    import matplotlib.pyplot as plt

.. hint::

    The visualization library `matplotlib <https://matplotlib.org/>`__ is not installed by default but can be installed with pip as follows:
    
    .. code-block:: none

        pip install matplotlib

Mesh construction
-----------------

The ``mesh`` is a Python dictionary generated using the `smash.factory.generate_mesh` function.
This function requires a flow direction file ``France_flwdir.tif`` and gauge attributes information from ``gauge_attributes.csv`` (see the Gauges' attributes section in :ref:`user_guide.data_and_format_description.format_description`).

.. ipython:: python

    gauge_attributes = pd.read_csv("./Cance-dataset/gauge_attributes.csv")

    mesh = smash.factory.generate_mesh(
        flwdir_path="./Cance-dataset/France_flwdir.tif",
        x=list(gauge_attributes["x"]),
        y=list(gauge_attributes["y"]),
        area=list(gauge_attributes["area"] * 1e6), # Convert km² to m²
        code=list(gauge_attributes["code"]),
    )

.. note::

    The gauge attributes can also be passed directly without using a ``csv`` file.

    .. ipython:: python
        :verbatim:

        mesh = smash.factory.generate_mesh(
            flwdir_path="./Cance-dataset/France_flwdir.tif",
            x=[840_261, 826_553, 828_269],
            y=[6_457_807, 6_467_115, 6_469_198],
            area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6], # Convert km² to m²
            code=["V3524010", "V3515010", "V3517010"],
        )

.. ipython:: python

    mesh.keys()

Mesh attributes and visualization
---------------------------------

To get into more details, this ``mesh`` is composed of:

- ``xres``, ``yres``
    The spatial resolution (unit of the flow directions map, **meter**)

    .. ipython:: python

        mesh["xres"], mesh["yres"]

- ``xmin``, ``ymax``
    The coordinates of the upper left corner (unit of the flow directions map, **meter**)

    .. ipython:: python

        mesh["xmin"], mesh["ymax"]

- ``nrow``, ``ncol``
    The number of rows and columns

    .. ipython:: python

        mesh["nrow"], mesh["ncol"]

- ``dx``,  ``dy``
    The spatial step in **meter**. These variables are arrays of shape *(nrow, ncol)*. In this study, the mesh is a regular grid with a constant spatial step defining squared cells.

    .. ipython:: python
        
        mesh["dx"][0,0], mesh["dy"][0,0]

- ``flwdir``
    The flow direction that can be simply visualized that way

    .. ipython:: python

        plt.imshow(mesh["flwdir"]);
        plt.colorbar(label="Flow direction (D8)");
        @savefig user_guide.in_depth.classical_calibration_io.flwdir.png
        plt.title("Cance - Flow direction");
    
.. hint::

    If the plot is not displayed, try the ``plt.show()`` command.

- ``flwdst``
    The flow distance in **meter** from the most downstream outlet

    .. ipython:: python

        plt.imshow(mesh["flwdst"]);
        plt.colorbar(label="Flow distance (m)");
        @savefig user_guide.in_depth.classical_calibration_io.flwdst.png
        plt.title("Cance - Flow distance");

- ``flwacc``
    The flow accumulation in **square meter**

    .. ipython:: python

        plt.imshow(mesh["flwacc"]);
        plt.colorbar(label="Flow accumulation (m²)");
        @savefig user_guide.in_depth.classical_calibration_io.flwacc.png
        plt.title("Cance - Flow accumulation");

- ``npar``, ``ncpar``, ``cscpar``, ``cpar_to_rowcol``, ``flwpar``
    All the variables related to independent routing partitions. We won't go into too much detail about these variables,
    as they simply allow us, in parallel computation, to identify which are the independent cells in the drainage network.

    .. ipython:: python

        mesh["npar"], mesh["ncpar"], mesh["cscpar"], mesh["cpar_to_rowcol"]
        plt.imshow(mesh["flwpar"]);
        plt.colorbar(label="Flow partition (-)");
        @savefig user_guide.in_depth.classical_calibration_io.flwpar.png
        plt.title("Cance - Flow partition");

- ``nac``, ``active_cell``
    The number of active cells, ``nac`` and the mask of active cells, ``active_cell``. When meshing, we define a rectangular area of shape *(nrow, ncol)* in which only a certain 
    number of cells contribute to the discharge at the mesh gauges. This saves us computing time and memory. 

    .. ipython:: python

        mesh["nac"]
        plt.imshow(mesh["active_cell"]);
        plt.colorbar(label="Active cell (-)");
        @savefig user_guide.in_depth.classical_calibration_io.active_cell.png
        plt.title("Cance - Active cell");

- ``ng``, ``gauge_pos``, ``code``, ``area``, ``area_dln``
    All the variables related to the gauges. The number of gauges, ``ng``, the gauges position in terms of rows and columns, ``gauge_pos``, the gauges code, ``code``, 
    the "real" drainage area, ``area`` and the delineated drainage area, ``area_dln``.

    .. ipython:: python

        mesh["ng"], mesh["gauge_pos"], mesh["code"], mesh["area"], mesh["area_dln"]

Validation of mesh generation 
-----------------------------

An important step after generating the ``mesh`` is to check that the stations have been correctly placed on the flow direction map.
To do this, we can try to visualize on which cell each station is located and whether the delineated drainage area is close to the "real" drainage area entered.

.. ipython:: python

    base = np.zeros(shape=(mesh["nrow"], mesh["ncol"]))
    base = np.where(mesh["active_cell"] == 0, np.nan, base)
    for pos in mesh["gauge_pos"]:
        base[pos[0], pos[1]] = 1
    plt.imshow(base, cmap="Set1_r");
    @savefig user_guide.in_depth.classical_calibration_io.gauge_position.png
    plt.title("Cance - Gauge position");

.. ipython:: python

    (mesh["area"] - mesh["area_dln"]) / mesh["area"] * 100 # Relative error in %

For this ``mesh``, we have a negative relative error on the simulated drainage area that varies from -0.3% for the most downstream gauge to -10% for the most upstream one
(which can be explained by the fact that small upstream catchments are more sensitive to the relatively coarse ``mesh`` resolution).

.. TODO FC link to automatic meshing

Saving the mesh
---------------

To avoid regenerating the ``mesh`` for each simulation with the same case study, we can save it using the `smash.io.save_mesh` function.
This function stores the mesh in an `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`__ file, which can later be read back using the `smash.io.read_mesh` function.

.. ipython:: python

    smash.io.save_mesh(mesh, "mesh.hdf5")
    mesh = smash.io.read_mesh("mesh.hdf5")
    mesh.keys()

.. hint::

    The meshes of demo data in `smash` can also be loaded using the function `smash.factory.load_dataset`.

.. ipython:: python
    :suppress:

    plt.close('all')
    os.remove("mesh.hdf5")
