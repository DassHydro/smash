.. _user_guide.others.model_input_data_convention:

===========================
Model input data convention
===========================

Observed dicharge
'''''''''''''''''

The observed discharge for one catchment is read from a ``.csv`` file with the following structure: 

.. csv-table:: V3524010.csv
    :align: center
    :header: "200601010000"
    :width: 50
    
    -99.000
    -99.000
    ...
    1.180
    1.185

It is a single-column ``.csv`` file containing the observed discharge values in m\ :sup:`3` \/s (negative values correspond to a gap in the chronicle) and whose header is the first time step of the chronicle.
The name of the file, for any catchment, must contains the code of the gauge which is filled in the ``mesh`` dictionary.
    
.. note::
    
    The time step of the header does not have to match the first simulation time step. `smash` manages to read the corresponding lines from ``start_time``, ``end_time`` and ``dt``.


Precipitation
'''''''''''''

The precipitation files must be stored for each time step of the simulation in ``tif`` format. For one time step, `smash` will recursively search in the ``prcp_directory``, a file with the following name structure: ``*<%Y%m%d%H%M>*.tif``.
An example of file name in tif format for the date 2014-09-15 00:00: ``prcp_201409150000.tif``. The spatial resolution must be identical to the spatial resolution of the flow directions used for the meshing.

.. warning::
    
    ``%Y%m%d%H%M`` is a unique key, the ``prcp_directory`` (and all subdirectories) can not contains files with similar dates.
    
Potential evapotranspiration
''''''''''''''''''''''''''''

The potential evapotranspiration files must be stored for each each time step of the simulation in ``tif`` format. For one time step, `smash` will recursively search in the ``pet_directory``, a file with the following name structure: ``*<%Y%m%d%H%M>*.tif``.
An example of file name in tif format for the date 2014-09-15 00:00: ``pet_201409150000.tif``. The spatial resolution must be identical to the spatial resolution of the flow directions used for the meshing.

.. warning::
    
    ``%Y%m%d%H%M`` is a unique key, the ``pet_directory`` (and all subdirectories) can not contains files with similar dates.
    
In case of ``daily_interannual_pet``, `smash` will recursively search in the ``pet_directory``, a file with the following name structure: ``*<%m%d>*.<pet_format>``.
An example of file name in tif format for the date 09-15: ``dia_pet_0915.tif``. This file will be desaggregated to the corresponding time step ``dt``.

Catchment descriptors
'''''''''''''''''''''

The catchment descriptors files must be stored in ``tif`` format. For each descriptor name filled in the setup argument ``descriptor_name``, `smash` will recursively search in the ``descriptor_directory``, a file with the following name structure: ``<descriptor_name>.tif``.
An example of file name in tif format for the slope descriptor: ``slope.tif``. The spatial resolution must be identical to the spatial resolution of the flow directions used for the meshing.

.. warning::
    
    ``descriptor_name`` is a unique key, the ``descriptor_directory`` (and all subdirectories) can not contains files with similar decriptor name.
    
Directories tree examples
'''''''''''''''''''''''''

Basic case
**********

The following directories tree is the most basic example on how all the input data of the model can be stored.

.. code-block:: text

    .
    └── Cance/
        ├── descriptor/
        │   ├── slope.tif
        │   └── dd.tif
        ├── pet/
        │   ├── pet_201409150000.tif
        │   ├── pet_201409160000.tif
        │   ├── pet_201509150000.tif
        │   └── pet_201509160000.tif
        ├── prcp/
        │   ├── prcp_201409150000.tif
        │   ├── prcp_201409160000.tif
        │   ├── prcp_201509150000.tif
        │   └── prcp_201509160000.tif
        └── qobs/
            ├── V3524010.csv
            └── V3504010.csv
    
In this case, the user must define the input data options setup as follows (assuming that the working directory is located in the ``Cance`` directory).

.. ipython:: python

    setup = {
        "read_descriptor": True,
        "descriptor_directory": "descriptor",
        "descriptor_name": ["slope", "dd"],
        "read_pet": True,
        "pet_directory": "pet",
        "read_prcp": True,
        "prcp_directory": "prcp",
        "read_qobs": True,
        "qobs_directory": "qobs",
    }
    
Any subdirectories can be added to the tree without changing the input data setup options. As example, adding subdirectories for the atmospheric input data files.

.. code-block:: text
    
    .
    └── Cance/
        ├── descriptor/
        │   ├── slope.tif
        │   └── dd.tif
        ├── pet/
        │   ├── 2014/
        │   │   └── 09/
        │   │       ├── pet_201409150000.tif
        │   │       └── pet_201409160000.tif
        │   └── 2015/
        │       └── 09/
        │           ├── pet_201509150000.tif
        │           └── pet_201509160000.tif
        ├── prcp/
        │   ├── 2014/
        │   │   └── 09/
        │   │       ├── prcp_201409150000.tif
        │   │       └── prcp_201409160000.tif
        │   └── 2015/
        │       └── 09/
        │           ├── prcp_201509150000.tif
        │           └── prcp_201509160000.tif
        └── qobs/
            ├── V3524010.csv
            └── V3504010.csv
            
Daily interannual potential evapotranspiration case
***************************************************
            
Working with daily interannual potential evapotranspiration files causes changes to the pet file names and input data setup options.

.. code-block:: text

    .
    └── Cance/
        ├── descriptor/
        │   ├── slope.tif
        │   └── dd.tif
        ├── pet/
        │   ├── dia_pet_0915.tif
        │   └── dia_pet_0916.tif
        ├── prcp/
        │   ├── 2014/
        │   │   └── 09/
        │   │       ├── prcp_201409150000.tif
        │   │       └── prcp_201409160000.tif
        │   └── 2015/
        │       └── 09/
        │           ├── prcp_201509150000.tif
        │           └── prcp_201509160000.tif
        └── qobs/
            ├── V3524010.csv
            └── V3504010.csv

And setting ``True`` to the ``daily_interannual_pet`` input data setup option (assuming that the working directory is located in the ``Cance`` directory).

.. ipython:: python

        setup = {
        "read_descriptor": True,
        "descriptor_directory": "descriptor",
        "descriptor_name": ["slope", "dd"],
        "read_pet": True,
        "pet_directory": "pet",
        "daily_interannual_pet": True,
        "read_prcp": True,
        "prcp_directory": "prcp",
        "read_qobs": True,
        "qobs_directory": "qobs",
    }
 


