.. _user_guide.model_initialization:

.. role:: bolditalic
    :class: bolditalic

====================
Model initialization
====================

.. _user_guide.model_initialization.setup:

Setup
-----

In this section all the setup options that can be passed to the ``setup`` dictionary needed to initialize the :class:`.Model` object will be presented.


Structure options
*****************

``structure``:bolditalic:`: str, default "gr-a"`
    Model structure. Possible model structures are:

    - "gr-a"
        4 parameters and 3 states structure.
        
    - "gr-b"
        4 parameters and 4 states structure.
        
    - "gr-c"
        5 parameters and 5 states structure.
        
    .. note::
        See the User Guide section: :ref:`user_guide.model_structure` for more.


Time options
************

``dt``:bolditalic:`: int, default 3600`
    Simulation time step in seconds.

``start_time``:bolditalic:`: str`
    Simulation start time. Required format is either ``%Y%m%d%H%M`` or ``%Y-%m-%d %H%M``.

``start_time``:bolditalic:`: str`
    Simulation end time. Required format is either ``%Y%m%d%H%M`` or ``%Y-%m-%d %H%M``.
    

Input data options
******************

``read_qobs``:bolditalic:`: bool, default False`
    Enables the reading of observed dicharge files.
    
``qobs_directory``:bolditalic:`: str`
    Path to the directory with the observed discharges files.
    
``read_prcp``:bolditalic:`: bool, default False`
    Enables the reading of precipitation files.
    
``prcp_format``:bolditalic:`: str, default "tif"`
    Precipitation files format selection. Possible formats are:
    
    - "tif"
        ``Tag Image File Format``
        
    - "nc"
        .. warning::
            
            ``NetCDF``. Section in development.
            
``prcp_conversion_factor``:bolditalic:`: float, default 1.0`
    Precipitaton conversion factor. Precipitation will be **multiplied** by the conversion factor.
    
``prcp_directory``:bolditalic:`: str`
    Path to the directory with precipitaton files.

``read_pet``:bolditalic:`: bool, default False`
    Enables the reading of evapotranspiration (PET) files.
    
``pet_format``:bolditalic:`: str, default "tif"`
    PET files format selection. Possible formats are:
    
    - "tif"
        ``Tag Image File Format``
        
    - "nc"
        .. warning::
            
            ``NetCDF``. Section in development.
            
``pet_conversion_factor``:bolditalic:`: float, default 1.0`
    PET conversion factor. PET will be **multiplied** by the conversion factor.
    
``daily_interannual_pet``:bolditalic:`: bool, default False`
    Enables the reading of PET in the form of interannual PET.
    
``pet_directory``:bolditalic:`: str`
    Path to the directory with PET files.
    
``sparse_storage``:bolditalic:`: bool, default False`
    Enables the sparse storage of atmospheric data (i.e. precipitation and PET) and simulated discharge.
    
``mean_forcing``:bolditalic:`: bool, default True`
    Enables the calculation of average atmospheric data (i.e. precipitation and PET) by catchment.

``read_descriptor``:bolditalic:`: bool, default False`
    Enables the reading of physiographic descriptor files.

``descriptor_directory``:bolditalic:`: str`
    Path to the directory with physiographic descriptor files.

``descriptor_name``:bolditalic:`: list[str]`
    List of physiographic descriptor names (the size of the list will be used to allocate the descriptor array and used to read the corresponding files).

    .. note::
        See the User Guide section: :ref:`user_guide.model_input_data_convention` for more.

Output options
**************

``save_qsim_domain``:bolditalic:`: bool, default False`
    Enables the save of simulated discharge on the entire domain.
    
``save_net_prcp_domain``:bolditalic:`: bool, default False`
    Enables the save of simulated net precipitation on the entire domain.


.. _user_guide.model_initialization.mesh:

Mesh
----

In this section all the mesh options that can be passed to the ``mesh`` dictionary needed to initialize the :class:`.Model` object will be presented.

Spatial options
***************

``dx``:bolditalic:`: float, default 1000`
    Simulation spatial step in meters.
    
``nrow``:bolditalic:`: int`
    Number of rows in the grid.
    
``ncol``:bolditalic:`: int`
    Number of columns in the grid.
    
``xmin``:bolditalic:`: float`
    Lower left corner x value. This value depends on the projection system used.
    
``ymax``:bolditalic:`: float`
    Upper left corner y value. This value depends on the projection system used.
    
Gauge options
*************

``ng``:bolditalic:`: int`
    Number of gauges in the grid.
    
``gauge_pos``:bolditalic:`: numpy.ndarray, shape=(2, ng), dtype=np.int32`
    Position of gauges in the grid.


``code``:bolditalic:`: numpy.ndarray, shape=(20, ng), dtype=U`
    Code of gauges.


``area``:bolditalic:`: numpy.ndarray, shape=(ng), dtype=np.float32`
    Area of gauges in square meters.
    

Grid options
************

``flwdir``:bolditalic:`: numpy.ndarray, shape=(nrow, ncol), dtype=np.int32`
    Grid flow directions. `smash` is using a D8 flow directions with the following convention.

    .. image:: ../../_static/flwdir_convention.png
        :width: 100
        :align: center

``flwacc``:bolditalic:`: numpy.ndarray, shape=(nrow, ncol), dtype=np.int32`
    Grid flow accumulation in number of cells.

``flwdst``:bolditalic:`: numpy.ndarray, shape=(nrow, ncol), dtype=np.float32`
    Grid flow distances from the most downstream outlet for each group of nested catchments.

``path``:bolditalic:`: numpy.ndarray, shape=(2, nrow * ncol), dtype=np.int32`
    Grid calculation path. Sorting grid cells in ascending order of flow accumulation.

    
Active cell options
*******************

``nac``:bolditalic:`: int`
    Number of active cells.
    
``active_cell``:bolditalic:`: numpy.ndarray, shape=(nrow, ncol), dtype=np.int32`
    Grid active cells. Cells that contribute to the discharge of any gauge on the grid.
        
        
        
        
