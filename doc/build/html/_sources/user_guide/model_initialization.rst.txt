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
    Enables the sparse storage of atmospheric data (i.e. precipitation and PET).
    
``mean_forcing``:bolditalic:`: bool, default False`
    Enables the calculation of average atmospheric data (i.e. precipitation and PET) by catchment.
    
    
Operator options
****************


``interception_module``:bolditalic:`: int, default 0`
    Interception module selection:
    
    - 0: No interception
    
    - 1: ``GR`` interception
    
``production_module``:bolditalic:`: int, default 0`
    Production module selection:
    
    - 0: ``GR`` production
    
``transfer_module``:bolditalic:`: int, default 0`
    Transfer module selection:
    
    - 0: ``GR4`` transfer
    
    - 1: ``GR6`` transfer
    
``exchange_module``:bolditalic:`: int, default 0`
    Exchange module selection:
    
    - 0: No exchange
    
    - 1: ``GR4`` exchange
    
``routing_module``:bolditalic:`: int, default 0`
    Routing module selection:
    
    - 0: No routing (direct sum of cells discharges)
    
    - 1: Linear routing module


Output options
**************

``save_qsim_domain``:bolditalic:`: bool, default False`
    Enables the save of simulated discharge on the entire domain.


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
    
``gauge_pos``:bolditalic:`: NumPy array, shape=(2, ng), dtype=np.int32`
    Gauge position in the grid.
    
    .. warning::
        
        The user must pay attention to the index used for this argument. Indexing in Python is from 0 to N-1 except in Fortran, the basic indexing is from 1 to N. For this argument, the position of the gauges on the grid must be defined according to the Fortran indexing.
        

``code``:bolditalic:`: NumPy array, shape=(20, ng), dtype=np.uint8`
    Code of gauges.
    
    .. warning::
        
        This argument is tricky to use because any NumPy uint8 array wrapped must be filled with ASCII values.
        
``area``:bolditalic:`: NumPy array, shape=(ng), dtype=np.float32`
    Area of gauges in square meters.
    

Grid options
************

``flow``:bolditalic:`: NumPy array, shape=(nrow, ncol), dtype=np.int32`
    Grid flow directions. `smash` is using a D8 flow directions with the following convention (**TODO**)
    
``drained_area``:bolditalic:`: NumPy array, shape=(nrow, ncol), dtype=np.int32`
    Grid drained area in number of cells.
    
``path``:bolditalic:`: NumPy array, shape=(2, nrow * ncol), dtype=np.int32`
    Grid calculation path. Sorting grid cells in ascending order of drained area.
    
    .. warning::
        
        The user must pay attention to the index used for this argument. Indexing in Python is from 0 to N-1 except in Fortran, the basic indexing is from 1 to N. For this argument, the path calculation on the grid must be defined according to the Fortran indexing.
    
    
Active cell options
*******************

``nac``:bolditalic:`: int`
    Number of active cells.
    
``active_cell``:bolditalic:`: NumPy array, shape=(nrow, ncol), dtype=np.int32`
    Grid active cells. Cells that contribute to the discharge of any gauge on the grid.
        
        
        
        
        
