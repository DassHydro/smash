.. _user_guide.model_initialization:

====================
Model initialization
====================

.. _user_guide.model_initialization.setup:

Setup
-----

In this section all the setup options that can be passed to the ``setup`` dictionary needed to initialize the :class:`Model` object will be presented.


Time options
************

- ``dt``: int, default 3600

	Simulation time step in seconds.
	
- ``start_time``: str

	Simulation start time. Required format is either ``%Y%m%d%H%M`` or ``%Y-%m-%d %H%M``.
	
- ``start_time``: str

	Simulation end time. Required format is either ``%Y%m%d%H%M`` or ``%Y-%m-%d %H%M``.
	

Input data options
******************

- ``read_qobs``: bool, default False

	Enables the reading of observed dicharge files.
	
- ``qobs_directory``: str
	
	Path to the directory with the observed discharges files.
	
- ``read_prcp``: bool, default False

	Enables the reading of precipitation files.
	
- ``prcp_format``: str, default "tif"

	Precipitation files format selection. Possible formats are:
	
	- "tif"
	
		``Tag Image File Format``
		
	- "nc"
	
		.. warning::
			
			``NetCDF``. Section in development.
			
- ``prcp_conversion_factor``: float, default 1.0

	Precipitaton conversion factor. Precipitation will be **multiplied** by the conversion factor.
	
- ``prcp_directory``: str

	Path to the directory with precipitaton files.


- ``read_pet``: bool, default False

	Enables the reading of evapotranspiration (PET) files.
	
- ``pet_format``: str, default "tif"

	PET files format selection. Possible formats are:
	
	- "tif"
	
		``Tag Image File Format``
		
	- "nc"
	
		.. warning::
			
			``NetCDF``. Section in development.
			
- ``pet_conversion_factor``: float, default 1.0

	PET conversion factor. PET will be **multiplied** by the conversion factor.
	
- ``daily_interannual_pet``: bool, default False

	Enables the reading of PET in the form of interannual PET.
	
- ``pet_directory``: str

	Path to the directory with PET files.
	
- ``sparse_storage``: bool, default False

	Enables the sparse storage of atmospheric data (i.e. precipitation and PET).
	
- ``mean_forcing``: bool, default False

	Enables the calculation of average atmospheric data (i.e. precipitation and PET) by catchment.
	
	
Operator options
****************


- ``interception_module``: int, default 0

	Interception module selection:
	
	- 0: No interception
	
	- 1: ``GR`` interception
	
- ``production_module``: int, default 0

	Production module selection:
	
	- 0: ``GR`` production
	
- ``transfer_module``: int, default 0

	Transfer module selection:
	
	- 0: ``GR4`` transfer
	
	- 1: ``GR6`` transfer
	
- ``exchange_module``: int, default 0

	Exchange module selection:
	
	- 0: No exchange
	
	- 1: ``GR4`` exchange
	
- ``routing_module``: int, default 0

	Routing module selection:
	
	- 0: No routing (direct sum of cell discharge)
	
	- 1: Linear routing module


Output options
**************

- ``save_qsim_domain``: bool, default False

	Enables the save of simulated discharge on the entire domain.


.. _user_guide.model_initialization.mesh:

Mesh
----

In this section all the mesh options that can be passed to the ``mesh`` dictionary needed to initialize the :class:`Model` object will be presented.

Spatial options
***************

- ``dx``: float, default 1000

	Simulation spatial step in meters.
	
- ``nrow``: int

	Number of rows in the grid.
	
- ``ncol``: int

	Number of columns in the grid.
	
- ``xmin``: float

	Lower left corner x value. This value depends on the projection system used.
	
- ``ymax``: float

	Upper left corner y value. This value depends on the projection system used.
	
Gauge options
*************

- ``ng``: int

	Number of gauges in the grid.
	
- ``gauge_pos``: NumPy int array, dimension(2, ``ng``)

	Gauge position in the grid.
	
	.. warning::
		
		The user must pay attention to the index used for this argument. Indexing in Python is from 0 to N-1 except in Fortran, the basic indexing is from 1 to N. For this argument, the position of the gauges on the grid must be defined according to the Fortran indexing.
		

- ``code``: NumPy char array, dimension(20, ``ng``)

	Code of gauges.
	
	.. warning::
		
		This argument is tricky to use because any NumPy character array wrapped must be filled with ASCII values.
		
- ``area``: NumPy real array, dimension(``ng``)

	Area of gauges in square meters.
	

Grid options
************

- ``flow``: NumPy int array, dimension(``nrow``, ``ncol``)

	Grid flow directions. `smash` is using a D8 flow directions with the following convention (**TODO**)
	
- ``drained_area``: NumPy int array, dimension(``nrow``, ``ncol``)

	Grid drained area in number of cells.
	
- ``path``: NumPy int array, dimension(2, ``nrow`` * ``ncol``)

	Grid calculation path. Sorting grid cells in ascending order of drained area.
	
	.. warning::
		
		The user must pay attention to the index used for this argument. Indexing in Python is from 0 to N-1 except in Fortran, the basic indexing is from 1 to N. For this argument, the path calculation on the grid must be defined according to the Fortran indexing.
	
	
Active cell options
*******************

- ``nac``: int

	Number of active cells.
	
- ``active_cell``: NumPy int array, dimension(``nrow``, ``ncol``)

	Grid active cells. Cells that contribute to the discharge of any gauge on the grid.
		
		
		
		
		
