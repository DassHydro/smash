class SetupDT:
    def __init__(self, nd):
        # Define attributes
        self.snow_module = "..."  # Snow module
        self.hydrological_module = "..."  # Hydrological module
        self.routing_module = "..."  # Routing module
        self.serr_mu_mapping = "..."  # Mapping for structural error model
        self.serr_sigma_mapping = "..."  # Mapping for structural error model
        self.dt = -99.0  # Solver time step [s]
        self.start_time = "..."  # Simulation start time [%Y%m%d%H%M]
        self.end_time = "..."  # Simulation end time [%Y%m%d%H%M]
        self.adjust_interception = True  # Adjust interception reservoir capacity
        self.compute_mean_atmos = True  # Compute mean atmospheric data for each gauge
        self.read_qobs = False  # Read observed discharge
        self.qobs_directory = "..."  # Observed discharge directory path
        self.read_prcp = False  # Read precipitation
        self.prcp_format = "..."  # Precipitation format
        self.prcp_conversion_factor = 1.0  # Precipitation conversion factor
        self.prcp_directory = "..."  # Precipiation directory path
        self.prcp_access = "..."  # Precipiation access tree
        self.read_pet = False  # Read potential evapotranspiration
        self.pet_format = "..."  # Potential evapotranspiration format
        self.pet_conversion_factor = 1.0  # Potential evapotranpisration conversion factor
        self.pet_directory = "..."  # Potential evapotranspiration directory path
        self.pet_access = "..."  # Potential evapotranspiration access tree
        self.daily_interannual_pet = False  # Read daily interannual potential evapotranspiration
        self.read_snow = False  # Read snow
        self.snow_format = "..."  # Snow format
        self.snow_conversion_factor = 1.0  # Snow conversion factor
        self.snow_directory = "..."  # Snow directory path
        self.snow_access = "..."  # Snow access tree
        self.read_temp = False  # Read temperatur
        self.temp_format = "..."  # Temperature format
        self.temp_directory = "..."  # Temperature directory path
        self.temp_access = "..."  # Temperature access tree
        self.read_sm = False  # Read soil moisture (default: .false.)
        self.sm_format = "..."  # Soil moisture format (default: 'tif')
        self.sm_conversion_factor = 1.0  # Soil moisture format conversion factor (default: 1)
        self.sm_directory = "..."  # Soil moisture directory path (default: '...')
        self.sm_access = "..."  # Soil moisture access tree
        self.sm_metric = "..."  # Soil moisture metric
        self.prcp_partitioning = False  # Precipitation partitioning
        self.sparse_storage = False
        self.read_descriptor = False  # Read descriptor map(s)
        self.descriptor_format = "..."  # Descriptor maps format
        self.descriptor_directory = "..."  # Descriptor maps directory
        self.descriptor_name = ["..."] * nd  # Initialize descriptor_name with nd elements
        self.structure = "..."  # Structure combining all modules
        self.snow_module_present = False  # Presence of snow module
        self.ntime_step = -99  # Number of time steps
        self.nd = nd  # Number of descriptor maps
        self.nrrp = -99  # Number of rainfall-runoff parameters
        self.nrrs = -99  # Number of rainfall-runoff states
        self.nsep_mu = -99  # Number of structural error parameters for mu
        self.nsep_sigma = -99  # Number of structural error parameters for sigma
        self.nqz = -99  # Size of the temporal buffer for discharge grids
        self.maxiter = -99

    def copy(self):
        return SetupDT(self.nd)
