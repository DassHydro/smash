import numpy as np

class U_Response_DataDT:
    """
    User-provided observation uncertainties for the hydrological model response variables

    Variables:
        q_stdev: Discharge uncertainty at gauges (standard deviation of independent error) [m3/s]
    """

    def __init__(self, setup, mesh):
        """
        Initialises the q_stdev array
        """
        self.q_stdev = np.zeros((mesh.ng, setup.ntime_step), dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return U_Response_DataDT()