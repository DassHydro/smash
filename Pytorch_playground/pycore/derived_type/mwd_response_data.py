import numpy as np

class Response_DataDT:
    """
    User-provided observation for the hydrological model response variables

    Variables:
        q: Observed discharge at gauges [m3/s]
    """

    def __init__(self, setup, mesh):
        """
        Initialises the q array
        """
        self.q = np.full((mesh.ng, setup.ntime_step), -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return Response_DataDT()