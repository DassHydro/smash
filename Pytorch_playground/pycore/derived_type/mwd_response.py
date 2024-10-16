import numpy as np

class ResponseDT:
    """
    Response simulated by the hydrological model.

    Variables:
        q: Simulated discharge at gauges [m3/s]
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
        return ResponseDT()