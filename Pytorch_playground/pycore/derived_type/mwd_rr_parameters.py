import numpy as np

class RR_ParametersDT:
    """
    Matrices containing spatialized parameters of hydrological operators.
    (reservoir max capacity, lag time ...)

    Variables:
        keys: Rainfall-runoff parameters keys
        values: Rainfall-runoff parameters values
    """

    def __init__(self, setup, mesh):
        """
        Initialises the keys and values arrays
        """
        self.keys = np.full(setup.nrrp, "...", dtype=str)
        self.values = np.full((mesh.nrow, mesh.ncol, setup.nrrp), -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return RR_ParametersDT()