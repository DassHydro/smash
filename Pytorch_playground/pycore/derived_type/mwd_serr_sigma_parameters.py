import numpy as np

class SErr_Sigma_ParametersDT:
    """
    Vectors containing hyper parameters of the temporalisation function for sigma, 
    the standard deviation of structural errors (sg0, sg1, sg2, ...)

    Variables:
        keys: Structural errors sigma hyper parameters keys
        values: Structural errors sigma hyper parameters values
    """

    def __init__(self, setup, mesh):
        """
        Initialises the keys and values arrays
        """
        self.keys = np.full(setup.nsep_sigma, "...", dtype=str)
        self.values = np.full((mesh.ng, setup.nsep_sigma), -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return SErr_Sigma_ParametersDT()