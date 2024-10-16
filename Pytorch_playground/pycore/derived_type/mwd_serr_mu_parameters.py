import numpy as np

class SErr_Mu_ParametersDT:
    """
    Vectors containing hyper parameters of the temporalisation function for mu, 
    the mean of structural errors (mg0, mg1, ...)

    Variables:
        keys: Structural errors mu hyper parameters keys
        values: Structural errors mu hyper parameters values
    """

    def __init__(self, setup, mesh):
        """
        Initialises the keys and values arrays
        """
        self.keys = np.full(setup.nsep_mu, "...", dtype=str)
        self.values = np.full((mesh.ng, setup.nsep_mu), -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return SErr_Mu_ParametersDT()