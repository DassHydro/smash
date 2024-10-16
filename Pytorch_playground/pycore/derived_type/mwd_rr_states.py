import numpy as np

class Rr_StatesDT:
    """
    Matrices containing spatialized states of hydrological operators.
    (reservoir level ...) The matrices are updated at each time step.

    Variables:
        keys: Rainfall-runoff states keys
        values: Rainfall-runoff states values
    """

    def __init__(self, setup, mesh):
        """
        Initialises the keys and values arrays
        """
        self.keys = np.full(setup.nrrs, "...", dtype=str)
        self.values = np.full((mesh.nrow, mesh.ncol, setup.nrrs), -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return Rr_StatesDT()