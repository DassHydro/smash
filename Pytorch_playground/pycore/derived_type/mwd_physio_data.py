import numpy as np

class Physio_DataDT:
    """
    Physiographic data used to force the regionalization, among other things.

    Variables:
        descriptor: Descriptor maps field
        l_descriptor: Descriptor maps field min value
        u_descriptor: Descriptor maps field max value
    """

    def __init__(self, setup, mesh):
        """
        Initialises the descriptor, l_descriptor, and u_descriptor arrays
        """
        self.descriptor = np.full((mesh.nrow, mesh.ncol, setup.nd), -99.0, dtype=float)
        self.l_descriptor = np.full(setup.nd, -99.0, dtype=float)
        self.u_descriptor = np.full(setup.nd, -99.0, dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return Physio_DataDT()