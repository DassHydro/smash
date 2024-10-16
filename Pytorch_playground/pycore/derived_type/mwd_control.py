import numpy as np

class ControlDT:
    """
    Control vector used in optimize and quantities required by the optimizer

    Variables:
        x: Control vector
        l: Control vector lower bound
        u: Control vector upper bound
        x_bkg: Control vector background
        l_bkg: Control vector lower bound background
        u_bkg: Control vector upper bound background
        nbd: Control vector kind of bound
    """

    def __init__(self, nbk=np.array([0, 0, 0, 0])):
        self.finalise()
        self.nbk = nbk
        self.n = np.sum(nbk)
        self.x = np.full(self.n, -99.0)
        self.l = np.full(self.n, -99.0)
        self.u = np.full(self.n, -99.0)
        self.x_bkg = np.zeros(self.n)
        self.l_bkg = np.full(self.n, -99.0)
        self.u_bkg = np.full(self.n, -99.0)
        self.nbd = np.full(self.n, -99, dtype=int)
        self.name = np.full(self.n, "...", dtype=str)

    def finalise(self):
        """
        Deallocates the arrays
        """
        self.x = None
        self.l = None
        self.u = None
        self.x_bkg = None
        self.l_bkg = None
        self.u_bkg = None
        self.nbd = None
        self.name = None

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return ControlDT(self.nbk)
