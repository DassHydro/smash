import numpy as np

class Sparse_MatrixDT:
    def __init__(self, n=0, coo_fmt=True, zvalue=0.0):
        """
        Initializes the Sparse_MatrixDT object.

        Parameters:
        - n: int, optional, default: 0
            The size of the matrix.
        - coo_fmt: bool, optional, default: True
            Specifies whether to use COO format for the matrix.
        - zvalue: float, optional, default: 0.0
            The default value for the matrix elements.
        """
        self.n = n
        self.coo_fmt = coo_fmt
        self.zvalue = zvalue
        self.indices = np.zeros(n, dtype=int) if coo_fmt else None
        self.values = np.zeros(n, dtype=float)

    def initialise(self, n, coo_fmt, zvalue):
        """
        Initializes the Sparse_MatrixDT object.

        Parameters:
        - n: int
            The size of the matrix.
        - coo_fmt: bool
            Specifies whether to use COO format for the matrix.
        - zvalue: float
            The default value for the matrix elements.
        """
        self.finalise()

        self.n = n
        self.coo_fmt = coo_fmt
        self.zvalue = zvalue

        self.values = np.zeros(n, dtype=float)
        if coo_fmt:
            self.indices = np.zeros(n, dtype=int)

    def finalise(self):
        """
        Finalizes the Sparse_MatrixDT object by deallocating memory.
        """
        self.values = None
        self.indices = None

    def initialise_array(self, n, coo_fmt, zvalue):
        """
        Initializes an array of Sparse_MatrixDT objects.

        Parameters:
        - n: int
            The size of the matrix.
        - coo_fmt: bool
            Specifies whether to use COO format for the matrix.
        - zvalue: float
            The default value for the matrix elements.
        """
        for i in range(len(self)):
            self[i].initialise(n, coo_fmt, zvalue)

    def alloc(self, n, coo_fmt, zvalue):
        """
        Allocates memory for the Sparse_MatrixDT object.

        Parameters:
        - n: int
            The size of the matrix.
        - coo_fmt: bool
            Specifies whether to use COO format for the matrix.
        - zvalue: float
            The default value for the matrix elements.
        """
        self.initialise(n, coo_fmt, zvalue)

    def copy(self):
        return Sparse_MatrixDT(self.n, self.coo_fmt, self.zvalue)