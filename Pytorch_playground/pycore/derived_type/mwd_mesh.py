import numpy as np

class MeshDT:
    """
    (MWD) Module Wrapped and Differentiated.

    Type
    ----

    - MeshDT
        Meshing data

        ======================== =======================================
        Variables                Description
        ======================== =======================================
        ``xres``                 X cell size derived from flwdir                               [m / degree]
        ``yres``                 Y cell size derived from flwdir                               [m / degree]
        ``xmin``                 X mininimum value derived from flwdir                         [m / degree]
        ``ymax``                 Y maximum value derived from flwdir                           [m / degree]
        ``nrow``                 Number of rows
        ``ncol``                 Number of columns
        ``dx``                   X cells size (meter approximation)                            [m]
        ``dy``                   Y cells size (meter approximation)                            [m]
        ``flwdir``               Flow directions
        ``flwacc``               Flow accumulation                                             [m2]
        ``flwdst``               Flow distances from main outlet(s)                            [m]
        ``npar``                 Number of partition
        ``ncpar``                Number of cells per partition
        ``cscpar``               Cumulative sum of cells per partition
        ``cpar_to_rowcol``       Matrix linking partition cell (c) to (row, col)
        ``flwpar``               Flow partitions
        ``nac``                  Number of active cell
        ``active_cell``          Mask of active cell
        ``ng``                   Number of gauge
        ``gauge_pos``            Gauge position
        ``code``                 Gauge code
        ``area``                 Drained area at gauge position                                [m2]
        ``area_dln``             Drained area at gauge position delineated                     [m2]
        ``rowcol_to_ind_ac``     Matrix linking (row, col) couple to active cell indice (k)
        ``local_active_cell``    Mask of local active cells
    """

    # Define attributes
    xres = -99.0
    yres = -99.0
    xmin = -99.0
    ymax = -99.0
    nrow = -99
    ncol = -99
    dx = None
    dy = None
    flwdir = None
    flwacc = None
    flwdst = None
    npar = -99
    ncpar = None
    cscpar = None
    cpar_to_rowcol = None
    flwpar = None
    nac = -99
    active_cell = None
    ng = -99
    gauge_pos = None
    code = None
    area = None
    area_dln = None
    rowcol_to_ind_ac = None
    local_active_cell = None
    setup = None

    def __init__(self, setup, nrow, ncol, npar, ng):
        self.nrow = nrow
        self.ncol = ncol
        self.npar = npar
        self.ng = ng
        self.setup = setup

        self.dx = np.full((self.nrow, self.ncol), -99.0)
        self.dy = np.full((self.nrow, self.ncol), -99.0)
        self.flwdir = np.full((self.nrow, self.ncol), -99)
        self.flwacc = np.full((self.nrow, self.ncol), -99.0)
        self.flwdst = np.full((self.nrow, self.ncol), -99.0)
        self.ncpar = np.full(self.npar, -99)
        self.cscpar = np.full(self.npar, -99)
        self.cpar_to_rowcol = np.full((self.nrow*self.ncol, 2), -99)
        self.flwpar = np.full((self.nrow, self.ncol), -99)
        self.active_cell = np.full((self.nrow, self.ncol), -99)
        self.gauge_pos = np.full((self.ng, 2), -99)
        self.code = np.full(self.ng, "...")
        self.area = np.full(self.ng, -99.0)
        self.area_dln = np.full(self.ng, -99.0)
        self.rowcol_to_ind_ac = np.full((self.nrow, self.ncol), -99)
        self.local_active_cell = np.full((self.nrow, self.ncol), -99)

    def copy(self):
        """
        Copy the MeshDT object.

        Returns:
            MeshDT: A copy of the MeshDT object.
        """
        return MeshDT(self)