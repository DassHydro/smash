import numpy as np
from pycore.derived_type.mwd_sparse_matrix import Sparse_MatrixDT

class Atmos_DataDT:
    """
    Class representing atmospheric data.

    Attributes:
        prcp (ndarray): Array representing precipitation data.
        pet (ndarray): Array representing potential evapotranspiration data.
        snow (ndarray): Array representing snow data.
        temp (ndarray): Array representing temperature data.
        sm (ndarray): Array representing soil moisture data.
        sparse_prcp (list): List of Sparse_MatrixDT objects representing sparse precipitation data.
        sparse_pet (list): List of Sparse_MatrixDT objects representing sparse potential evapotranspiration data.
        sparse_snow (list): List of Sparse_MatrixDT objects representing sparse snow data.
        sparse_temp (list): List of Sparse_MatrixDT objects representing sparse temperature data.
        sparse_sm (list): List of Sparse_MatrixDT objects representing sparse soil moisture data.
        mean_prcp (ndarray): Array representing mean precipitation data.
        mean_pet (ndarray): Array representing mean potential evapotranspiration data.
        mean_snow (ndarray): Array representing mean snow data.
        mean_temp (ndarray): Array representing mean temperature data.
        mean_sm (ndarray): Array representing mean soil moisture data.

    Methods:
        __init__(setup, mesh): Initializes an instance of Atmos_DataDT.
        copy(): Creates a copy of the Atmos_DataDT object.
    """

    def __init__(self, setup, mesh):
        """
        Initializes an instance of Atmos_DataDT.

        Args:
            setup (SetupDT): Setup object containing setup parameters.
            mesh (MeshDT): Mesh object containing mesh parameters.
        """
        if setup.sparse_storage:
            self.sparse_prcp = [Sparse_MatrixDT(0, True, -99.0) for _ in range(setup.ntime_step)]
            self.sparse_pet = [Sparse_MatrixDT(0, True, -99.0) for _ in range(setup.ntime_step)]
            if setup.snow_module_present:
                self.sparse_snow = [Sparse_MatrixDT(0, True, -99.0) for _ in range(setup.ntime_step)]
                self.sparse_temp = [Sparse_MatrixDT(0, True, -99.0) for _ in range(setup.ntime_step)]
            if setup.read_sm:
                self.sparse_sm = [Sparse_MatrixDT(0, True, -99.0) for _ in range(setup.ntime_step)]
        else:
            self.prcp = np.full((mesh.nrow, mesh.ncol, setup.ntime_step), -99.0)
            self.pet = np.full((mesh.nrow, mesh.ncol, setup.ntime_step), -99.0)
            if setup.snow_module_present:
                self.snow = np.full((mesh.nrow, mesh.ncol, setup.ntime_step), -99.0)
                self.temp = np.full((mesh.nrow, mesh.ncol, setup.ntime_step), -99.0)
            if setup.read_sm:
                self.sm = np.full((mesh.nrow, mesh.ncol, setup.ntime_step), -99.0)

        self.mean_prcp = np.full((mesh.ng, setup.ntime_step), -99.0)
        self.mean_pet = np.full((mesh.ng, setup.ntime_step), -99.0)
        if setup.snow_module_present:
            self.mean_snow = np.full((mesh.ng, setup.ntime_step), -99.0)
            self.mean_temp = np.full((mesh.ng, setup.ntime_step), -99.0)
        if setup.read_sm:
            self.mean_sm = np.full((mesh.ng, setup.ntime_step), -99.0)

    def copy(self):
        """
        Creates a copy of the Atmos_DataDT object.

        Returns:
            Atmos_DataDT: Copy of the Atmos_DataDT object.
        """
        return Atmos_DataDT(self)