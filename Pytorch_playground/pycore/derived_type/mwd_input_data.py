import Res
from pycore.derived_type.mwd_response_data import Response_DataDT
from pycore.derived_type.mwd_u_response_data import U_Response_DataDT
from pycore.derived_type.mwd_physio_data import Physio_DataDT
from pycore.derived_type.mwd_atmos_data import Atmos_DataDT

class Input_DataDT:
    """
    Container for all user input data (not only forcing data but all inputs
    needed to run and/or optimize the model). This data are not meant to be
    changed at runtime once read.

    Variables:
        response_data: Response_DataDT
        u_response_data: U_Response_DataDT
        physio_data: Physio_DataDT
        atmos_data: Atmos_DataDT
    """

    def __init__(self, setup, mesh):
        """
        Initialises the response_data, u_response_data, physio_data, and atmos_data instances
        """
        self.response_data = Response_DataDT(setup, mesh)
        self.u_response_data = U_Response_DataDT(setup, mesh)
        self.physio_data = Physio_DataDT(setup, mesh)
        self.atmos_data = Atmos_DataDT(setup, mesh)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return Input_DataDT()