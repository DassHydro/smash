from pycore.derived_type.mwd_control import ControlDT
from pycore.derived_type.mwd_rr_parameters import RR_ParametersDT
from pycore.derived_type.mwd_rr_states import RR_StatesDT
from pycore.derived_type.mwd_serr_mu_parameters import SErr_Mu_ParametersDT
from pycore.derived_type.mwd_serr_sigma_parameters import SErr_Sigma_ParametersDT

class ParametersDT:
    """
    (MWD) Module Wrapped and Differentiated.

    Type
    ----
    ParametersDT
        Container for all parameters. The goal is to keep the control vector in sync with the spatial matrices
        of rainfall-runoff parameters and the hyper parameters for mu/sigma of structural errors

    Variables
    ---------
    control : ControlDT
    rr_parameters : RR_ParametersDT
    rr_initial_states : RR_StatesDT
    serr_mu_parameters : SErr_Mu_ParametersDT
    serr_sigma_parameters : SErr_Sigma_ParametersDT
    """

    def __init__(self, setup, mesh):
        """
        ParametersDT_initialise subroutine in Fortran
        """
        self.control = ControlDT()  # Assuming ControlDT is a class
        self.rr_parameters = RR_ParametersDT(setup, mesh)  # Assuming RR_ParametersDT is a class with setup and mesh as parameters
        self.rr_initial_states = RR_StatesDT(setup, mesh)  # Assuming RR_StatesDT is a class with setup and mesh as parameters
        self.serr_mu_parameters = SErr_Mu_ParametersDT(setup, mesh)  # Assuming SErr_Mu_ParametersDT is a class with setup and mesh as parameters
        self.serr_sigma_parameters = SErr_Sigma_ParametersDT(setup, mesh)  # Assuming SErr_Sigma_ParametersDT is a class with setup and mesh as parameters

    def copy(self):
        """
        ParametersDT_copy subroutine in Fortran
        """
        return ParametersDT(self)