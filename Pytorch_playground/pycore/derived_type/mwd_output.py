
import numpy as np
from pycore.derived_type.mwd_response import ResponseDT
from pycore.derived_type.mwd_rr_states import Rr_StatesDT


class OutputDT:
    """
    (MWD) Module Wrapped and Differentiated.

    Type
    ----

    - OutputDT
        Output data

        ======================== =======================================
        Variables                Description
        ======================== =======================================
        ``cost``                 Value of cost function
        ``response``             ResponseDT
        ``rr_final_states``      Rr_StatesDT
        ======================== =======================================
    """

    def __init__(self, setup, mesh):
        """
        Initialises the response, rr_final_states instances and cost variables
        """
        self.response = ResponseDT(setup, mesh)
        self.rr_final_states = Rr_StatesDT(setup, mesh)
        self.cost = None
        self.cost_jobs_q = None
        self.cost_jreg = None
        self.cost_jobs_sm = None
        self.array_cost = np.zeros(setup.maxiter+1, dtype=float)
        self.array_cost_jobs_q = np.zeros(setup.maxiter+1, dtype=float)
        self.array_cost_jreg = np.zeros(setup.maxiter+1, dtype=float)
        self.array_cost_jobs_sm = np.zeros(setup.maxiter+1, dtype=float)
        self.hp_domain = np.zeros((mesh.nrow, mesh.ncol, setup.ntime_step), dtype=float)

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return OutputDT(self)