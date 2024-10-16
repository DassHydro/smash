from pycore.derived_type.mwd_cost_options import Cost_OptionsDT
from pycore.derived_type.mwd_optimize_options import Optimize_OptionsDT
from pycore.derived_type.mwd_common_options import Common_OptionsDT
import copy

class OptionsDT:
    def __init__(self, setup, mesh, njoc, njrc):
        self.cost = Cost_OptionsDT(setup, mesh, njoc, njrc)
        self.optimize = Optimize_OptionsDT(setup)
        self.comm = Common_OptionsDT()

    def copy(self):
        return copy.deepcopy(self)