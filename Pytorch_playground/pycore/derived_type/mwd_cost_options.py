import numpy as np
from pycore.external.mwd_bayesian_tools import PriorType
import copy

class Cost_OptionsDT:
    def __init__(self, setup, mesh, njoc, njrc):
        self.bayesian = False
        self.njoc = -99
        self.jobs_cmpt = None
        self.wjobs_cmpt = None
        self.jobs_cmpt_tfm = None
        self.njrc = -99
        self.wjreg = -99
        self.jreg_cmpt = None
        self.wjreg_cmpt = None
        self.nog = -99
        self.gauge = None
        self.wgauge = None
        self.end_warmup = -99
        self.n_event = None
        self.mask_event = None
        self.control_prior = None
        self.njoc = njoc
        self.njrc = njrc
        self.jobs_cmpt = np.full(njoc, "...")
        self.wjobs_cmpt = np.full(njoc, -99)
        self.jobs_cmpt_tfm = np.full(njoc, "...")
        self.jreg_cmpt = np.full(njrc, "...")
        self.wjreg_cmpt = np.full(njrc, -99)
        self.gauge = np.full(mesh.ng, -99)
        self.wgauge = np.full(mesh.ng, -99)
        self.n_event = np.full(mesh.ng, -99)
        self.mask_event = np.full((mesh.ng, setup.ntime_step), -99)


    def copy(self):
        return copy.deepcopy(self)

    def Cost_OptionsDT_alloc_control_prior(self, n, npar):
        self.control_prior = [PriorType() for _ in range(n)]
        for i in range(n):
            self.control_prior[i].initialise(npar[i])