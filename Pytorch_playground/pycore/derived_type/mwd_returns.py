import numpy as np

from pycore.derived_type.mwd_rr_states import Rr_StatesDT

class ReturnsDT:
    """
    (MWD) Module Wrapped and Differentiated.

    Type
    ----

    - ReturnsDT
        Useful quantities returned by the hydrological model other than response variables themselves.

        ======================== =======================================
        Variables                Description
        ======================== =======================================
        ``nmts``                 Number of time step to return
        ``mask_time_step``       Mask of time step
        ``rr_states``            Array of Rr_StatesDT
        ``rr_states_flag``       Return flag of rr_states
        ``q_domain``             Array of discharge
        ``q_domain_flag``        Return flag of q_domain
        ``iter_cost``            Array of cost iteration
        ``iter_cost_flag``       Return flag of iter_cost
        ``iter_projg``           Array of infinity norm of projected gradient iteration
        ``iter_projg_flag``      Return flag of iter_projg
        ``control_vector``       Array of control vector
        ``control_vector_flag``  Return flag of control_vector
        ``cost``                 Cost value
        ``cost_flag``            Return flag of cost
        ``jobs``                 Jobs value
        ``jobs_flag``            Return flag of jobs
        ``jreg``                 Jreg value
        ``jreg_flag``            Return flag of jreg
        ``log_lkh``              Log_lkh value
        ``log_lkh_flag``         Return flag of log_lkh
        ``log_prior``            Log_prior value
        ``log_prior_flag``       Return flag of log_prior
        ``log_h``                Log_h value
        ``log_h_flag``           Return flag of log_h
        ``serr_mu``              Serr mu value
        ``serr_mu_flag``         Return flag of serr_mu
        ``serr_sigma``           Serr sigma value
        ``serr_sigma_flag``      Return flag of serr_sigma
        ======================== =======================================
    """

    def __init__(self, setup, mesh, nmts, keys):
        """
        Initialises the instance variables
        """
        self.nmts = nmts
        self.mask_time_step = np.zeros(setup.ntime_step, dtype=bool)
        self.time_step_to_returns_time_step = np.full(setup.ntime_step, -99, dtype=int)
        self.rr_states = np.array([Rr_StatesDT(setup, mesh) for _ in range(nmts)])
        self.rr_states_flag = False
        self.q_domain = np.full((mesh.nrow, mesh.ncol, nmts), -99.0, dtype=float)
        self.q_domain_flag = False
        self.iter_cost = np.zeros(nmts, dtype=float)
        self.iter_cost_flag = False
        self.iter_projg = np.zeros(nmts, dtype=float)
        self.iter_projg_flag = False
        self.control_vector = np.zeros(nmts, dtype=float)
        self.control_vector_flag = False
        self.cost = 0.0
        self.cost_flag = False
        self.jobs = 0.0
        self.jobs_flag = False
        self.jreg = 0.0
        self.jreg_flag = False
        self.log_lkh = 0.0
        self.log_lkh_flag = False
        self.log_prior = 0.0
        self.log_prior_flag = False
        self.log_h = 0.0
        self.log_h_flag = False
        self.serr_mu = np.zeros((mesh.ng, setup.ntime_step), dtype=float)
        self.serr_mu_flag = False
        self.serr_sigma = np.zeros((mesh.ng, setup.ntime_step), dtype=float)
        self.serr_sigma_flag = False

        wkeys = [''.join(keys[:, i]) for i in range(keys.shape[1])]

        for key in wkeys:
            if key == 'rr_states':
                self.rr_states_flag = True
                self.rr_states = np.array([Rr_StatesDT(setup, mesh) for _ in range(nmts)])
            elif key == 'q_domain':
                self.q_domain_flag = True
                self.q_domain = np.full((mesh.nrow, mesh.ncol, nmts), -99.0, dtype=float)
            elif key == 'iter_cost':
                self.iter_cost_flag = True
                self.iter_cost = np.zeros(nmts, dtype=float)
            elif key == 'iter_projg':
                self.iter_projg_flag = True
                self.iter_projg = np.zeros(nmts, dtype=float)
            elif key == 'control_vector':
                self.control_vector_flag = True
                self.control_vector = np.zeros(nmts, dtype=float)
            elif key == 'cost':
                self.cost_flag = True
                self.cost = 0.0
            elif key == 'jobs':
                self.jobs_flag = True
                self.jobs = 0.0
            elif key == 'jreg':
                self.jreg_flag = True
                self.jreg = 0.0
            elif key == 'log_lkh':
                self.log_lkh_flag = True
                self.log_lkh = 0.0
            elif key == 'log_prior':
                self.log_prior_flag = True
                self.log_prior = 0.0
            elif key == 'log_h':
                self.log_h_flag = True
                self.log_h = 0.0
            elif key == 'serr_mu':
                self.serr_mu_flag = True
                self.serr_mu = np.zeros((mesh.ng, setup.ntime_step), dtype=float)
            elif key == 'serr_sigma':
                self.serr_sigma_flag = True
                self.serr_sigma = np.zeros((mesh.ng, setup.ntime_step), dtype=float)


    def copy(self):
        """
        Creates a copy of the current instance
        """
        return ReturnsDT()