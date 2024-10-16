import numpy as np
import copy

class Optimize_OptionsDT:
    """
    (MWD) Module Wrapped and Differentiated.

    Type
    ----
    Optimize_OptionsDT
        Optimization options passed by user to define the 'parameters-to-control' mapping,
        parameters to optimize and optimizer options (factr, pgtol, bounds)

    Variables
    ---------
    mapping : str
        Control mapping name
    optimizer : str
        Optimizer name
    control_tfm : str
        Type of transformation applied to control
    rr_parameters : np.ndarray
        RR parameters to optimize
    l_rr_parameters : np.ndarray
        RR parameters lower bound
    u_rr_parameters : np.ndarray
        RR parameters upper bound
    rr_parameters_descriptor : np.ndarray
        RR parameters descriptor to use
    rr_initial_states : np.ndarray
        RR initial states to optimize
    l_rr_initial_states : np.ndarray
        RR initial states lower bound
    u_rr_initial_states : np.ndarray
        RR initial states upper bound
    rr_initial_states_descriptor : np.ndarray
        RR initial states descriptor use
    serr_mu_parameters : np.ndarray
        SErr mu parameters to optimize
    l_serr_mu_parameters : np.ndarray
        SErr mu parameters lower bound
    u_serr_mu_parameters : np.ndarray
        SErr mu parameters upper bound
    serr_sigma_parameters : np.ndarray
        SErr sigma parameters to optimize
    l_serr_sigma_parameters : np.ndarray
        SErr sigma parameters lower bound
    u_serr_sigma_parameters : np.ndarray
        SErr sigma parameters upper bound
    maxiter : int
        Maximum number of iterations
    factr : float
        LBFGSB cost function criterion
    pgtol : float
        LBFGSB gradient criterion
    """

    def __init__(self, setup):
        """
        Optimize_OptionsDT_initialise subroutine in Fortran
        """
        self.mapping = "..."
        self.optimizer = "..."
        self.control_tfm = "..."

        self.rr_parameters = np.full(setup.nrrp, -99)
        self.l_rr_parameters = np.full(setup.nrrp, -99.0)
        self.u_rr_parameters = np.full(setup.nrrp, -99.0)
        self.rr_parameters_descriptor = np.full((setup.nd, setup.nrrp), -99)

        self.rr_initial_states = np.full(setup.nrrs, -99)
        self.l_rr_initial_states = np.full(setup.nrrs, -99.0)
        self.u_rr_initial_states = np.full(setup.nrrs, -99.0)
        self.rr_initial_states_descriptor = np.full((setup.nd, setup.nrrs), -99)

        self.serr_mu_parameters = np.full(setup.nsep_mu, -99)
        self.l_serr_mu_parameters = np.full(setup.nsep_mu, -99.0)
        self.u_serr_mu_parameters = np.full(setup.nsep_mu, -99.0)

        self.serr_sigma_parameters = np.full(setup.nsep_sigma, -99)
        self.l_serr_sigma_parameters = np.full(setup.nsep_sigma, -99.0)
        self.u_serr_sigma_parameters = np.full(setup.nsep_sigma, -99.0)

        self.maxiter = setup.maxiter
        self.factr = -99.0
        self.pgtol = -99.0

    def copy(self):
        """
        Optimize_OptionsDT_copy subroutine in Fortran
        """
        return copy.deepcopy(self)