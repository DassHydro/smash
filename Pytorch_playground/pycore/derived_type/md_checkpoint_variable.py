import numpy as np

class Checkpoint_VariableDT:
    """
    Checkpoint variables passed to simulation_checkpoint subroutine. It stores variables that must
    be checkpointed by the adjoint model (i.e. variables that are push/pop each time step)

    Attributes:
    - ac_rr_parameters: Active cell rainfall-runoff parameters
    - ac_rr_states: Active cell rainfall-runoff states
    - ac_mlt: Active cell melt flux (snow module output)
    - ac_qtz: Active cell elemental discharge with time buffer (hydrological module output)
    - ac_qz: Active cell surface discharge with time buffer (routing module output)
    """
    def __init__(self):
        self.ac_rr_parameters = np.array([])
        self.ac_rr_states = np.array([])
        self.ac_mlt = np.array([])
        self.ac_qtz = np.array([])
        self.ac_qz = np.array([])