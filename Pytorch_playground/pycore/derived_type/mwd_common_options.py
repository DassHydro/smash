class Common_OptionsDT:
    """
    Common options passed by user

    Variables:
        ncpu: Number of CPUs (default: 1)
        verbose: Enable verbose (default: True)
    """

    def __init__(self):
        self.ncpu = 1
        self.verbose = True

    def copy(self):
        """
        Creates a copy of the current instance
        """
        return Common_OptionsDT()