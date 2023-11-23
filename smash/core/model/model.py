from __future__ import annotations

from smash._constant import (
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    DEFAULT_BOUNDS_RR_PARAMETERS,
    DEFAULT_BOUNDS_RR_INITIAL_STATES,
    DEFAULT_BOUNDS_SERR_MU_PARAMETERS,
    DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS,
)

from smash.core.model._build_model import (
    _map_dict_to_fortran_derived_type,
    _build_mesh,
    _build_input_data,
    _build_parameters,
    _build_output,
)
from smash.core.model._standardize import (
    _standardize_model_args,
    _standardize_get_rr_parameters_args,
    _standardize_get_rr_initial_states_args,
    _standardize_get_serr_mu_parameters_args,
    _standardize_get_serr_sigma_parameters_args,
    _standardize_get_rr_final_states_args,
    _standardize_set_rr_parameters_args,
    _standardize_set_rr_initial_states_args,
    _standardize_set_serr_mu_parameters_args,
    _standardize_set_serr_sigma_parameters_args,
)
from smash.core.simulation.run.run import _forward_run
from smash.core.simulation.run._standardize import _standardize_forward_run_args
from smash.core.simulation.optimize.optimize import (
    _optimize,
    _bayesian_optimize,
)
from smash.core.simulation.optimize._standardize import (
    _standardize_optimize_args,
    _standardize_bayesian_optimize_args,
)
from smash.core.simulation.estimate.estimate import _multiset_estimate
from smash.core.simulation.estimate._standardize import (
    _standardize_multiset_estimate_args,
)

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT
from smash.fcore._mwd_parameters_manipulation import (
    get_serr_mu as wrap_get_serr_mu,
    get_serr_sigma as wrap_get_serr_sigma,
)

import numpy as np

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.simulation.optimize.optimize import (
        Optimize,
        MultipleOptimize,
        BayesianOptimize,
    )
    from smash.core.simulation.run.run import ForwardRun, MultipleForwardRun
    from smash.core.simulation.estimate.estimate import MultisetEstimate

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.

    Parameters
    ----------
    setup : dict
        Model initialization setup dictionary.
        TODO FC: exlpain attributes.

    mesh : dict
        Model initialization mesh dictionary.
        TODO FC: exlpain attributes.

    See Also
    --------
    smash.io.save_model : Save the Model object.
    smash.io.read_model : Read the Model object.
    smash.io.save_setup : Save the Model initialization setup dictionary.
    smash.io.read_setup : Read the Model initialization setup dictionary.
    smash.factory.generate_mesh : Automatic mesh generation.
    smash.io.save_mesh : Save the Model initialization mesh dictionary.
    smash.io.read_mesh : Read the Model initialization mesh dictionary.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    </> Reading precipitation: 100%|█████████████████████████████| 1440/1440 [00:00<00:00, 10323.28it/s]
    </> Reading daily interannual pet: 100%|███████████████████████| 366/366 [00:00<00:00, 13735.82it/s]
    </> Disaggregating daily interannual pet: 100%|█████████████| 1440/1440 [00:00<00:00, 132565.08it/s]
    </> Adjusting GR interception capacity
    """

    def __init__(self, setup: dict | None, mesh: dict | None):
        if setup and mesh:
            args = [deepcopy(arg) for arg in [setup, mesh]]
            setup, mesh = _standardize_model_args(*args)

            self.setup = SetupDT(setup["nd"])

            _map_dict_to_fortran_derived_type(setup, self.setup)

            self.mesh = MeshDT(
                self.setup, mesh["nrow"], mesh["ncol"], mesh["npar"], mesh["ng"]
            )

            _map_dict_to_fortran_derived_type(mesh, self.mesh)

            _build_mesh(self.setup, self.mesh)

            self._input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self._input_data)

            self._parameters = ParametersDT(self.setup, self.mesh)

            _build_parameters(self.setup, self.mesh, self._input_data, self._parameters)

            self._output = OutputDT(self.setup, self.mesh)

            _build_output(self.setup, self._output)

    def __copy__(self):
        copy = Model(None, None)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy._input_data = self._input_data.copy()
        copy._parameters = self._parameters.copy()
        copy._output = self._output.copy()

        return copy

    def __repr__(self):
        # % Nested function. This avoids duplicating the attribute check.
        def _valid_attr(obj, attr):
            if attr.startswith("_"):
                return False
            try:
                value = getattr(obj, attr)
            except:
                return False
            if callable(value):
                return False

            return True

        ret = [self.__class__.__name__]
        for attr in dir(self):
            if not _valid_attr(self, attr):
                continue
            value = getattr(self, attr)

            sub_attr_list = [
                sub_attr for sub_attr in dir(value) if _valid_attr(value, sub_attr)
            ]

            # % Do not print too much attributes
            if len(sub_attr_list) > 4:
                sub_attr_list = sub_attr_list[0:2] + ["..."] + sub_attr_list[-3:-1]

            ret.append(f"    {attr}: {sub_attr_list}")

        return "\n".join(ret)

    @property
    def setup(self):
        """
        The setup used to create the Model object.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.setup.<TAB>
        model.setup.adjust_interception     model.setup.pet_directory
        model.setup.copy()                  model.setup.pet_format
        model.setup.daily_interannual_pet   model.setup.prcp_conversion_factor
        model.setup.descriptor_directory    model.setup.prcp_directory
        model.setup.descriptor_format       model.setup.prcp_format
        model.setup.descriptor_name         model.setup.qobs_directory
        model.setup.dt                      model.setup.read_descriptor
        model.setup.end_time                model.setup.read_pet
        model.setup.from_handle(            model.setup.read_prcp
        model.setup.nd                      model.setup.read_qobs
        model.setup.nop                     model.setup.serr_mu_mapping
        model.setup.nos                     model.setup.serr_sigma_mapping
        model.setup.nsep_mu                 model.setup.sparse_storage
        model.setup.nsep_sigma              model.setup.start_time
        model.setup.ntime_step              model.setup.structure
        model.setup.pet_conversion_factor
        """

        return self._setup

    @setup.setter
    def setup(self, value):
        self._setup = value

    @property
    def mesh(self):
        """
        The mesh used to create the Model object.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.mesh.<TAB>
        model.mesh.active_cell           model.mesh.local_active_cell
        model.mesh.area                  model.mesh.nac
        model.mesh.area_dln              model.mesh.ncol
        model.mesh.code                  model.mesh.ng
        model.mesh.copy()                model.mesh.nrow
        model.mesh.dx                    model.mesh.path
        model.mesh.dy                    model.mesh.rowcol_to_ind_sparse
        model.mesh.flwacc                model.mesh.xmin
        model.mesh.flwdir                model.mesh.xres
        model.mesh.flwdst                model.mesh.ymax
        model.mesh.from_handle(          model.mesh.yres
        model.mesh.gauge_pos
        """

        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def response_data(self):
        """
        Observation response data.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.response_data.<TAB>
        model.response_data.copy()        model.response_data.q
        model.response_data.from_handle(
        """

        return self._input_data.response_data

    @response_data.setter
    def response_data(self, value):
        self._input_data.response_data = value

    @property
    def u_response_data(self):
        """
        Observation uncertainties response data.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.u_response_data.<TAB>
        model.u_response_data.copy()        model.u_response_data.q_stdev
        model.u_response_data.from_handle(
        """

        return self._input_data.u_response_data

    @u_response_data.setter
    def u_response_data(self, value):
        self._input_data.u_response_data = value

    @property
    def physio_data(self):
        """
        Physiographic data.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.physio_data.<TAB>
        model.physio_data.copy()        model.physio_data.l_descriptor
        model.physio_data.descriptor    model.physio_data.u_descriptor
        model.physio_data.from_handle(
        """

        return self._input_data.physio_data

    @physio_data.setter
    def physio_data(self, value):
        self._input_data.physio_data = value

    @property
    def atmos_data(self):
        """
        Atmospheric and meteorological data.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.atmos_data.<TAB>
        model.atmos_data.copy()                    model.atmos_data.mean_prcp
        model.atmos_data.from_handle(              model.atmos_data.pet
        model.atmos_data.init_array_sparse_pet()   model.atmos_data.prcp
        model.atmos_data.init_array_sparse_prcp()  model.atmos_data.sparse_pet
        model.atmos_data.mean_pet
        """

        return self._input_data.atmos_data

    @atmos_data.setter
    def atmos_data(self, value):
        self._input_data.atmos_data = value

    @property
    def rr_parameters(self):
        """
        Get rainfall-runoff parameters for the actual structure of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.rr_parameters.<TAB>
        model.rr_parameters.copy()        model.rr_parameters.keys
        model.rr_parameters.from_handle(  model.rr_parameters.values
        """

        return self._parameters.rr_parameters

    @rr_parameters.setter
    def rr_parameters(self, value):
        self._parameters.rr_parameters = value

    @property
    def rr_initial_states(self):
        """
        Get rainfall-runoff initial states for the actual structure of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.rr_initial_states.<TAB>
        model.rr_initial_states.copy()        model.rr_initial_states.keys
        model.rr_initial_states.from_handle(  model.rr_initial_states.values
        """

        return self._parameters.rr_initial_states

    @rr_initial_states.setter
    def rr_initial_states(self, value):
        self._parameters.rr_initial_states = value

    @property
    def serr_mu_parameters(self):
        """
        Get structural error mu parameters for the actual mu mapping of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.serr_mu_parameters.<TAB>
        model.serr_mu_parameters.copy()        model.serr_mu_parameters.keys
        model.serr_mu_parameters.from_handle(  model.serr_mu_parameters.values

        """

        return self._parameters.serr_mu_parameters

    @serr_mu_parameters.setter
    def serr_mu_parameters(self, value):
        self._parameters.serr_mu_parameters = value

    @property
    def serr_sigma_parameters(self):
        """
        Get structural error sigma parameters for the actual sigma mapping of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.serr_sigma_parameters.<TAB>
        model.serr_sigma_parameters.copy()        model.serr_sigma_parameters.keys
        model.serr_sigma_parameters.from_handle(  model.serr_sigma_parameters.values
        """

        return self._parameters.serr_sigma_parameters

    @serr_sigma_parameters.setter
    def serr_sigma_parameters(self, value):
        self._parameters.serr_sigma_parameters = value

    @property
    def response(self):
        """
        Simulated response data.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.response.<TAB>
        model.response.copy()        model.response.from_handle(  model.response.q
        """

        return self._output.response

    @response.setter
    def response(self, value):
        self._output.response = value

    @property
    def rr_final_states(self):
        """
        Get rainfall-runoff final states for the actual structure of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        If you are using IPython, tab completion allows you to visualize all the attributes and methods:

        >>> model.rr_final_states.<TAB>
        model.rr_final_states.copy()        model.rr_final_states.keys
        model.rr_final_states.from_handle(  model.rr_final_states.values
        """

        return self._output.rr_final_states

    @rr_final_states.setter
    def rr_final_states(self, value):
        self._output.rr_final_states = value

    def copy(self):
        """
        Make a deepcopy of the Model.

        Returns
        -------
        Model
            A copy of Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        Create a deepcopy of Model:

        >>> model_dc = model.copy()
        >>> model_dc.set_rr_parameters("cp", 100)

        >>> model_dc.get_rr_parameters("cp")[0, 0]
        100.0

        >>> model.get_rr_parameters("cp")[0, 0]
        200.0
        """

        return self.__copy__()

    def get_rr_parameters(self, key: str) -> np.ndarray:
        """
        Get the values of a rainfall-runoff model parameter.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff parameter.

        Returns
        -------
        value : numpy.ndarray
            A 2D-array representing the values of the rainfall-runoff parameter.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> cp = model.get_rr_parameters("cp")
        >>> cp.shape
        (28, 28)

        See Also
        --------
        Model.rr_parameters : Get rainfall-runoff parameters for the actual structure of the Model.
        """

        key = _standardize_get_rr_parameters_args(self, key)
        ind = np.argwhere(self._parameters.rr_parameters.keys == key).item()

        return self._parameters.rr_parameters.values[..., ind]

    def set_rr_parameters(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for a rainfall-runoff model parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff parameter.

        value : Numeric or np.ndarray
            The values to set for the rainfall-runoff parameter.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.set_rr_parameters("cp", 150)
        >>> model.get_rr_parameters("cp")[0, 0]
        150.0

        See Also
        --------
        Model.rr_parameters : Get rainfall-runoff parameters for the actual structure of the Model.
        """

        key, value = _standardize_set_rr_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.rr_parameters.keys == key).item()

        self._parameters.rr_parameters.values[..., ind] = value

    def get_rr_initial_states(self, key: str) -> np.ndarray:
        """
        Get the values of a rainfall-runoff model initial state.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff initial state.

        Returns
        -------
        value : numpy.ndarray
            A 2D-array representing the values of the rainfall-runoff initial state.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> hp = model.get_rr_initial_states("hp")
        >>> hp.shape
        (28, 28)

        See Also
        --------
        Model.rr_initial_states : Get rainfall-runoff initial states for the actual structure of the Model.
        """

        key = _standardize_get_rr_initial_states_args(self, key)
        ind = np.argwhere(self._parameters.rr_initial_states.keys == key).item()

        return self._parameters.rr_initial_states.values[..., ind]

    def set_rr_initial_states(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for a rainfall-runoff model initial state.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff initial state.

        value : Numeric or np.ndarray
            The values to set for the rainfall-runoff initial state.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.set_rr_initial_states("hp", 0.5)
        >>> model.get_rr_initial_states("hp")[0, 0]
        0.55

        See Also
        --------
        Model.rr_initial_states : Get rainfall-runoff initial states for the actual structure of the Model.
        """

        key, value = _standardize_set_rr_initial_states_args(self, key, value)
        ind = np.argwhere(self._parameters.rr_initial_states.keys == key).item()

        self._parameters.rr_initial_states.values[..., ind] = value

    def get_serr_mu_parameters(self, key: str) -> np.ndarray:
        """
        Get the values of a stuctural error mu parameter.

        Parameters
        ----------
        key : str
            The name of the structural error mu parameter.

        Returns
        -------
        value : numpy.ndarray
            A 2D-array representing the values of the structural error mu parameter.

        Examples
        --------
        TODO FC: Fill

        See Also
        --------
        Model.serr_mu_parameters : Get structural error mu parameters for the actual mu mapping of the Model.
        """

        key = _standardize_get_serr_mu_parameters_args(self, key)
        ind = np.argwhere(self._parameters.serr_mu_parameters.keys == key).item()

        return self._parameters.serr_mu_parameters.values[..., ind]

    def set_serr_mu_parameters(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for a structural error mu parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the structural error mu parameter.

        value : Numeric or np.ndarray
            The values to set for the structural error mu parameter.

        Examples
        --------
        TODO FC: Fill

        See Also
        --------
        Model.serr_mu_parameters : Get structural error mu parameters for the actual mu mapping of the Model.
        """

        key, value = _standardize_set_serr_mu_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.serr_mu_parameters.keys == key).item()

        self._parameters.serr_mu_parameters.values[..., ind] = value

    def get_serr_sigma_parameters(self, key: str) -> np.ndarray:
        """
        Get the values of a stuctural error sigma parameter.

        Parameters
        ----------
        key : str
            The name of the structural error sigma parameter.

        Returns
        -------
        value : numpy.ndarray
            A 2D-array representing the values of the structural error sigma parameter.

        Examples
        --------
        TODO FC: Fill

        See Also
        --------
        Model.serr_sigma_parameters : Get structural error sigma parameters for the actual sigma mapping of the Model.
        """

        key = _standardize_get_serr_sigma_parameters_args(self, key)
        ind = np.argwhere(self._parameters.serr_sigma_parameters.keys == key).item()

        return self._parameters.serr_sigma_parameters.values[..., ind]

    def set_serr_sigma_parameters(self, key: str, value: Numeric | np.ndarray):
        """
        Set the values for a structural error sigma parameter.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        key : str
            The name of the structural error sigma parameter.

        value : Numeric or np.ndarray
            The values to set for the structural error sigma parameter.

        Examples
        --------
        TODO FC: Fill

        See Also
        --------
        Model.serr_sigma_parameters : Get structural error sigma parameters for the actual sigma mapping of the Model.
        """

        key, value = _standardize_set_serr_sigma_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.serr_sigma_parameters.keys == key).item()

        self._parameters.serr_sigma_parameters.values[..., ind] = value

    def get_rr_final_states(self, key: str) -> np.ndarray:
        """
        Get the values of a rainfall-runoff model final state.

        Parameters
        ----------
        key : str
            The name of the rainfall-runoff final state.

        Returns
        -------
        value : numpy.ndarray
            A 2D-array representing the values of the rainfall-runoff final state.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> model.forward_run()

        >>> ht = model.get_rr_final_states("ht")
        >>> ht
        array([[0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.2807956 , 0.28895575, 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      ],
               ...
               [0.3043022 , 0.3066468 , 0.30687398, 0.30758098, 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
                0.01      , 0.01      , 0.01      ]], dtype=float32)

        See Also
        --------
        Model.rr_final_states : Get rainfall-runoff final states for the actual structure of the Model.
        """

        key = _standardize_get_rr_final_states_args(self, key)
        ind = np.argwhere(self._output.rr_final_states.keys == key).item()

        return self._output.rr_final_states.values[..., ind]

    def get_rr_parameters_bounds(self) -> dict:
        """
        Get the boundary condition for the rainfall-runoff model parameters.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each rainfall-runoff parameter for the actual structure of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> bounds = model.get_rr_parameters_bounds()
        >>> bounds
        {'ci': (1e-06, 100.0), 'cp': (1e-06, 1000.0), 'ct': (1e-06, 1000.0),
         'kexc': (-50, 50), 'llr': (1e-06, 1000.0)}
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_RR_PARAMETERS.items()
            if key in STRUCTURE_RR_PARAMETERS[self.setup.structure]
        }

    def get_rr_initial_states_bounds(self) -> dict:
        """
        Get the boundary condition for the rainfall-runoff model initial states.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each rainfall-runoff initial state in the actual structure of the Model.

        Examples
        --------
        >>> import smash
        >>> from smash.factory import load_dataset
        >>> setup, mesh = load_dataset("cance")
        >>> model = smash.Model(setup, mesh)

        >>> bounds = model.get_rr_initial_states_bounds()
        >>> bounds
        {'hi': (1e-06, 0.999999), 'hp': (1e-06, 0.999999),
         'ht': (1e-06, 0.999999), 'hlr': (1e-06, 1000.0)}
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_RR_INITIAL_STATES.items()
            if key in STRUCTURE_RR_STATES[self.setup.structure]
        }

    def get_serr_mu_parameters_bounds(self) -> dict:
        """
        Get the boundary condition for the structural error mu parameters.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each structural error mu parameter in the actual mu mapping of the Model.

        Examples
        --------
        TODO FC: Fill
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_SERR_MU_PARAMETERS.items()
            if key in SERR_MU_MAPPING_PARAMETERS[self.setup.serr_mu_mapping]
        }

    def get_serr_sigma_parameters_bounds(self) -> dict:
        """
        Get the boundary condition for the structural error sigma parameters.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each structural error sigma parameter in the actual sigma mapping of the Model.

        Examples
        --------
        TODO FC: Fill
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_SERR_SIGMA_PARAMETERS.items()
            if key in SERR_SIGMA_MAPPING_PARAMETERS[self.setup.serr_sigma_mapping]
        }

    def get_serr_mu(self) -> np.ndarray:
        """
        Get the mu value by applying the mu mapping to mu parameters.

        Returns
        -------
        value : numpy.ndarray
            An array of shape *(ng, ntime_step)* representing the values of mu for each gauge and each timestep.

        Examples
        --------
        TODO FC: Fill
        """
        serr_mu = np.zeros(
            shape=(self.mesh.ng, self.setup.ntime_step), order="F", dtype=np.float32
        )
        wrap_get_serr_mu(self.setup, self.mesh, self._parameters, self._output, serr_mu)
        return serr_mu

    def get_serr_sigma(self) -> np.ndarray:
        """
        Get the sigma value by applying the sigma mapping to sigma parameters.

        Returns
        -------
        value : numpy.ndarray
            An array of shape *(ng, ntime_step)* representing the values of sigma for each gauge and each timestep.

        Examples
        --------
        TODO FC: Fill
        """
        serr_sigma = np.zeros(
            shape=(self.mesh.ng, self.setup.ntime_step), order="F", dtype=np.float32
        )
        wrap_get_serr_sigma(
            self.setup, self.mesh, self._parameters, self._output, serr_sigma
        )
        return serr_sigma

    def forward_run(
        self,
        cost_options: dict | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> ForwardRun | None:
        """
        Run the forward Model.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        cost_options : dict or None, default None
            Dictionary containing computation cost options for simulated and observed responses. The elements are:

            jobs_cmpt : str or ListLike, default 'nse'
                Type of observation objective function(s) to be computed. Should be one or a sequence of any of

                - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
                - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
                - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

                .. hint::
                    See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

            jobs_cmpt_tfm : str or ListLike, default 'keep'
                Type of transformation applied to discharge in observation objective function(s). Should be one or a sequence of any of

                - 'keep' : No transformation
                - 'sqrt' : Square root transformation
                - 'inv' : Multiplicative inverse transformation

            wjobs_cmpt : AlphaNumeric or ListLike, default 'mean'
                The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of **jobs_cmpt**.
                - An alias among 'mean'.

            gauge : str or ListLike, default 'dws'
                Type of gauge to be computed. There are two ways to specify it:

                - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
                - An alias among 'all' (all gauge codes) or 'dws' (most downstream gauge code(s)).

            wgauge : AlphaNumeric or ListLike, default 'mean'
                Type of gauge weights. There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of gauges optimized.
                - An alias among 'mean', 'lquartile' (1st quantile or lower quantile), 'median', or 'uquartile' (3rd quantile or upper quantile).

            event_seg : dict, default {'peak_quant': 0.995, 'max_duration': 240}
                A dictionary of event segmentation options when calculating flood event signatures for cost computation (i.e., **jobs_cmpt** includes flood events signatures).
                See `smash.hydrograph_segmentation` for more.

            end_warmup : str or pandas.Timestamp, default model.setup.start_time
                The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-05-23', '1998-05-24'])

                .. note::
                    It only applies to the following variables: 'rr_states' and 'q_domain'

            rr_states : bool, default False
                Whether to return rainfall-runoff states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_forward_run : ForwardRun or None, default None
            It returns a `smash.ForwardRun` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

        See Also
        --------
        smash.forward_run : Run the forward Model.
        ForwardRun : Represents forward run optional results.
        """

        args_options = [
            deepcopy(arg) for arg in [cost_options, common_options, return_options]
        ]

        args = _standardize_forward_run_args(self, *args_options)

        return _forward_run(self, *args)

    def optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict | None = None,
        cost_options: dict | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> Optimize | None:
        """
        Model assimilation using numerical optimization algorithms.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        mapping : str, default 'uniform'
            Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial', 'ann'.

        optimizer : str or None, default None
            Name of optimizer. Should be one of 'sbs', 'lbfgsb', 'sgd', 'adam', 'adagrad', 'rmsprop'.

            .. note::
                If not given, a default optimizer will be set depending on the optimization mapping:

                - **mapping** = 'uniform'; **optimizer** = 'sbs'
                - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'
                - **mapping** = 'ann'; **optimizer** = 'adam'

        optimize_options : dict or None, default None
            Dictionary containing optimization options for fine-tuning the optimization process.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See the returned parameters in `smash.default_optimize_options` for more.

        cost_options : dict or None, default None
            Dictionary containing computation cost options for simulated and observed responses. The elements are:

            jobs_cmpt : str or ListLike, default 'nse'
                Type of observation objective function(s) to be minimized. Should be one or a sequence of any of

                - 'nse', 'nnse', 'kge', 'mae', 'mape', 'mse', 'rmse', 'lgrm' (classical evaluation metrics)
                - 'Crc', 'Crchf', 'Crclf', 'Crch2r', 'Cfp2', 'Cfp10', 'Cfp50', 'Cfp90' (continuous signatures-based error metrics)
                - 'Eff', 'Ebf', 'Erc', 'Erchf', 'Erclf', 'Erch2r', 'Elt', 'Epf' (flood event signatures-based error metrics)

                .. hint::
                    See a detailed explanation on the objective function in :ref:`Math / Num Documentation <math_num_documentation.signal_analysis.cost_functions>` section.

            jobs_cmpt_tfm : str or ListLike, default 'keep'
                Type of transformation applied to discharge in observation objective function(s). Should be one or a sequence of any of

                - 'keep' : No transformation
                - 'sqrt' : Square root transformation
                - 'inv' : Multiplicative inverse transformation

            wjobs_cmpt : AlphaNumeric or ListLike, default 'mean'
                The corresponding weighting of observation objective functions in case of multi-criteria (i.e., a sequence of objective functions to compute). There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of **jobs_cmpt**.
                - An alias among 'mean'.

            wjreg : Numeric, default 0
                The weighting of regularization term. There are two ways to specify it:

                - A numeric value greater than or equal to 0
                - An alias among 'fast' or 'lcurve'. **wjreg** will be auto-computed by one of these methods.

            jreg_cmpt : str or ListLike, default 'prior'
                Type(s) of regularization function(s) to be minimized when regularization term is set (i.e., **wjreg** > 0). Should be one or a sequence of any of

                - 'prior' : Squared difference between control and control background
                - 'smoothing' : Spatial derivative **not** penalized by background
                - 'hard-smoothing' : Spatial derivative penalized by background

            wjreg_cmpt : AlphaNumeric or ListLike, default 'mean'
                The corresponding weighting of regularization functions in case of multi-regularization (i.e., a sequence of regularization functions to compute). There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of **jreg_cmpt**.
                - An alias among 'mean'.

            gauge : str or ListLike, default 'dws'
                Type of gauge to be computed. There are two ways to specify it:

                - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
                - An alias among 'all' (all gauge codes) or 'dws' (most downstream gauge code(s)).

            wgauge : AlphaNumeric or ListLike, default 'mean'
                Type of gauge weights. There are two ways to specify it:

                - A sequence of value whose size must be equal to the number of gauges optimized.
                - An alias among 'mean', 'lquartile' (1st quantile or lower quantile), 'median', or 'uquartile' (3rd quantile or upper quantile).

            event_seg : dict, default {'peak_quant': 0.995, 'max_duration': 240}
                A dictionary of event segmentation options when calculating flood event signatures for cost computation (i.e., **jobs_cmpt** includes flood events signatures).
                See `smash.hydrograph_segmentation` for more.

            end_warmup : str or pandas.Timestamp, default model.setup.start_time
                The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-05-23', '1998-05-24'])

                .. note::
                    It only applies to the following variables: 'rr_states' and 'q_domain'

            rr_states : bool, default False
                Whether to return rainfall-runoff states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            iter_cost : bool, default False
                Whether to return cost iteration values.

            iter_projg : bool, default False
                Whether to return infinity norm of the projected gradient iteration values.

            control_vector : bool, default False
                Whether to return control vector at end of optimization. In case of optimization with ANN-based mapping, the control vector is represented in `smash.factory.Net.layers` instead.

            net : Net, default False
                Whether to return the trained neural network `smash.factory.Net`. Only used with ANN-based mapping.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            jreg : bool, default False
                Whether to return jreg (regularization component of cost) value.

            lcurve_wjreg : bool, default False
                Whether to return the wjreg lcurve. Only used if **wjreg** in cost_options is equal to 'lcurve'.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_optimize : Optimize or None, default None
            It returns a `smash.Optimize` object containing the intermediate variables defined in **return_options**. If no intermediate variables
            are defined, it returns None.

        See Also
        --------
        smash.optimize : Model assimilation using numerical optimization algorithms.
        Optimize : Represents optimize optional results.
        """

        args_options = [
            deepcopy(arg)
            for arg in [optimize_options, cost_options, common_options, return_options]
        ]

        args = _standardize_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
        )

        return _optimize(self, *args)

    def multiset_estimate(
        self,
        multiset: MultipleForwardRun | MultipleOptimize,
        alpha: Numeric | ListLike | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> MultisetEstimate | None:
        """
        Model assimilation using Bayesian-like estimation on multiple sets of solutions.

        This method performs an in-place operation on the Model object.

        Parameters
        ----------
        multiset : MultipleForwardRun or MultipleOptimize
            The returned object created by the `smash.multiple_forward_run` or `smash.multiple_optimize` method containing information about multiple sets of rainfall-runoff parameters or initial states.

        alpha : Numeric, ListLike, or None, default None
            A regularization parameter that controls the decay rate of the likelihood function. If **alpha** is a list-like object, the L-curve approach will be used to find an optimal value for the regularization parameter.

            .. note:: If not given, a default numeric range will be set for optimization through the L-curve process.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-05-23', '1998-05-24'])

                .. note::
                    It only applies to the following variables: 'rr_states' and 'q_domain'

            rr_states : bool, default False
                Whether to return rainfall-runoff states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            cost : bool, default False
                Whether to return cost value.

            jobs : bool, default False
                Whether to return jobs (observation component of cost) value.

            lcurve_multiset : bool, default False
                Whether to return the multiset estimate lcurve.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_multiset_estimate : MultisetEstimate or None, default None
            It returns a `smash.MultisetEstimate` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

        See Also
        --------
        smash.multiset_estimate : Model assimilation using Bayesian-like estimation on multiple sets of solutions.
        MultisetEstimate : Represents multiset estimate optional results.
        MultipleForwardRun : Represents multiple forward run computation result.
        MultipleOptimize : Represents multiple optimize computation result.
        """

        args_options = [deepcopy(arg) for arg in [common_options, return_options]]

        args = _standardize_multiset_estimate_args(self, multiset, alpha, *args_options)

        return _multiset_estimate(self, *args)

    def bayesian_optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict | None = None,
        cost_options: dict | None = None,
        common_options: dict | None = None,
        return_options: dict | None = None,
    ) -> BayesianOptimize | None:
        """
        Model bayesian assimilation using numerical optimization algorithms.

        Parameters
        ----------
        mapping : str, default 'uniform'
            Type of mapping. Should be one of 'uniform', 'distributed', 'multi-linear', 'multi-polynomial'.

        optimizer : str or None, default None
            Name of optimizer. Should be one of 'sbs', 'lbfgsb'.

            .. note::
                If not given, a default optimizer will be set depending on the optimization mapping:

                - **mapping** = 'uniform'; **optimizer** = 'sbs'
                - **mapping** = 'distributed', 'multi-linear', or 'multi-polynomial'; **optimizer** = 'lbfgsb'

        optimize_options : dict or None, default None
            Dictionary containing optimization options for fine-tuning the optimization process.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element. See the returned parameters in `smash.default_optimize_options` for more.

        cost_options : dict or None, default None
            Dictionary containing computation cost options for simulated and observed responses. The elements are:

            gauge : str or ListLike, default 'dws'
                Type of gauge to be computed. There are two ways to specify it:

                - A gauge code or any sequence of gauge codes. The gauge code(s) given must belong to the gauge codes defined in the Model mesh.
                - An alias among 'all' (all gauge codes) or 'dws' (most downstream gauge code(s)).

            control_prior: dict or None, default None
                A dictionary containing the type of prior to link to control parameters. The keys are any control parameter name (i.e. 'cp0', 'cp1-1', 'cp-slope-a', etc), see `smash.bayesian_optimize_control_info` to retrieve control parameters names.
                The values are ListLike of length 2 containing distribution information (i.e. distribution name and parameters). Below, the set of available distributions and the associated number of parameters:

                - 'FlatPrior',   []                                 (0)
                - 'Uniform',     [lower_bound, higher_bound]        (2)
                - 'Gaussian',    [mean, standard_deviation]         (2)
                - 'Exponential', [threshold, scale]                 (2)
                - 'LogNormal',   [mean_log, standard_deviation_log] (2)
                - 'Triangle',    [peak, lower_bound, higher_bound]  (3)

                .. note:: If not given, a 'FlatPrior' is set to each control parameters (equivalent to no prior)

            end_warmup : str or pandas.Timestamp, default model.setup.start_time
                The end of the warm-up period, which must be between the start time and the end time defined in the Model setup. By default, it is set to be equal to the start time.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        common_options : dict or None, default None
            Dictionary containing common options with two elements:

            verbose : bool, default False
                Whether to display information about the running method.

            ncpu : bool, default 1
                Whether to perform a parallel computation.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        return_options : dict or None, default None
            Dictionary containing return options to save intermediate variables. The elements are:

            time_step : str, pandas.Timestamp, pandas.DatetimeIndex or ListLike, default 'all'
                Returned time steps. There are five ways to specify it:

                - A date as a character string which respect pandas.Timestamp format (i.e., '1997-12-21', '19971221', etc.).
                - An alias among 'all' (return all time steps).
                - A pandas.Timestamp object.
                - A pandas.DatetimeIndex object.
                - A sequence of dates as character string or pandas.Timestamp (i.e., ['1998-05-23', '1998-05-24'])

                .. note::
                    It only applies to the following variables: 'rr_states' and 'q_domain'

            rr_states : bool, default False
                Whether to return rainfall-runoff states for specific time steps.

            q_domain : bool, defaul False
                Whether to return simulated discharge on the whole domain for specific time steps.

            iter_cost : bool, default False
                Whether to return cost iteration values.

            iter_projg : bool, default False
                Whether to return infinity norm of the projected gardient iteration values.

            control_vector : bool, default False
                Whether to return control vector at end of optimization.

            cost : bool, default False
                Whether to return cost value.

            log_lkh : bool, default False
                Whether to return log likelihood component value.

            log_prior : bool, default False
                Whether to return log prior component value.

            log_h : bool, default False
                Whether to return log h component value.

            serr_mu : bool, default False
                Whether to return mu, the mean of structural errors.

            serr_sigma : bool, default False
                Whether to return sigma, the standard deviation of structural errors.

            .. note:: If not given, default values will be set for all elements. If a specific element is not given in the dictionary, a default value will be set for that element.

        Returns
        -------
        ret_bayesian_optimize : BayesianOptimize or None, default None
            It returns a `smash.BayesianOptimize` object containing the intermediate variables defined in **return_options**. If no intermediate variables are defined, it returns None.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        smash.bayesian_optimize : Model bayesian assimilation using numerical optimization algorithms.
        BayesianOptimize : Represents bayesian optimize optional results.
        """

        args_options = [
            deepcopy(arg)
            for arg in [optimize_options, cost_options, common_options, return_options]
        ]

        args = _standardize_bayesian_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
        )

        return _bayesian_optimize(self, *args)
