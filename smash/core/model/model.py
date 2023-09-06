from __future__ import annotations

from smash._constant import (
    STRUCTURE_OPR_PARAMETERS,
    STRUCTURE_OPR_STATES,
    DEFAULT_BOUNDS_OPR_PARAMETERS,
    DEFAULT_BOUNDS_OPR_INITIAL_STATES,
)

from smash.core.model._build_model import (
    _map_dict_to_object,
    _build_setup,
    _build_mesh,
    _build_input_data,
    _build_parameters,
    _build_output,
)
from smash.core.model._standardize import (
    _standardize_get_opr_parameters_args,
    _standardize_get_opr_initial_states_args,
    _standardize_get_opr_final_states_args,
    _standardize_set_opr_parameters_args,
    _standardize_set_opr_initial_states_args,
)
from smash.core.simulation.run.run import _forward_run
from smash.core.simulation.run._standardize import _standardize_forward_run_args
from smash.core.simulation.optimize.optimize import _optimize
from smash.core.simulation.optimize._standardize import _standardize_optimize_args
from smash.core.simulation.estimate._standardize import (
    _standardize_multiset_estimate_args,
)
from smash.core.simulation.estimate.estimate import _multiset_estimate

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT

import numpy as np

from copy import deepcopy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric, ListLike
    from smash.core.simulation.optimize.optimize import MultipleOptimize
    from smash.core.simulation.run.run import MultipleForwardRun

__all__ = ["Model"]


class Model(object):
        
    """
    Primary data structure of the hydrological model `smash`.

    Parameters
    ----------
    setup : dict
        Model initialization setup dictionary (TODO: add reference).

    mesh : dict
        Model initialization mesh dictionary (TODO: add reference).

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
    </> Reading precipitation: 100%|█████████| 1440/1440 [00:00<00:00, 10512.83it/s]
    </> Reading daily interannual pet: 100%|███| 366/366 [00:00<00:00, 13638.56it/s]
    </> Disaggregating daily interannual pet: 100%|█| 1440/1440 [00:00<00:00, 129442
    """
        
    def __init__(self, setup: dict | None, mesh: dict | None):
        if setup and mesh:
            if isinstance(setup, dict):
                nd = np.array(setup.get("descriptor_name", [])).size

                self.setup = SetupDT(nd)

                _map_dict_to_object(setup, self.setup)

                _build_setup(self.setup)

            else:
                raise TypeError(f"setup argument must be dict")

            if isinstance(mesh, dict):
                self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

                _map_dict_to_object(mesh, self.mesh)

                _build_mesh(self.setup, self.mesh)

            else:
                raise TypeError(f"mesh argument must be dict")

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

    @property
    def setup(self):

        """
        The setup used to create the Model object.

        TODO: Fill
        """

        return self._setup

    @setup.setter
    def setup(self, value):
        self._setup = value

    @property
    def mesh(self):

        """
        The mesh used to create the Model object.

        TODO: Fill
        """

        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def obs_response(self):

        """
        Observation response data.

        TODO: Fill
        """

        return self._input_data.obs_response

    @obs_response.setter
    def obs_response(self, value):
        self._input_data.obs_response = value

    @property
    def physio_data(self):

        """
        Physiographic data.

        TODO: Fill
        """

        return self._input_data.physio_data

    @physio_data.setter
    def physio_data(self, value):
        self._input_data.physio_data = value

    @property
    def atmos_data(self):

        """
        Atmospheric and meteorological data.

        TODO: Fill
        """

        return self._input_data.atmos_data

    @atmos_data.setter
    def atmos_data(self, value):
        self._input_data.atmos_data = value

    @property
    def opr_parameters(self):

        """
        Get operator parameters for the actual structure of the Model.

        TODO: Fill
        """
        
        return self._parameters.opr_parameters

    @opr_parameters.setter
    def opr_parameters(self, value):
        self._parameters.opr_parameters = value

    @property
    def opr_initial_states(self):

        """
        Get operator initial states for the actual structure of the Model.

        TODO: Fill
        """
                
        return self._parameters.opr_initial_states

    @opr_initial_states.setter
    def opr_initial_states(self, value):
        self._parameters.opr_initial_states = value

    @property
    def sim_response(self):

        """
        Simulated response data.

        TODO: Fill
        """

        return self._output.sim_response

    @sim_response.setter
    def sim_response(self, value):
        self._output.sim_response = value

    @property
    def opr_final_states(self):

        """
        Get operator final states for the actual structure of the Model.

        TODO: Fill
        """

        return self._output.opr_final_states

    @opr_final_states.setter
    def opr_final_states(self, value):
        self._output.opr_final_states = value

    def copy(self):

        """
        Make a deepcopy of the Model.

        Returns
        -------
        Model
            A copy of Model.

        Examples
        --------
        TODO: Fill
        """

        return self.__copy__()

    def get_opr_parameters(self, key: str):

        """
        Get the values of an operator model parameter.

        Parameters
        ----------
        key : str
            The name of the operator parameter.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator parameter.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_parameters : Get operator parameters for the actual structure of the Model.
        """
        
        key = _standardize_get_opr_parameters_args(self, key)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        return self._parameters.opr_parameters.values[..., ind]

    def set_opr_parameters(self, key: str, value: Numeric | np.ndarray):

        """
        Set the values for an operator model parameter.

        Parameters
        ----------
        key : str
            The name of the operator parameter.

        value : Numeric or np.ndarray
            The values to set for the operator parameter.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_parameters : Get operator parameters for the actual structure of the Model.
        """

        key, value = _standardize_set_opr_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        self._parameters.opr_parameters.values[..., ind] = value

    def get_opr_initial_states(self, key: str):

        """
        Get the values of an operator model initial state.

        Parameters
        ----------
        key : str
            The name of the operator initial state.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator initial state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_initial_states : Get operator initial states for the actual structure of the Model.
        """

        key = _standardize_get_opr_initial_states_args(self, key)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        return self._parameters.opr_initial_states.values[..., ind]

    def set_opr_initial_states(self, key: str, value: Numeric | np.ndarray):

        """
        Set the values for an operator model initial state.

        Parameters
        ----------
        key : str
            The name of the operator initial state.

        value : Numeric or np.ndarray
            The values to set for the operator initial state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_initial_states : Get operator initial states for the actual structure of the Model.
        """

        key, value = _standardize_set_opr_initial_states_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        self._parameters.opr_initial_states.values[..., ind] = value

    def get_opr_final_states(self, key: str):

        """
        Get the values of an operator model final state.

        Parameters
        ----------
        key : str
            The name of the operator final state.

        Returns
        -------
        value : np.ndarray
            A 2D-array representing the values of the operator final state.

        Examples
        --------
        TODO: Fill

        See Also
        --------
        Model.opr_final_states : Get operator final states for the actual structure of the Model.
        """

        key = _standardize_get_opr_final_states_args(self, key)
        ind = np.argwhere(self._output.opr_final_states.keys == key).item()

        return self._output.opr_final_states.values[..., ind]

    def get_opr_parameters_bounds(self):

        """
        Get the boundary condition for the operator model parameters.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each operator parameter in the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_PARAMETERS.items()
            if key in STRUCTURE_OPR_PARAMETERS[self.setup.structure]
        }

    def get_opr_initial_states_bounds(self):

        """
        Get the boundary condition for the operator model initial states.

        Returns
        -------
        bounds : dict
            A dictionary representing the boundary condition for each operator initial state in the actual structure of the Model.

        Examples
        --------
        TODO: Fill
        """    

        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_INITIAL_STATES.items()
            if key in STRUCTURE_OPR_STATES[self.setup.structure]
        }

    def forward_run(
        self,
        cost_options: dict | None = None,
        common_options: dict | None = None,
    ):
        
        """
        Run the forward Model.

        TODO: Fill
        """

        args_options = [deepcopy(arg) for arg in [cost_options, common_options]]

        args = _standardize_forward_run_args(self, *args_options)

        _forward_run(self, *args)

    def optimize(
        self,
        mapping: str = "uniform",
        optimizer: str | None = None,
        optimize_options: dict | None = None,
        cost_options: dict | None = None,
        common_options: dict | None = None,
    ):

        """
        Model assimilation using numerical optimization algorithms.

        TODO: Fill
        """

        args_options = [
            deepcopy(arg) for arg in [optimize_options, cost_options, common_options]
        ]

        args = _standardize_optimize_args(
            self,
            mapping,
            optimizer,
            *args_options,
        )

        _optimize(self, *args)

    def multiset_estimate(
        self,
        multiset: MultipleForwardRun | MultipleOptimize,
        alpha: Numeric | ListLike | None = None,
        common_options: dict | None = None,
    ):
        
        """
        Model assimilation using a Bayesian-like estimation method with multiple sets of operator parameters or/and initial states.

        TODO: Fill
        """
        
        arg_options = deepcopy(common_options)

        args = _standardize_multiset_estimate_args(multiset, alpha, arg_options)

        _multiset_estimate(self, *args)
