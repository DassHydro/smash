from __future__ import annotations

from smash.tools._common_function import _map_dict_to_object, _default_bound_constraints

from smash.core._build_model import (
    _build_setup,
    _build_mesh,
    _build_input_data,
    _build_parameters,
)
from smash.core._standardize import (
    _standardize_parameters,
    _standardize_states,
    _standardize_parameter_name,
    _standardize_state_name,
)

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mwd_options import OptionsDT
from smash.solver._mwd_returns import ReturnsDT

from smash.simulation.run.run import _forward_run
from smash.simulation.optimize.optimize import _optimize

import numpy as np

__all__ = ["Model"]


class Model(object):
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

            self._parameters = ParametersDT(self.mesh)

            _build_parameters(self.setup, self.mesh, self._input_data, self._parameters)

            self._output = OutputDT(self.setup, self.mesh)

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
        return self._setup

    @setup.setter
    def setup(self, value):
        self._setup = value

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def obs_response(self):
        return self._input_data.obs_response

    @obs_response.setter
    def obs_response(self, value):
        self._input_data.obs_response = value

    @property
    def physio_data(self):
        return self._input_data.physio_data

    @physio_data.setter
    def physio_data(self, value):
        self._input_data.physio_data = value

    @property
    def atmos_data(self):
        return self._input_data.atmos_data

    @atmos_data.setter
    def atmos_data(self, value):
        self._input_data.atmos_data = value

    @property
    def opr_parameters(self):
        return self._parameters.opr_parameters

    @opr_parameters.setter
    def opr_parameters(self, value):
        self._parameters.opr_parameters = value

    @property
    def opr_initial_states(self):
        return self._parameters.opr_initial_states

    @opr_initial_states.setter
    def opr_initial_states(self, value):
        self._parameters.opr_initial_states = value

    @property
    def sim_response(self):
        return self._output.sim_response

    @sim_response.setter
    def sim_response(self, value):
        self._output.sim_response = value

    @property
    def opr_final_states(self):
        return self._output.opr_final_states

    @opr_final_states.setter
    def opr_final_states(self, value):
        self._output.opr_final_states = value

    def copy(self):
        return self.__copy__()

    def forward_run(
        self, options: OptionsDT | None = None, returns: ReturnsDT | None = None
    ):
        _forward_run(self, options, returns)

    def optimize(
        self, options: OptionsDT | None = None, returns: ReturnsDT | None = None
    ):
        _optimize(self, options, returns)

    def set_parameters(self, parameters: dict):
        mesh_shape = (self.mesh.nrow, self.mesh.ncol)

        parameters = _standardize_parameters(parameters, mesh_shape)

        for key, value in parameters.items():
            setattr(self._parameters.opr_parameters, key, value)

    def set_initial_states(self, states: dict):
        mesh_shape = (self.mesh.nrow, self.mesh.ncol)

        states = _standardize_states(states, mesh_shape)

        for key, value in states.items():
            setattr(self._parameters.opr_initial_states, key, value)

    def get_parameter(self, parameter: str):
        parameter = _standardize_parameter_name(parameter)

        return getattr(self._parameters.opr_parameters, parameter)

    def get_initial_state(self, state: str):
        state = _standardize_state_name(state)

        return getattr(self._parameters.opr_initial_states, state)

    def get_final_state(self, state: str):
        state = _standardize_state_name(state)

        return getattr(self._output.opr_final_states, state)

    def default_bound_constraints(self, states: bool = False):
        """
        Get the boundary default constraints of the Model parameters/states.

        Parameters
        ----------
        states : bool, default True
            If True, return boundary constraints of the Model states instead of Model parameters.

        Returns
        -------
        problem : dict
            The boundary constraint problem of the Model parameters/states. The keys are

            - 'num_vars': The number of Model parameters/states.
            - 'names': The name of Model parameters/states.
            - 'bounds': The upper and lower bounds of each Model parameters/states (a sequence of ``(min, max)``).

        Examples
        --------
        >>> setup, mesh = smash.factory.load_dataset("cance")
        >>> model = smash.Model(setup, mesh)
        >>> problem = model.default_bound_constraints()
        >>> problem
        {
            'num_vars': 4,
            'names': ['cp', 'cft', 'exc', 'lr'],
            'bounds': [[1e-06, 1000], [1e-06, 1000], [-50, 50], [1e-06, 1000]]
        }

        """

        return _default_bound_constraints(self.setup, states)
