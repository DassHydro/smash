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
from smash.core.simulation.optimize.optimize import _optimize

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT
from smash.fcore._mwd_options import OptionsDT
from smash.fcore._mwd_returns import ReturnsDT

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import Numeric

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

    def get_opr_parameters(self, key: str):
        key = _standardize_get_opr_parameters_args(self, key)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        return self._parameters.opr_parameters.values[..., ind]

    def set_opr_parameters(self, key: str, value: Numeric | np.ndarray):
        key, value = _standardize_set_opr_parameters_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_parameters.keys == key).item()

        self._parameters.opr_parameters.values[..., ind] = value

    def get_opr_initial_states(self, key: str):
        key = _standardize_get_opr_initial_states_args(self, key)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        return self._parameters.opr_initial_states.values[..., ind]

    def set_opr_initial_states(self, key: str, value: Numeric | np.ndarray):
        key, value = _standardize_set_opr_initial_states_args(self, key, value)
        ind = np.argwhere(self._parameters.opr_initial_states.keys == key).item()

        self._parameters.opr_initial_states.values[..., ind] = value

    def get_opr_final_states(self, key: str):
        key = _standardize_get_opr_final_states_args(self, key)
        ind = np.argwhere(self._output.opr_final_states.keys == key).item()

        return self._output.opr_final_states.values[..., ind]

    def get_opr_parameters_bounds(self):
        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_PARAMETERS.items()
            if key in STRUCTURE_OPR_PARAMETERS[self.setup.structure]
        }

    def get_opr_initial_states_bounds(self):
        return {
            key: value
            for key, value in DEFAULT_BOUNDS_OPR_INITIAL_STATES.items()
            if key in STRUCTURE_OPR_STATES[self.setup.structure]
        }

    def forward_run(
        self, options: OptionsDT | None = None, returns: ReturnsDT | None = None
    ):
        _forward_run(self, options, returns)

    def optimize(
        self, options: OptionsDT | None = None, returns: ReturnsDT | None = None
    ):
        _optimize(self, options, returns)
