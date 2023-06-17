from __future__ import annotations

from smash.tools._common_function import _map_dict_to_object

from smash.core._build_model import (
    _build_setup,
    _build_mesh,
    _build_input_data,
    _build_parameters,
)

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_output import OutputDT

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
            self.obs_response = self._input_data.obs_response
            self.physio_data = self._input_data.physio_data
            self.atmos_data = self._input_data.atmos_data

            _build_input_data(self.setup, self.mesh, self._input_data)

            self._parameters = ParametersDT(self.setup, self.mesh)
            self.opr_parameters = getattr(
                self._parameters.opr_parameters, self.setup.structure
            )
            self.opr_initial_states = getattr(
                self._parameters.opr_initial_states, self.setup.structure
            )

            _build_parameters(self.setup, self.mesh, self._input_data, self._parameters)

            self._output = OutputDT(self.setup, self.mesh)
            self.sim_response = self._output.sim_response
            self.opr_final_states = getattr(
                self._output.opr_final_states, self.setup.structure
            )

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
        return self._obs_response

    @obs_response.setter
    def obs_response(self, value):
        self._obs_response = value

    @property
    def physio_data(self):
        return self._physio_data

    @physio_data.setter
    def physio_data(self, value):
        self._physio_data = value

    @property
    def atmos_data(self):
        return self._atmos_data

    @atmos_data.setter
    def atmos_data(self, value):
        self._atmos_data = value

    @property
    def opr_parameters(self):
        return self._opr_parameters

    @opr_parameters.setter
    def opr_parameters(self, value):
        self._opr_parameters = value

    @property
    def opr_initial_states(self):
        return self._opr_initial_states

    @opr_initial_states.setter
    def opr_initial_states(self, value):
        self._opr_initial_states = value

    @property
    def sim_response(self):
        return self._sim_response

    @sim_response.setter
    def sim_response(self, value):
        self._sim_response = value

    @property
    def opr_final_states(self):
        return self._opr_final_states

    @opr_final_states.setter
    def opr_final_states(self, value):
        self._opr_final_states = value

    def copy(self):
        return self.__copy__()
