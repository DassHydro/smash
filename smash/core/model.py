from __future__ import annotations

from smash.tools._common_function import _map_dict_to_object

from smash.core._build_model import (
    _build_setup,
    _build_mesh,
    _build_input_data,
    _build_paramstates,
)

from smash.tools._common_function import _default_bound_constraints

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mwd_options import OptionsDT
from smash.solver._mwd_returns import ReturnsDT

from smash.solver._mw_optimize import sbs_optimize, lbfgsb_optimize
from smash.solver._mw_forward import forward_run as fw_run

from smash.simulation.run.run import _forward_run

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

            _build_paramstates(self.setup, self.mesh, self._input_data, self._parameters)

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
    
    def forward_run(self, options: OptionsDT | None = None, returns: ReturnsDT | None = None):

        _forward_run(self, options, returns)

    def optimize(self):
        options = OptionsDT(self.setup)
        returns = ReturnsDT()

        options.comm.ncpu = 6

        fw_run(
            self.setup,
            self.mesh,
            self._input_data,
            self._parameters,
            self._output,
            options,
            returns,
        )
        # ci, cp, cft, cst, kexc, llr, akw, bkw
        options.optimize.opr_parameters = [0, 1, 1, 0, 1, 0, 1, 1]
        options.optimize.opr_initial_states = [0, 0, 0, 0, 0]
        options.optimize.l_opr_parameters = [
            1e-6,
            1e-6,
            1e-6,
            1e-6,
            -50,
            1e-6,
            1e-3,
            1e-3,
        ]
        options.optimize.u_opr_parameters = [100, 1000, 1000, 10_000, 50, 1000, 50, 1]
        options.optimize.l_opr_initial_states = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
        options.optimize.u_opr_initial_states = [
            0.999999,
            0.999999,
            0.999999,
            0.999999,
            1000,
        ]

        # ~ options.optimize.optimizer = "sbs"
        # ~ options.optimize.mapping = "uniform"
        # ~ options.optimize.maxiter = 5
        # ~ options.optimize.control_tfm = "sbs"

        # ~ optimize_func = eval(options.optimize.optimizer + "_optimize")

        # ~ optimize_func(
        # ~ self.setup,
        # ~ self.mesh,
        # ~ self._input_data,
        # ~ self._parameters,
        # ~ self._output,
        # ~ options,
        # ~ returns,
        # ~ )

        options.optimize.optimizer = "lbfgsb"
        options.optimize.mapping = "distributed"
        options.optimize.maxiter = 10
        options.optimize.control_tfm = "normalize"

        optimize_func = eval(options.optimize.optimizer + "_optimize")

        optimize_func(
            self.setup,
            self.mesh,
            self._input_data,
            self._parameters,
            self._output,
            options,
            returns,
        )

        # ~ options.optimize.optimizer = "lbfgsb"
        # ~ options.optimize.mapping = "multi-linear"
        # ~ options.optimize.maxiter = 100
        # ~ options.optimize.control_tfm = "normalize"
        # ~ opd = np.ones(
        # ~ shape=options.optimize.opr_parameters_descriptor.shape,
        # ~ dtype=np.int32,
        # ~ order="F",
        # ~ )
        # ~ #opd[:, 5] = 0
        # ~ options.optimize.opr_parameters_descriptor = opd
        # ~ options.optimize.opr_initial_states_descriptor = 0

        # ~ optimize_func = eval(options.optimize.optimizer + "_optimize")

        # ~ optimize_func(
        # ~ self.setup,
        # ~ self.mesh,
        # ~ self._input_data,
        # ~ self._parameters,
        # ~ self._output,
        # ~ options,
        # ~ returns,
        # ~ )

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
