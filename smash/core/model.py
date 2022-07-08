from __future__ import annotations

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mw_run import forward_run, adjoint_run, tangent_linear_run
from smash.solver._mw_adjoint_test import scalar_product_test, gradient_test
from smash.solver._mw_optimize import optimize_sbs, optimize_lbfgsb

from smash.core._build_derived_type import (
    _parse_derived_type,
    _build_setup,
    _build_mesh,
    _build_input_data,
)

import numpy as np


__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.
    **S**\patially distributed **M**\odelling and **AS**\simillation for **H**\ydrology.
    """

    def __init__(
        self,
        setup: (str, None) = None,
        mesh: (dict, None) = None,
        build: bool = True,
    ):

        if build:

            self.setup = SetupDT()

            if setup is None:

                raise ValueError(f"'setup' argument must be defined")

            else:

                if isinstance(setup, dict):
                    _parse_derived_type(self.setup, setup)

                else:

                    raise TypeError(
                        f"'setup' argument must be dictionary, not {type(setup)}"
                    )

            _build_setup(self.setup)

            if mesh is None:

                raise ValueError(f"'mesh' argument must be defined")
            else:

                if isinstance(mesh, dict):

                    self.mesh = MeshDT(
                        self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"]
                    )

                    _parse_derived_type(self.mesh, mesh)

                else:
                    raise TypeError(
                        f"'mesh' argument must be dictionary, not {type(mesh)}"
                    )

            _build_mesh(self.setup, self.mesh)

            self.input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self.input_data)

            self.parameters = ParametersDT(self.setup, self.mesh)

            self.states = StatesDT(self.setup, self.mesh)

            self.output = OutputDT(self.setup, self.mesh)

    @property
    def setup(self):

        return self._setup

    @setup.setter
    def setup(self, value):

        if isinstance(value, SetupDT):
            self._setup = value

        else:
            raise TypeError(
                f"'setup' attribute must be set with {type(SetupDT())}, not {type(value)}"
            )

    @property
    def mesh(self):

        return self._mesh

    @mesh.setter
    def mesh(self, value):

        if isinstance(value, MeshDT):
            self._mesh = value

        else:
            raise TypeError(
                f"'mesh' attribute must be set with {type(MeshDT())}, not {type(value)}"
            )

    @property
    def input_data(self):

        return self._input_data

    @input_data.setter
    def input_data(self, value):

        if isinstance(value, Input_DataDT):
            self._input_data = value

        else:
            raise TypeError(
                f"'input_data' attribute must be set with {type(Input_DataDT())}, not {type(value)}"
            )

    @property
    def parameters(self):

        return self._parameters

    @parameters.setter
    def parameters(self, value):

        if isinstance(value, ParametersDT):

            self._parameters = value

        else:
            raise TypeError(
                f"'parameters' attribute must be set with {type(ParametersDT())}, not {type(value)}"
            )

    @property
    def states(self):

        return self._states

    @states.setter
    def states(self, value):

        if isinstance(value, StatesDT):

            self._states = value

        else:
            raise TypeError(
                f"'states' attribute must be set with {type(StatesDT())}, not {type(value)}"
            )

    @property
    def output(self):

        return self._output

    @output.setter
    def output(self, value):

        if isinstance(value, OutputDT):

            self._output = value

        else:
            raise TypeError(
                f"'output' attribute must be set with {type(OutputDT())}, not {type(value)}"
            )

    def copy(self):

        copy = Model(build=False)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy.input_data = self.input_data.copy()
        copy.parameters = self.parameters.copy()
        copy.states = self.states.copy()
        copy.output = self.output.copy()

        return copy

    def run(self, case: str = "fwd", inplace: bool = False):

        if inplace:

            instance = self

        else:

            instance = self.copy()

        if case == "fwd":

            forward_run(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        elif case == "adj":

            adjoint_run(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        elif case == "tl":

            tangent_linear_run(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        else:

            raise ValueError(
                f"'case' argument must be one of ['fwd', 'adj', 'tl'] not '{case}'"
            )

        return instance

    def adjoint_test(self, case: str = "spt", inplace: bool = False):

        if inplace:

            instance = self

        else:

            instance = self.copy()

        if case == "spt":

            scalar_product_test(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        elif case == "gt":

            gradient_test(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        else:

            raise ValueError(
                f"'case' argument must be one of ['spt', 'gt'] not '{case}'"
            )

        return instance

    def optimize(self, algorithm: str = "sbs", inplace: bool = False):

        if inplace:

            instance = self

        else:

            instance = self.copy()

        if algorithm == "sbs":

            optimize_sbs(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        elif algorithm == "l-bfgs-b":

            optimize_lbfgsb(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

        return instance
