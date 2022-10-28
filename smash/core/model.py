from __future__ import annotations

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT
from smash.solver._mw_run import forward_run, adjoint_run, tangent_linear_run
from smash.solver._mw_adjoint_test import scalar_product_test, gradient_test

from smash.core._build_model import (
    _parse_derived_type,
    _build_setup,
    _build_mesh,
    _build_input_data,
)

from smash.core._optimize import (
    _standardize_optimize_args,
    _standardize_optimize_options,
    _optimize_sbs,
    _optimize_lbfgsb,
    _optimize_nelder_mead,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import Timestamp

import numpy as np

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.
    **S**\patially distributed **M**\odelling and **AS**\simillation for **H**\ydrology.
    """

    def __init__(self, setup: dict, mesh: dict):

        if setup or mesh:

            if isinstance(setup, dict):

                dn = setup.get("descriptor_name", [])

                self.setup = SetupDT(1 if isinstance(dn, str) else len(dn))

                _parse_derived_type(self.setup, setup)

            else:

                raise TypeError(
                    f"'setup' argument must be dictionary, not {type(setup)}"
                )

            _build_setup(self.setup)

            if isinstance(mesh, dict):

                self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

                _parse_derived_type(self.mesh, mesh)

            else:
                raise TypeError(f"'mesh' argument must be dictionary, not {type(mesh)}")

            _build_mesh(self.setup, self.mesh)

            self.input_data = Input_DataDT(self.setup, self.mesh)

            _build_input_data(self.setup, self.mesh, self.input_data)

            self.parameters = ParametersDT(self.mesh)

            self.states = StatesDT(self.mesh)

            self.output = OutputDT(self.setup, self.mesh)

            self._last_update = "Initialization"

    def __repr__(self):

        dim = f"Model dimension: (time: {self.setup._ntime_step}, nrow: {self.mesh.nrow}, ncol: {self.mesh.ncol})"
        last_update = f"Model last update: {self._last_update}"

        return f"{dim}\n{last_update}"

    @property
    def setup(self):

        """
        setup attr
        """

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

    @property
    def _last_update(self):

        return self.__last_update

    @_last_update.setter
    def _last_update(self, value):

        if isinstance(value, str):

            self.__last_update = value

        else:
            raise TypeError(
                f"'_last_update' attribute must be set with {str}, not {type(value)}"
            )

    def copy(self):

        copy = Model(None, None)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy.input_data = self.input_data.copy()
        copy.parameters = self.parameters.copy()
        copy.states = self.states.copy()
        copy.output = self.output.copy()
        copy._last_update = self._last_update

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
                instance.parameters.copy(),
                instance.states,
                instance.states.copy(),
                instance.output,
                True,
            )

            instance._last_update = "Forward Run"

        elif case == "adj":

            adjoint_run(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

            instance._last_update = "Adjoint Run"

        elif case == "tl":

            tangent_linear_run(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

            instance._last_update = "Tangent Linear Run"

        else:

            raise ValueError(
                f"'case' argument must be one of ['fwd', 'adj', 'tl'] not '{case}'"
            )

        if not inplace:

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

            instance._last_update = "Scalar Product Test"

        elif case == "gt":

            gradient_test(
                instance.setup,
                instance.mesh,
                instance.input_data,
                instance.parameters,
                instance.states,
                instance.output,
            )

            instance._last_update = "Gradient Test"

        else:

            raise ValueError(
                f"'case' argument must be one of ['spt', 'gt'] not '{case}'"
            )

        if not inplace:

            return instance

    def optimize(
        self,
        algorithm: str,
        control_vector: (str, list, tuple, set),
        jobs_fun: (str, None) = None,
        mapping: (str, None) = None,
        bounds: (list, tuple, set, None) = None,
        gauge: (str, list, tuple, set, None) = None,
        wgauge: (str, list, tuple, set, None) = None,
        ost: (str, Timestamp, None) = None,
        options: (dict, None) = None,
        inplace: bool = False,
    ):

        if inplace:

            instance = self

        else:

            instance = self.copy()

        (
            algorithm,
            control_vector,
            jobs_fun,
            mapping,
            bounds,
            wgauge,
            ost,
        ) = _standardize_optimize_args(
            algorithm,
            control_vector,
            jobs_fun,
            mapping,
            bounds,
            gauge,
            wgauge,
            ost,
            instance.setup,
            instance.mesh,
            instance.input_data,
        )

        options = _standardize_optimize_options(options)

        if algorithm == "sbs":

            _optimize_sbs(
                instance,
                control_vector,
                jobs_fun,
                mapping,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "Step By Step Optimization"

        elif algorithm == "l-bfgs-b":

            _optimize_lbfgsb(
                instance,
                control_vector,
                jobs_fun,
                mapping,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "L-BFGS-B Optimization"

        elif algorithm == "nelder-mead":

            _optimize_nelder_mead(
                instance,
                control_vector,
                jobs_fun,
                mapping,
                bounds,
                wgauge,
                ost,
                **options,
            )

            instance._last_update = "Nelder-Mead Optimization"

        #% TODO
        # elif algorithm == "nsga":

        if not inplace:

            return instance
