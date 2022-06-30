from __future__ import annotations

import numpy as np

from smash.solver.mw_setup import SetupDT
from smash.solver.mw_mesh import MeshDT
from smash.solver.mw_input_data import Input_DataDT
from smash.solver.mw_parameters import ParametersDT
from smash.solver.mw_states import StatesDT
from smash.solver.mw_output import OutputDT
from smash.solver.mw_run import forward_run, adjoint_run, tangent_linear_run
from smash.solver.mw_validate import scalar_product_test, gradient_test_adj
from smash.solver.mw_optimize import optimize_sbs, optimize_lbfgsb

from smash.io.yaml import read_yaml_configuration

from smash.core._build_derived_type import (
    _derived_type_parser,
    _build_setup,
    _build_mesh,
    _build_input_data,
)

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.
    **S**\patially distributed **M**\odelling and **AS**\simillation for **H**\ydrology.
    """

    def __init__(
        self,
        configuration: (str, None) = None,
        mesh: (dict, None) = None,
        build: bool = True,
    ):

        if build:
            
            self.setup = SetupDT()

            if configuration is None:
                
                raise ValueError(
                        f"configuration argument must be defined, if build is True"
                    )
            else:
                
                if isinstance(configuration, str):
                    _derived_type_parser(
                        self.setup, read_yaml_configuration(configuration)
                    )

                else:
                    
                    raise TypeError(
                        f"configuration argument must be string, not {type(configuration)}"
                    )

            _build_setup(self.setup)
            
            if mesh is None:
                
                raise ValueError(
                        f"mesh argument must be defined, if build is True"
                    )
            else:
                
                if isinstance(mesh, dict):

                    self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

                    _derived_type_parser(self.mesh, mesh)

                else:
                    raise TypeError(f"mesh argument must be dictionary, not {type(mesh)}")

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
                f"setup attribute must be set with {type(SetupDT())}, not {type(value)}"
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
                f"mesh attribute must be set with {type(MeshDT())}, not {type(value)}"
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
                f"input_data attribute must be set with {type(Input_DataDT())}, not {type(value)}"
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
                f"parameters attribute must be set with {type(ParametersDT())}, not {type(value)}"
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
                f"states attribute must be set with {type(StatesDT())}, not {type(value)}"
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
                f"output attribute must be set with {type(OutputDT())}, not {type(value)}"
            )

    def copy(self):

        copy = Model(build=False)
        copy.setup = self.setup.copy()
        copy.mesh = self.mesh.copy()
        copy.input_data = self.input_data.copy()
        copy.paramters = self.parameters.copy()
        copy.states = self.states.copy()
        copy.output = self.output.copy()

        return copy
        
    
    def run(self, case: str = "fwd", inplace: bool = False):
        
        if inplace:
            
            instance = self
            
        else:
            
            instance = self.copy()
            
        
        if case == "fwd":
            
            cost = np.float32(0.0)
            
            forward_run(
                self.setup,
                self.mesh,
                self.input_data, 
                self.parameters,
                self.states,
                self.output,
                cost)
                
        elif case == "adj":
            
            cost = np.float32(0.0)
            
            adjoint_run(
                self.setup,
                self.mesh,
                self.input_data, 
                self.parameters,
                self.states,
                self.output,
                cost)
                
        else:
            
            raise ValueError(f"case must be one of ['fwd', 'adj', 'tl'] not {case}")
                
            
    def validate(self, case: str = "spt"):
        
        if case == "spt":
            
            cost = np.float32(0.0)
            
            scalar_product_test(
                self.setup,
                self.mesh,
                self.input_data,
                self.parameters,
                self.states,
                self.output,
                cost
            )
            
        elif case == "gt_adj":
            
            cost = np.float32(0.0)
            
            gradient_test_adj(
                self.setup,
                self.mesh,
                self.input_data,
                self.parameters,
                self.states,
                self.output,
                cost
            )
    
    

    def optimize(self, solver: str = "sbs", inplace: bool = False):

        if inplace:

            instance = self

        else:

            instance = self.copy()
            
        if solver == "sbs":
            
            cost = np.float32(0.0)

            optimize_sbs(
                self.setup,
                self.mesh,
                self.input_data,
                self.parameters,
                self.states,
                self.output,
                cost,
            )
        
        elif solver == "l-bfgs-b":
            
            cost = np.float32(0.0)
            
            optimize_lbfgsb(
                self.setup,
                self.mesh,
                self.input_data,
                self.parameters,
                self.states,
                self.output,
                cost,
            )
