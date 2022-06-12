from __future__ import annotations

from smash.solver.m_setup import SetupDT
from smash.solver.m_mesh import MeshDT
from smash.solver.m_input_data import Input_DataDT

from smash.io._configuration import _read_yaml_configuration

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
        mesh: (dict),
        configuration: (str, None) = None,
        setup: (dict, None) = None,
    ):

        if configuration is None and setup is None:
            raise ValueError(
                f"At least one of configuration or setup argument must be specified"
            )

        self.setup = SetupDT()

        if configuration is not None:

            if isinstance(configuration, str):
                _derived_type_parser(
                    self.setup, _read_yaml_configuration(configuration)
                )

            else:
                raise TypeError(
                    f"configuration argument must be string, not {type(configuration)}"
                )

        if setup is not None:

            if isinstance(setup, dict):
                _derived_type_parser(self.setup, setup)

            else:
                raise TypeError(f"setup argument must be dictionary, not {type(setup)}")

        _build_setup(self.setup)

        if isinstance(mesh, dict):

            self.mesh = MeshDT(self.setup, mesh["nrow"], mesh["ncol"], mesh["ng"])

            _derived_type_parser(self.mesh, mesh)

        else:
            raise TypeError(f"mesh argument must be dictionary, not {type(mesh)}")

        _build_mesh(self.setup, self.mesh)

        self.input_data = Input_DataDT(self.setup, self.mesh)

        _build_input_data(self.setup, self.mesh, self.input_data)

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
