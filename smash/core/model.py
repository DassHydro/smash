from __future__ import annotations

from smash.wrapping.m_setup import SetupDT
from smash.wrapping.m_mesh import MeshDT

from smash.io._configuration import _read_yaml_configuration

from smash.core._utils import _derived_type_parser, _build_setup

__all__ = ["Model"]


class Model(object):

    """
    Primary data structure of the hydrological model `smash`.
    **S**\patially distributed **M**\odelling and **AS**\simillation for **H**\ydrology.
    """

    def __init__(
        self,
        configuration: (str, None) = None,
        setup: (dict, None) = None,
    ):

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
        
        self.mesh = MeshDT(self.setup, 10, 10, 1)

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
