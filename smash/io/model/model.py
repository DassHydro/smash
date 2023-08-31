from __future__ import annotations

from smash.io._error import ReadHDF5MethodError

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT

from smash.core.model._build_model import _build_mesh
from smash.io.model._parse import (
    _parse_derived_type_to_hdf5,
    _parse_hdf5_to_derived_type,
)

import os
import errno
import h5py
import pandas as pd

import smash

__all__ = ["save_model", "read_model"]


def save_model(model: Model, path: str):
    """
    Save Model object.

    Parameters
    ----------
    model : Model
        The Model object to be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file.

    path : str
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file path.

    See Also
    --------
    read_model: Read Model object.
    Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> setup, mesh = smash.factory.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save Model

    >>> smash.io.save_model(model, "model.hdf5")

    Read Model

    >>> model_rld = smash.io.read_model("model.hdf5")
    """

    if not path.endswith(".hdf5"):
        path = path + ".hdf5"

    with h5py.File(path, "w") as f:
        for derived_type_key in [
            "setup",
            "mesh",
            "atmos_data",
            "physio_data",
            "obs_response",
            "sim_response",
            "opr_parameters",
            "opr_initial_states",
            "opr_final_states",
        ]:
            derived_type = getattr(model, derived_type_key)

            grp = f.create_group(derived_type_key)

            _parse_derived_type_to_hdf5(derived_type, grp)

        f.attrs["_save_func"] = "save_model"


# % TODO: enhance Model initialization, using setup and mesh which ara not None (disable read data in setup)
def read_model(path: str) -> Model:
    """
    Read Model object.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    Model :
        A Model object loaded from HDF5 file.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_model`.

    See Also
    --------
    save_model: Save Model object.
    Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> setup, mesh = smash.factory.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save Model

    >>> smash.io.save_model(model, "model.hdf5")

    Read Model

    >>> model_rld = smash.io.read_model("model.hdf5")
    """

    if os.path.isfile(path):
        with h5py.File(path, "r") as f:
            if f.attrs.get("_save_func") == "save_model":
                instance = smash.Model(None, None)

                if "descriptor_name" in f["setup"].keys():
                    nd = f["setup"]["descriptor_name"].size

                else:
                    nd = 0

                instance.setup = SetupDT(nd, f["mesh"].attrs["ng"])

                _parse_hdf5_to_derived_type(f["setup"], instance.setup)

                st = pd.Timestamp(instance.setup.start_time)

                et = pd.Timestamp(instance.setup.end_time)

                instance.setup._ntime_step = (
                    et - st
                ).total_seconds() / instance.setup.dt

                instance.mesh = MeshDT(
                    instance.setup,
                    f["mesh"].attrs["nrow"],
                    f["mesh"].attrs["ncol"],
                    f["mesh"].attrs["ng"],
                )

                _parse_hdf5_to_derived_type(f["mesh"], instance.mesh)

                _build_mesh(instance.setup, instance.mesh)

                instance._input_data = Input_DataDT(instance.setup, instance.mesh)

                instance._parameters = ParametersDT(instance.setup, instance.mesh)

                instance._output = OutputDT(instance.setup, instance.mesh)

                for derived_type_key in [
                    "atmos_data",
                    "physio_data",
                    "obs_response",
                    "sim_response",
                    "opr_parameters",
                    "opr_initial_states",
                    "opr_final_states",
                ]:
                    _parse_hdf5_to_derived_type(
                        f[derived_type_key], getattr(instance, derived_type_key)
                    )

                return instance

            else:
                raise ReadHDF5MethodError(
                    f"Unable to read '{path}' with 'read_model' method. The file may not have been created with 'save_model' method."
                )

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
