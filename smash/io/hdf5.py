from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT

from smash.core._build_model import (
    _build_setup,
    _build_mesh,
)

import h5py
import numpy as np

import smash

__all__ = ["save_mesh", "read_mesh", "save_model", "read_model"]


def _parse_hdf5_to_derived_type(data, derived_type):

    for ds in data.keys():

        setattr(derived_type, ds, data[ds][:])

    for attr in data.attrs.keys():

        setattr(derived_type, attr, data.attrs[attr])


def save_mesh(mesh: dict, path: str):

    """
    Save mesh
    """

    with h5py.File(path, "w") as f:

        for key, value in mesh.items():

            if isinstance(value, np.ndarray):

                f.create_dataset(
                    key,
                    shape=value.shape,
                    dtype=value.dtype,
                    data=value,
                    compression="gzip",
                    chunks=True,
                )

            elif isinstance(value, bytes):
                value = value.strip()

                f.attrs[key] = value

            else:

                f.attrs[key] = value


def read_mesh(path: str) -> dict:

    """
    Read mesh
    """

    mesh = {}

    with h5py.File(path, "r") as f:

        ac = "active_cell" in list(f.keys())

        for attr in list(f.attrs.keys()):

            mesh[attr] = f.attrs[attr]

        for ds in list(f.keys()):

            if ds in ["flow", "drained_area"]:

                if ac:

                    mesh[ds] = np.ma.masked_array(
                        f[ds][:], mask=(1 - f["active_cell"][:])
                    )

                else:

                    mesh[ds] = f[ds][:]

            else:

                mesh[ds] = f[ds][:]

    return mesh


def save_model(instance: Model, path: str):

    """
    Save model
    """

    if not path.endswith(".hdf5"):

        path = path + ".hdf5"

    with h5py.File(path, "w") as f:

        for ins_attr in [
            "setup",
            "mesh",
            "input_data",
            "parameters",
            "states",
            "output",
        ]:

            grp = f.create_group(ins_attr)

            ins_getattr = getattr(instance, ins_attr)

            for attr in dir(ins_getattr):

                if not attr.startswith("_") and not attr in ["from_handle", "copy"]:

                    try:

                        value = getattr(ins_getattr, attr)

                        if isinstance(value, np.ndarray):

                            if value.dtype == "object" or value.dtype.char == "U":

                                value = value.astype("S")

                            grp.create_dataset(
                                attr,
                                shape=value.shape,
                                dtype=value.dtype,
                                data=value,
                                compression="gzip",
                                chunks=True,
                            )

                        else:

                            if isinstance(value, bytes):

                                value = value.strip()

                            grp.attrs[attr] = value

                    except:

                        pass


def read_model(path: str) -> Model:

    """
    Read model
    """

    with h5py.File(path, "r") as f:

        instance = smash.Model(None, None)

        instance.setup = SetupDT()

        _parse_hdf5_to_derived_type(f["setup"], instance.setup)

        _build_setup(instance.setup)

        instance.mesh = MeshDT(
            instance.setup,
            f["mesh"].attrs["nrow"],
            f["mesh"].attrs["ncol"],
            f["mesh"].attrs["ng"],
        )

        _parse_hdf5_to_derived_type(f["mesh"], instance.mesh)

        _build_mesh(instance.setup, instance.mesh)

        instance.input_data = Input_DataDT(instance.setup, instance.mesh)

        instance.parameters = ParametersDT(instance.mesh)

        instance.states = StatesDT(instance.mesh)

        instance.output = OutputDT(instance.setup, instance.mesh)

        for ins_attr in ["input_data", "parameters", "states", "output"]:

            _parse_hdf5_to_derived_type(f[ins_attr], getattr(instance, ins_attr))

        return instance
