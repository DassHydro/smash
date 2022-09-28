from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT, Prcp_IndiceDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT

from smash.core._build_model import (
    _build_setup,
    _build_mesh,
)

import os
import h5py
import numpy as np

import smash

__all__ = ["save_model", "read_model"]


def _parse_hdf5_to_derived_type(hdf5_ins, derived_type):

    for ds in hdf5_ins.keys():

        if isinstance(hdf5_ins[ds], h5py.Group):

            hdf5_ins_imd = hdf5_ins[ds]

            _parse_hdf5_to_derived_type(hdf5_ins_imd, getattr(derived_type, ds))

        else:

            setattr(derived_type, ds, hdf5_ins[ds][:])

    for attr in hdf5_ins.attrs.keys():

        setattr(derived_type, attr, hdf5_ins.attrs[attr])


def _parse_derived_type_to_hdf5(derived_type, hdf5_ins):

    for attr in dir(derived_type):

        if not attr.startswith("_") and not attr in ["from_handle", "copy"]:

            try:

                value = getattr(derived_type, attr)

                if isinstance(
                    value,
                    (
                        SetupDT,
                        MeshDT,
                        Input_DataDT,
                        Prcp_IndiceDT,
                        ParametersDT,
                        StatesDT,
                        OutputDT,
                    ),
                ):

                    hdf5_ins_imd = hdf5_ins.create_group(attr)

                    _parse_derived_type_to_hdf5(value, hdf5_ins_imd)

                elif isinstance(value, np.ndarray):

                    if value.dtype == "object" or value.dtype.char == "U":

                        value = value.astype("S")

                    hdf5_ins.create_dataset(
                        attr,
                        shape=value.shape,
                        dtype=value.dtype,
                        data=value,
                        compression="gzip",
                        chunks=True,
                    )

                elif isinstance(value, bytes):

                    value = value.strip()

                    hdf5_ins.attrs[attr] = value

                else:

                    hdf5_ins.attrs[attr] = value

            except:

                pass


def save_model(model: Model, path: str):

    """
    Save model
    """

    if not path.endswith(".hdf5"):

        path = path + ".hdf5"

    with h5py.File(path, "w") as f:

        for derived_type_key in [
            "setup",
            "mesh",
            "input_data",
            "parameters",
            "states",
            "output",
        ]:

            derived_type = getattr(model, derived_type_key)

            grp = f.create_group(derived_type_key)

            _parse_derived_type_to_hdf5(derived_type, grp)

            f.attrs["_last_update"] = model._last_update

        f.close()


def read_model(path: str) -> Model:

    """
    Read model
    """

    if os.path.isfile(path):

        with h5py.File(path, "r") as f:

            instance = smash.Model(None, None)

            if "descriptor_name" in f["setup"].keys():

                nd = f["setup"]["descriptor_name"].shape[1]

            else:

                nd = 0

            instance.setup = SetupDT(nd)

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

            for derived_type_key in ["input_data", "parameters", "states", "output"]:

                _parse_hdf5_to_derived_type(
                    f[derived_type_key], getattr(instance, derived_type_key)
                )

            instance._last_update = f.attrs["_last_update"]

            f.close()

        return instance

    else:

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
