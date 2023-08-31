from __future__ import annotations

from smash.fcore._mwd_setup import SetupDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_output import OutputDT

import numpy as np
import h5py


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
                        ParametersDT,
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

                else:
                    hdf5_ins.attrs[attr] = value

            except:
                pass
