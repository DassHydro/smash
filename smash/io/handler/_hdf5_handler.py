from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
from f90wrap.runtime import FortranDerivedType, FortranDerivedTypeArray

from smash.fcore._mwd_sparse_matrix import Sparse_MatrixDT

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model
    from smash.util._typing import AlphaNumeric


def _dump_alphanumeric(name: str, value: AlphaNumeric, h5: h5py.File | h5py.Group):
    h5.attrs[name] = value


def _dump_npndarray(name: str, value: np.ndarray, h5: h5py.File | h5py.Group):
    if value.dtype == "object" or value.dtype.char == "U":
        # % zero-terminated bytes
        value = value.astype("S")

    h5.create_dataset(
        name,
        shape=value.shape,
        dtype=value.dtype,
        data=value,
        compression="gzip",
        chunks=True,
    )


def _dump_dict(name: str, dct: dict[str, Any], h5: h5py.File | h5py.Group):
    h5g = h5.create_group(name)

    for key, value in dct.items():
        # % Recursive call
        if isinstance(value, dict):
            _dump_dict(key, value, h5g)

        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.array(value)
            _dump_npndarray(key, value, h5g)

        elif isinstance(value, (str, int, float, np.number)):
            _dump_alphanumeric(key, value, h5g)

        # % Should be unreachable
        else:
            pass


def _dump_object(name: str, obj: object, h5: h5py.File | h5py.Group):
    h5g = h5.create_group(name)

    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if callable(value):
            continue

        elif isinstance(value, FortranDerivedType):
            _dump_fortran_derived_type(attr, value, h5g)

        elif isinstance(value, FortranDerivedTypeArray):
            _dump_fortran_derived_type_array(attr, value, h5g)

        elif isinstance(value, dict):
            _dump_dict(attr, value, h5g)

        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.array(value)
            _dump_npndarray(attr, value, h5g)

        elif isinstance(value, (str, int, float, np.number)):
            _dump_alphanumeric(attr, value, h5g)

        # % Should be unreachable
        else:
            pass


# % Same as object atm
def _dump_model(name: str, model: Model, h5: h5py.File | h5py.Group):
    _dump_object(name, model, h5)


# % Same as object atm
def _dump_fortran_derived_type(name: str, fdt: FortranDerivedType, h5: h5py.File | h5py.Group):
    _dump_object(name, fdt, h5)


# % Handle by hand each derived type array (maybe there is a way to be generic ...)
def _dump_fortran_derived_type_array(name: str, fdta: FortranDerivedTypeArray, h5: h5py.File | h5py.Group):
    if len(fdta) == 0:
        return

    # % Sparse_MatrixDT
    if isinstance(fdta[0], Sparse_MatrixDT):
        nc = np.amax([fdt.values.shape for fdt in fdta.items()])
        ntime_step = len(fdta)
        attr_arr = {
            "n": np.zeros(shape=ntime_step, dtype=np.int32),
            "coo_fmt": np.zeros(shape=ntime_step, dtype=np.int32),
            "zvalue": np.zeros(shape=ntime_step, dtype=np.float32),
            "indices": np.zeros(shape=(nc, ntime_step), dtype=np.int32),
            "values": np.zeros(shape=(nc, ntime_step), dtype=np.float32),
        }

        for i, fdt in enumerate(fdta.items()):
            for attr in ["n", "coo_fmt", "zvalue"]:
                attr_arr[attr][i] = getattr(fdt, attr)

            for attr in ["indices", "values"]:
                try:
                    n = attr_arr["n"][i]
                    attr_arr[attr][0:n, i] = getattr(fdt, attr)
                except Exception:
                    pass

        _dump_dict(name, attr_arr, h5)


def _load_hdf5_dataset_to_npndarray(ds: h5py.Dataset) -> np.ndarray:
    value = ds[:]

    if value.dtype.char == "S":
        value = value.astype("U")

    return value


def _load_hdf5_to_dict(h5: h5py.File | h5py.Group) -> dict[str, Any]:
    dct = {}

    for key, value in h5.items():
        if isinstance(value, h5py.Group):
            dct[key] = _load_hdf5_to_dict(value)
        elif isinstance(value, h5py.Dataset):
            dct[key] = _load_hdf5_dataset_to_npndarray(value)

    dct.update(dict(h5.attrs.items()))

    return dct


def _map_hdf5_to_fortran_derived_type(h5: h5py.File | h5py.Group, fdt: FortranDerivedType):
    for key, value in h5.items():
        if isinstance(value, h5py.Group):
            sub_fdt = getattr(fdt, key)

            if isinstance(sub_fdt, FortranDerivedType):
                _map_hdf5_to_fortran_derived_type(value, sub_fdt)
            elif isinstance(sub_fdt, FortranDerivedTypeArray):
                _map_hdf5_to_fortran_derived_type_array(value, sub_fdt)
            # % Should be unreachable
            else:
                pass

        elif isinstance(value, h5py.Dataset):
            setattr(fdt, key, _load_hdf5_dataset_to_npndarray(value))

    for key, value in h5.attrs.items():
        setattr(fdt, key, value)


# % Must handle each case of derived type array by hand
def _map_hdf5_to_fortran_derived_type_array(h5: h5py.File | h5py.Group, fdta: FortranDerivedTypeArray):
    if len(fdta) == 0:
        return

    # % Sparse Matrix case
    if isinstance(fdta[0], Sparse_MatrixDT):
        for i, fdt in enumerate(fdta.items()):
            fdt.alloc(
                h5["n"][i],
                h5["coo_fmt"][i],
                h5["zvalue"][i],
            )
            for attr in ["indices", "values"]:
                try:
                    n = fdt.n
                    setattr(fdt, attr, h5[attr][0:n, i])
                except Exception:
                    pass

    # % Should be unreachable
    else:
        pass
