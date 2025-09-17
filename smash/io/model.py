from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

import h5py
import numpy as np
from f90wrap.runtime import FortranDerivedType, FortranDerivedTypeArray

import smash
from smash.fcore._mwd_input_data import Input_DataDT
from smash.fcore._mwd_mesh import MeshDT
from smash.fcore._mwd_output import OutputDT
from smash.fcore._mwd_parameters import ParametersDT
from smash.fcore._mwd_setup import SetupDT
from smash.io._error import ReadHDF5MethodError
from smash.io.handler._hdf5_handler import (
    _dump_model,
    _load_hdf5_dataset_to_npndarray,
    _load_hdf5_to_dict,
    _map_hdf5_to_fortran_derived_type,
    _map_hdf5_to_fortran_derived_type_array,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import FilePath

__all__ = ["read_model", "save_model"]


def save_model(model: Model, path: FilePath):
    """
    Save Model object to HDF5.

    Parameters
    ----------
    model : `Model <smash.Model>`
        The Model object to be saved to `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file.

    path : `str`
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file
        path.

    See Also
    --------
    read_model: Read Model object from HDF5.
    smash.Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_model, read_model
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Model
        atmos_data: ['mean_pet', 'mean_prcp', '...', 'sparse_prcp', 'sparse_snow']
        mesh: ['active_cell', 'area', '...', 'xres', 'ymax']
        ...
        setup: ['adjust_interception', 'compute_mean_atmos', '...', 'structure', 'temp_directory']
        u_response_data: ['q_stdev']
    >>> model.setup.hydrological_module, model.setup.routing_module
    ('gr4', 'lr')

    Save Model to HDF5

    >>> save_model(model, "model.hdf5")
    """
    if not path.endswith(".hdf5"):
        path += ".hdf5"

    with h5py.File(path, "w") as h5:
        _dump_model("model", model, h5)
        h5.attrs["_save_func"] = "save_model"


def read_model(path: FilePath) -> Model:
    """
    Read Model object from HDF5.


    Parameters
    ----------
    path : `str`
        The file path.

    Returns
    -------
    model : `Model <smash.Model>`
        A Model object loaded from HDF5.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_model`.

    See Also
    --------
    save_model: Save Model object to HDF5.
    smash.Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_model, read_model
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Model
        atmos_data: ['mean_pet', 'mean_prcp', '...', 'sparse_prcp', 'sparse_snow']
        mesh: ['active_cell', 'area', '...', 'xres', 'ymax']
        ...
        setup: ['adjust_interception', 'compute_mean_atmos', '...', 'structure', 'temp_directory']
        u_response_data: ['q_stdev']
    >>> model.setup.hydrological_module, model.setup.routing_module
    ('gr4', 'lr')

    Save Model to HDF5

    >>> save_model(model, "model.hdf5")

    Read Model from HDF5

    >>> model_rld = read_model("model.hdf5")
    >>> model_rld
    Model
        atmos_data: ['mean_pet', 'mean_prcp', '...', 'sparse_prcp', 'sparse_snow']
        mesh: ['active_cell', 'area', '...', 'xres', 'ymax']
        ...
        setup: ['adjust_interception', 'compute_mean_atmos', '...', 'structure', 'temp_directory']
        u_response_data: ['q_stdev']
    >>> model.setup.hydrological_module, model.setup.routing_module
    ('gr4', 'lr')
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    with h5py.File(path, "r") as h5:
        if h5.attrs.get("_save_func") == "save_model":
            h5m = h5["model"]
            model = smash.Model(None, None)

            model.setup = SetupDT(h5m["setup"].attrs["nd"])

            _map_hdf5_to_fortran_derived_type(h5m["setup"], model.setup)

            model.mesh = MeshDT(
                model.setup,
                h5m["mesh"].attrs["nrow"],
                h5m["mesh"].attrs["ncol"],
                h5m["mesh"].attrs["npar"],
                h5m["mesh"].attrs["ng"],
            )

            _map_hdf5_to_fortran_derived_type(h5m["mesh"], model.mesh)

            model._input_data = Input_DataDT(model.setup, model.mesh)

            model._parameters = ParametersDT(model.setup, model.mesh)

            model._output = OutputDT(model.setup, model.mesh)

            for attr in dir(model):
                if attr.startswith("_"):
                    continue
                try:
                    value = getattr(model, attr)
                except Exception:
                    pass

                if isinstance(value, FortranDerivedType):
                    _map_hdf5_to_fortran_derived_type(h5m[attr], value)

                elif isinstance(value, FortranDerivedTypeArray):
                    _map_hdf5_to_fortran_derived_type_array(h5m[attr], value)

                # % At the moment, there are only FortranDerivedType and
                # % FortranDerivedTypeArray as attributes of Model
                elif isinstance(value, dict):
                    value.update(_load_hdf5_to_dict(h5m[attr]))

                elif isinstance(value, list):
                    value = list(_load_hdf5_dataset_to_npndarray(h5m[attr]))

                elif isinstance(value, tuple):
                    value = tuple(_load_hdf5_dataset_to_npndarray(h5m[attr]))

                elif isinstance(value, np.ndarray):
                    value = _load_hdf5_dataset_to_npndarray(h5m[attr])

                elif isinstance(value, (str, int, float, np.number)):
                    value = h5m.attrs[attr]

        else:
            raise ReadHDF5MethodError(
                f"Unable to read '{path}' with 'read_model' method. The file may not have been created with "
                f"'save_model' method."
            )

    return model
