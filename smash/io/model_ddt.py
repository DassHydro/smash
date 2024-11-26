from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

import h5py

from smash._constant import MODEL_DDT_IO_ATTR_KEYS
from smash.io._error import ReadHDF5MethodError
from smash.io.handler._hdf5_handler import _dump_dict, _load_hdf5_to_dict

if TYPE_CHECKING:
    from typing import Any

    from smash.core.model.model import Model
    from smash.util._typing import FilePath

__all__ = ["read_model_ddt", "save_model_ddt"]


def save_model_ddt(model: Model, path: FilePath):
    """
    Save some derived data types of the Model object to HDF5.

    This method is considerably lighter than `smash.io.save_model <save_model>` method that saves
    the entire Model object. Note that there is no method to reconstruct the Model object with
    the saved data file from this method, as some information is left out to ensure a lighter memory backup.
    To read the saved data, use `smash.io.read_model_ddt <read_model_ddt>`,
    which returns a dictionary reflecting the structure of the Model object.

    The following data are stored into the `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file:

    - ``snow_module``, ``hydrological_module``, ``routing_module``, ``serr_mu_mapping``,
      ``serr_sigma_mapping``, ``start_time``, ``end_time``, ``dt``, ``descriptor_name`` from
      `Model.setup <smash.Model.setup>`
    - ``xres``, ``yres``, ``xmin``, ``ymax``, ``dx``, ``dy``, ``active_cell``, ``gauge_pos``, ``code``,
      ``area`` from `Model.mesh <smash.Model.mesh>`
    - ``q`` from `Model.response_data <smash.Model.response_data>`
    - ``descriptor`` from `Model.physio_data <smash.Model.physio_data>`
    - ``mean_prcp``, ``mean_pet``, ``mean_snow``, ``mean_temp`` from
      `Model.atmos_data <smash.Model.atmos_data>` (``mean_snow`` and ``mean_temp``
      are only stored if a snow module has been selected)
    - ``keys``, ``values`` from `Model.rr_parameters <smash.Model.rr_parameters>`
    - ``keys``, ``values`` from `Model.rr_initial_states <smash.Model.rr_initial_states>`
    - ``weight_1``, ``bias_1``, ``weight_2``, ``bias_2``, ``weight_3``, ``bias_3`` from
      `Model.nn_parameters <smash.Model.nn_parameters>` (depending on the hydrological module)
    - ``keys``, ``values`` from `Model.serr_mu_parameters <smash.Model.serr_mu_parameters>`
    - ``keys``, ``values`` from `Model.serr_sigma_parameters <smash.Model.serr_sigma_parameters>`
    - ``q`` from `Model.response <smash.Model.response>`
    - ``keys``, ``values`` from `Model.rr_final_states <smash.Model.rr_final_states>`

    Parameters
    ----------
    model : `Model <smash.Model>`
        The Model object to save derived data types as a HDF5 file.

    path : `str`
        The file path.

    See Also
    --------
    read_model_ddt: Read derived data types of the Model object from HDF5.
    smash.Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_model_ddt
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save some derived data types of the Model object to HDF5

    >>> save_model_ddt(model, "model_ddt.hdf5")
    """

    if not path.endswith(".hdf5"):
        path += ".hdf5"

    model_ddt = {}
    with h5py.File(path, "w") as h5:
        for attr, keys in MODEL_DDT_IO_ATTR_KEYS.items():
            try:
                model_ddt[attr] = {k: getattr(getattr(model, attr), k) for k in keys}
            except Exception:
                continue
        _dump_dict("model_ddt", model_ddt, h5)
        h5.attrs["_save_func"] = "save_model_ddt"


def read_model_ddt(path: FilePath) -> dict[str, dict[str, Any]]:
    """
    Read some derived data types of the Model object from HDF5.

    This method does not reconstruct the Model object because certain information has not been saved from the
    `smash.io.save_model_ddt <save_model_ddt>` method
    in order to have light memory backup. This method returns a dictionary whose organisation is similar to
    the Model object.

    Parameters
    ----------
    path : `str`
        The file path.

    Returns
    -------
    model_ddt : `dict[str, dict[str, Any]]`
        A dictionary with derived data types loaded from HDF5.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_model_ddt`.

    See Also
    --------
    save_model_ddt: Read derived data types of the Model object from HDF5.
    smash.Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_model_ddt, read_model_ddt
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save some derived data types of the Model object to HDF5

    >>> save_model_ddt(model, "model_ddt.hdf5")

    Read some derived data types of the Model object from HDF5

    >>> model_ddt = read_model_ddt("model_ddt.hdf5")
    >>> model_ddt.keys()
    dict_keys(['atmos_data', 'mesh', 'nn_parameters', 'physio_data', 'response',
    'response_data', 'rr_final_states', 'rr_initial_states', 'rr_parameters',
    'serr_mu_parameters', 'serr_sigma_parameters', 'setup'])

    Access to setup variables

    >>> model_ddt["setup"]
    {'descriptor_name': array(['slope', 'dd'], dtype='<U5'), 'dt': 3600.0,
    'end_time': '2014-11-14 00:00', 'hydrological_module': 'gr4', 'routing_module': 'lr',
    'serr_mu_mapping': 'Zero', 'serr_sigma_mapping': 'Linear', 'snow_module': 'zero',
    'start_time': '2014-09-15 00:00'}

    Access to rainfall-runoff parameter keys

    >>> model_ddt["rr_parameters"]["keys"]
    array(['ci', 'cp', 'ct', 'kexc', 'llr'], dtype='<U4')
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    with h5py.File(path, "r") as h5:
        if h5.attrs.get("_save_func") == "save_model_ddt":
            model_ddt = _load_hdf5_to_dict(h5["model_ddt"])

        else:
            raise ReadHDF5MethodError(
                f"Unable to read '{path}' with 'read_model_ddt' method. The file may not have been created "
                f"with 'save_model_ddt' method."
            )

    return model_ddt
