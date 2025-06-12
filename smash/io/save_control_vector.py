from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import numpy as np

from smash.io.handler._hdf5_handler import _dump_alphanumeric, _dump_dict, _dump_npndarray, _load_hdf5_to_dict

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.core.simulation.optimize import BayesianOptimize, Optimize
    from smash.util._typing import FilePath


def save_control_vector(
    model: Model,
    return_options: Optimize | BayesianOptimize,
    mapping: str,
    path: FilePath,
    optimize_options: dict | None = None,
):
    """
    Save the control vector after optimization.

    Parameters
    ----------
    model: `Model <smash.Model>`
        A SMASH model object
    return_options: `Class <smash..core.simulation.optimize.Optimize>`
        A returned object after optimization. This object must contain the attributes `control_vector`.
    mapping: str
        The mapping used for the optimization.
    path: str
        Hdf5 file path to save the control vector.
    optimize_options: dict
        Dictionary of the optimize_options used to optimize the model.

    Examples
    --------
    >>>import smash
    >>>setup, mesh = smash.factory.load_dataset("Lez")
    >>>model = smash.Model(setup, mesh)
    >>>optimize_options = {"termination_crit": {"maxiter": 2}}
    >>>model_first_guess = smash.optimize(model,
    >>>                                   optimize_options=optimize_options)
    >>>return_options={"control_vector": True}
    >>>model_ml, ret_optim_ml = smash.optimize(model_first_guess,
    >>>                                        mapping="multi-linear",
    >>>                                        optimize_options=optimize_options,
    >>>                                        return_options=return_options)
    >>>smash.io.save_control_vector(model=model_ml,
    >>>                          return_options=ret_optim_ml,
    >>>                          mapping="multi-linear",
    >>>                          path="mycontrol_ml.hdf5",
    >>>                          optimize_options=optimize_options)
    """
    if not hasattr(return_options, "control_vector"):
        raise ValueError(
            "ret_option must have the attribute 'control_vector'."
            "Set option return_options={'control_vector': True} in smash.optimize function."
        )

    if optimize_options is None:
        optimize_options = {}

    saved_control = {
        "l_descriptor": model.physio_data.l_descriptor,
        "u_descriptor": model.physio_data.u_descriptor,
        "control_vector": return_options.control_vector,
        "optimize_options": optimize_options,
        "mapping": mapping,
        "structure": model.setup.structure,
    }

    if not path.endswith(".hdf5"):
        path += ".hdf5"

    with h5py.File(path, "w") as h5:
        for key, item in saved_control.items():
            if isinstance(item, np.ndarray):
                _dump_npndarray(key, item, h5)
            elif isinstance(item, dict):
                _dump_dict(key, item, h5)
            elif isinstance(item, (str, int, float, np.number)):
                _dump_alphanumeric(key, item, h5)


def import_control_vector(model: Model, path: FilePath):
    """
    Import a control vector saved with the function `smash.io.save_control_vector`.

    Parameters
    ----------
    model: `Model <smash.Model>`
        A SMASH model object
    path: str
        Hdf5 file path to save the control vector.

    Examples
    --------
    >>>import smash
    >>>setup, mesh = smash.factory.load_dataset("Lez")
    >>>model = smash.Model(setup, mesh)
    >>>smash.io.import_control_vector(model,
    >>>                               "mycontrol_ml.hdf5")
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist.")

    with h5py.File(path, "r") as h5:
        l_descriptor = h5["l_descriptor"][:]
        u_descriptor = h5["u_descriptor"][:]
        control_vector = h5["control_vector"][:]
        mapping = h5.attrs["mapping"]
        structure = h5.attrs["structure"]
        optimize_options = _load_hdf5_to_dict(h5["optimize_options"])

    if model.setup.structure != structure:
        raise ValueError(
            "Cannot import the control vector in this SMASH model."
            f"The structure of the SMASH model `{model.setup.structure}` "
            "differ from the one used to generate the control vector `{structure}`"
        )

    model.physio_data.l_descriptor = l_descriptor
    model.physio_data.u_descriptor = u_descriptor

    model.set_control_optimize(
        control_vector,
        mapping=mapping,
        optimize_options=optimize_options,
    )
