from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

from smash.core._constant import STRUCTURE_PARAMETERS, STRUCTURE_STATES

from smash.io._error import ReadHDF5MethodError

import os
import errno
import warnings
import h5py
import numpy as np

__all__ = ["save_multi_model", "read_multi_model"]


def _default_save_data(structure: str):
    return {
        "setup": ["dt", "end_time", "start_time", "structure"],
        "mesh": ["active_cell", "area", "code", "dx", "flwdir"],
        "input_data": ["mean_prcp", "mean_pet", "qobs"],
        "parameters": STRUCTURE_PARAMETERS[
            structure
        ],  # only calibrated Model param will be stored
        "states": STRUCTURE_STATES[
            structure
        ],  # only initial Model states will be stored
        "output": [
            {
                "fstates": STRUCTURE_STATES[structure]
            },  # only final Model states will be stored
            "qsim",
            "lcurve",
        ],
    }


def _parse_selected_derived_type_to_hdf5(
    derived_type, list_attr, hdf5_ins, attr_suffix=""
):
    # TODO: clean function for attr_suffix

    for attr in list_attr:
        if isinstance(attr, str):
            try:
                value = getattr(derived_type, attr)

                attr += attr_suffix

                if isinstance(value, np.ndarray):
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

        elif isinstance(attr, dict):
            for derived_type_key, list_attr_imd in attr.items():
                try:
                    derived_type_imd = getattr(derived_type, derived_type_key)

                    _parse_selected_derived_type_to_hdf5(
                        derived_type_imd, list_attr_imd, hdf5_ins
                    )

                except:
                    pass


def save_multi_model(model: Model, path: str, group=None, sub_data=None, sub_only=False, replace=False):
    """
    Save some derived data types of the Model object.

    This method is considerably lighter than `smash.save_model` method that saves the entire Model object.
    However, it is not capable of reconstructing the Model object from the saved data file.

    By default, the following data are stored into the `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file:

    - ``dt``, ``end_time``, ``start_time``, ``structure`` from `Model.setup`
    - ``active_cell``, ``area``, ``code``, ``dx``, ``flwdir`` from `Model.mesh`
    - ``mean_prcp``, ``mean_pet``, ``qobs`` from `Model.input_data`
    - ``qsim`` from `Model.output`
    - The final Model states (depending upon the Model structure) from state derived type of `Model.output`
    - The initial Model states (depending upon the Model structure) from `Model.states`
    - The Model parameters (depending upon the Model structure) from `Model.parameters`

    Subsidiary data can be added by filling in ``sub_data``.

    Parameters
    ----------
    model : Model
        The Model object to save derived data types as a HDF5 file.

    path : str
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file path.
    
    group : str
        subgroup name to group data in the hdf5 file.
        
        .. note::
            If not given, no subgroub is created and data are stored at the roots.

    sub_data : dict or None, default None
        Dictionary which indicates the subsidiary data to store into the HDF5 file.

        .. note::
            If not given, no subsidiary data is saved

    sub_only : bool, default False
        Allow to only store subsidiary data.

    replace : bool, default False
        replace or not an existing hdf5 file

    See Also
    --------
    read_model_ddt: Read derived data types of the Model object from HDF5 file.
    Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization

    Save spatially distributed precipitation in addition to default derived data types of Model

    >>> smash.save_model_ddt(model, "model_ddt.hdf5", sub_data={"prcp": model.input_data.prcp})

    """

    if not path.endswith(".hdf5"):
        path = path + ".hdf5"
    
    if replace==True:
        f= h5py.File(path, "w")
    else:
        f= h5py.File(path, "a")
    
    if group is not None:
        groupe_name=os.path.basename(group)
        groupe_path=os.path.dirname(group)
        grp=f.create_group(group)
    else:
        grp=f
    
    if not sub_only:
        save_data = _default_save_data(model.setup.structure)

        for derived_type_key, list_attr in save_data.items():
            derived_type = getattr(model, derived_type_key)

            if derived_type_key == "states":
                _parse_selected_derived_type_to_hdf5(
                    derived_type, list_attr, grp, attr_suffix="_0"
                )

            else:
                _parse_selected_derived_type_to_hdf5(derived_type, list_attr, grp)

    if sub_data is not None:
        for attr, value in sub_data.items():
            if (attr in grp) or (attr in grp.attrs):
                warnings.warn(f"Ignore updating existing key ({attr})")

                continue

            if isinstance(value, np.ndarray):
                if value.dtype == "object" or value.dtype.char == "U":
                    value = value.astype("S")

                try:
                    grp.create_dataset(
                        attr,
                        shape=value.shape,
                        dtype=value.dtype,
                        data=value,
                        compression="gzip",
                        chunks=True,
                    )
                except:
                    warnings.warn(f"Can not store to HDF5: {attr}")
            
            else:
                try:
                    grp.attrs[attr] = value

                except:
                    warnings.warn(f"Can not store to HDF5: {attr}")

    if group is not None:
        grp.attrs["_save_func"] = "save_model_ddt"
        f[groupe_path].attrs["_save_func"] = "save_multi_model"
    else:
        grp.attrs["_save_func"] = "save_model_ddt"
    


# ~ elif isinstance(value, dict):

# ~ for subkey,subvalue in value.items():
    
    # ~ if isinstance(subvalue, np.ndarray):
        
        # ~ if subvalue.dtype == "object" or subvalue.dtype.char == "U":
            # ~ subvalue = subvalue.astype("S")
        
        # ~ try:
            # ~ grp.create_dataset(
                # ~ subkey,
                # ~ shape=subvalue.shape,
                # ~ dtype=subvalue.dtype,
                # ~ data=subvalue,
                # ~ compression="gzip",
                # ~ chunks=True,
            # ~ )
        # ~ except:
            # ~ warnings.warn(f"Can not store to HDF5: {subkey}")
    
    # ~ else:
        
        # ~ try:
            # ~ grp.attrs[subkey] = subvalue
        # ~ except:
            # ~ warnings.warn(f"Can not store to HDF5: {subkey}")



def read_multi_model(path: str, group=None) -> dict:
    """
    Read derived data types of the Model object from HDF5 file.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    data : dict
        A dictionary with derived data types loaded from HDF5 file.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_model_ddt`.

    See Also
    --------
    save_model_ddt: Save some derived data types of the Model object.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> smash.save_model_ddt(model, "model_ddt.hdf5")

    Read the derived data types from HDF5 file

    >>> data = smash.read_multi_model("model_ddt.hdf5")

    Then, to see the dataset keys

    >>> data.keys()
    dict_keys(['active_cell', 'area', 'cft', 'code', 'cp', 'exc', 'flwdir',
    'hft', 'hft_0', 'hlr', 'hlr_0', 'hp', 'hp_0', 'lr', 'mean_pet', 'mean_prcp',
    'qobs', 'qsim', 'dt', 'dx', 'end_time', 'start_time', 'structure'])

    And finally, to access to derived data

    >>> data["mean_prcp"]
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

    """

    if os.path.isfile(path):
        with h5py.File(path) as f:
            
            #recursive function to convert hdf5 to dict ?
            #res={}
            #res=read_hdf5(f,res)
            
            if group is not None:
                
                if group in list(f.keys()):
                    grp=f[group]
                else:
                    raise ReadHDF5MethodError(
                        f"Unable to acces to group '{group}' in hdf5 '{path}', '{group}' group does not exist."
                    )
            else:
                grp=f
            
            if grp.attrs.get("_save_func") == "save_multi_model":
                group_dict={}
                
                for name,group in grp.items():
                    
                    keys = list(group.keys())

                    values = [
                        group[key][:].astype("U") if group[key][:].dtype.char == "S" else group[key][:]
                        for key in keys
                    ]

                    attr_keys = list(group.attrs.keys())

                    attr_keys.remove("_save_func")

                    attr_values = [group.attrs[key] for key in attr_keys]

                    group_dict.update({name:dict(zip(keys + attr_keys, values + attr_values))}) 
                
                return group_dict
            
            elif grp.attrs.get("_save_func") == "save_model_ddt":
                keys = list(grp.keys())

                values = [
                    grp[key][:].astype("U") if grp[key][:].dtype.char == "S" else grp[key][:]
                    for key in keys
                ]

                attr_keys = list(grp.attrs.keys())

                attr_keys.remove("_save_func")

                attr_values = [grp.attrs[key] for key in attr_keys]

                return dict(zip(keys + attr_keys, values + attr_values))

            else:
                raise ReadHDF5MethodError(
                    f"Unable to read '{path}' with 'read_model_group' method. The file may not have been created with 'read_model_group' method."
                )

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)



def read_hdf5(h,res):
    
    for name,group in h.items():
        
        if group.attrs.get("_save_func") == "save_model_ddt":
            keys = list(group.keys())
            
            values = [
                group[key][:].astype("U") if group[key][:].dtype.char == "S" else group[key][:]
                for key in keys
            ]
            
            attr_keys = list(group.attrs.keys())
            
            attr_keys.remove("_save_func")
            
            attr_values = [group.attrs[key] for key in attr_keys]
            
            res.update({name:dict(zip(keys + attr_keys, values + attr_values))})
            
            return res
        else:
            res.update({name:{}})
            
            read_hdf5(group,res)
