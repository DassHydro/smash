from __future__ import annotations

from smash.core._constant import STRUCTURE_PARAMETERS, STRUCTURE_STATES

from smash.io._error import ReadHDF5MethodError

from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT

from smash.core._build_model import _build_mesh

from smash.tools import hdf5_handler
from smash.tools import object_handler

import os
import errno
import warnings
import h5py
import numpy as np
import pandas as pd
import smash

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

__all__ = ["save_smash_model_to_hdf5", "load_hdf5_file"]



def _generate_light_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    """
    this function create a light dictionnary containing the required data-structure to save a smash model object to an hdf5 file

    Parameters
    ----------
    structure : str
        the smash model structure used {gr-a, gr-b, gr-c, gr-d}
    structure_parameters: dict
        the dict containing the parameter to be saved for each model structure 
    structure_states: dict
        the dict containing the states to be saved for each model structure 

    Returns
    -------
    dict :
        A light dictionary matching the structure of the smash model object.
    """
    return {
        "setup": ["dt", "end_time", "start_time"],
        "mesh": ["active_cell", "area", "code", "dx", "ng", "ymax", "xmin", "nrow", "ncol", "gauge_pos", "flwacc"],
        "input_data": ["qobs"],
        "parameters": structure_parameters[
            structure
        ],  # only calibrated Model param will be stored
        "output": [
            {
                "fstates": structure_states[structure]
            },  # only final Model states will be stored
            "qsim",
        ],
    }



def _generate_medium_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    """
    this function create a medium dictionnary containing the required data-structure to save a smash model object to an hdf5 file

    Parameters
    ----------
    structure : str
        the smash model structure used {gr-a, gr-b, gr-c, gr-d}
    structure_parameters: dict
        the dict containing the parameter to be saved for each model structure 
    structure_states: dict
        the dict containing the states to be saved for each model structure 

    Returns
    -------
    dict :
        A medium dictionary matching the structure of the smash model object.
    """
    return {
        "setup": ["dt", "end_time", "start_time", "structure", "_ntime_step"],
        "mesh": ["active_cell", "area", "code", "dx", "flwdir", "nac", "ng", "path", "ymax", "xmin", "nrow", "ncol", "gauge_pos", "flwacc"],
        "input_data": ["mean_prcp", "mean_pet", "qobs"],
        "parameters": structure_parameters[
            structure
        ],  # only calibrated Model param will be stored
        "states": structure_states[
            structure
        ],  # only initial Model states will be stored
        "output": [
            {
                "fstates": structure_states[structure]
            },  # only final Model states will be stored
            "qsim",
            "cost",
            "cost_jobs",
            "cost_jreg"
        ],
    }


def _generate_full_smash_object_structure(instance):
    """
    this function create a full dictionnary containing all the structure of an smash model object in order to save it to an hdf5

    Parameters
    ----------
    instance : object
        a custom python object.
    
    Returns
    -------
    list :
        A list containing keys and dictionary matching the structure of the python object.
    """
    key_data=smash.tools.object_handler.generate_object_structure(instance)
    
    key_list=list()
    key_list.append(key_data)
    key_list.append("_last_update")
    
    return key_list



def generate_smash_object_structure(instance,typeofstructure="medium"):
    """
    this function create a dictionnary containing a complete ar partial structure of an object in order to save it to an hdf5. This functions is a conveninet way to generate the key_data as a dictionary. Then personnal keys can be added to the key_data dict.

    Parameters
    ----------
    instance : object
        a custom python object.
    typeofstructure : str
        the structure type : light, medium, full
    
    Returns
    -------
    dict :
        A list or dictionary matching the structure of the python object.
    """
    structure=instance.setup.structure
    
    if typeofstructure=="light":
        
        key_data=_generate_light_smash_object_structure(structure)
        
    elif typeofstructure=="medium":
        
        key_data=_generate_medium_smash_object_structure(structure)
        
    elif typeofstructure=="full":
        
        key_data=_generate_full_smash_object_structure(instance)
    
    return key_data



def save_smash_model_to_hdf5(path_to_hdf5, instance, keys_data=None, content="medium", location="./", sub_data=None, replace=True):
    """
    dump an object to an hdf5 file

    Parameters
    ----------
    path_to_hdf5 : str
        path to the hdf5 file
    instance : object
        python object
    keys_data : list | dict
        a list or a dictionary of the attribute to be saved
    content : str
        {light,medium,full}
    location : str
        path location or subgroup where to write data in the hdf5 file
    sub_data : dict | None
        a dictionary containing extra-data to be saved
    replace : Boolean
        replace an existing hdf5 file. Default is False
    
    Examples
    --------
    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)
    model.run(inplace=True)
    
    keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="medium")
    #add a new data to save:
    keys_data["parameters"].append('ci')
    
    #Save a single smash model
    smash.save_smash_model_to_hdf5("./model_light.hdf5", model, content="light", replace=True)
    smash.save_smash_model_to_hdf5("./model_medium.hdf5", model, content="medium", replace=True)
    smash.save_smash_model_to_hdf5("./model_full.hdf5", model, content="full", replace=True)
    smash.save_smash_model_to_hdf5("./model_user.hdf5", model, keys_data=keys_data, replace=True)

    #adding subdata
    sub_data={"sub_data1":"mydata"}
    sub_data.update({"sub_data2":2.5})
    sub_data.update({"sub_data3":{"sub_sub_data1":2.5,"sub_sub_data2":np.zeros(10)}})

    smash.save_smash_model_to_hdf5("./model_sub_data.hdf5", model, content="medium",sub_data=sub_data, replace=True)
    """
    if content == "light":
        
        keys_data=_generate_light_smash_object_structure(instance.setup.structure)
        
    elif content == "medium":
        
        keys_data=_generate_medium_smash_object_structure(instance.setup.structure)
        
    elif content == "full":
        
        keys_data=_generate_full_smash_object_structure(instance)
    
    if isinstance(keys_data,(dict,list)):
        
        smash.tools.hdf5_handler.save_object_to_hdf5(path_to_hdf5, instance, keys_data, location=location, sub_data=sub_data,replace=replace)
        
    else: 
        
        raise ValueError(
                    f"{keys_data} must be a instance of list or dict."
                )


def load_hdf5_file(f_hdf5,as_model=False):
    """
    Load an hdf5 file

    Parameters
    ----------
    f_hdf5 : str
        path to the hdf5 file
    as_model : Boolean
        load the hdf5 as a smash model. Default is False
    
    Return
    --------
    instance : an instance of the smash model or a dictionary
    
    Examples
    --------
    #load an hdf5 file to a dictionary
    dictionary=smash.load_hdf5_file("./multi_model.hdf5")
    dictionary["model1"].keys()
    dictionary["model1"]["mesh"].keys()
    
    #reload a full model object
    model_reloaded=smash.load_hdf5_file("./model_full.hdf5",as_model=True)
    model_reloaded
    model_reloaded.run()
    """
    if as_model:
        
        instance=read_hdf5_to_model_object(f_hdf5)
        return instance
        
    else:
        
        hdf5=smash.tools.hdf5_handler.open_hdf5(f_hdf5, read_only=True, replace=False)
        dictionary=smash.tools.hdf5_handler.read_hdf5_as_dict(hdf5)
        hdf5.close()
        return dictionary


def _parse_hdf5_to_derived_type(hdf5_ins, derived_type):
    for ds in hdf5_ins.keys():
        if isinstance(hdf5_ins[ds], h5py.Group):
            hdf5_ins_imd = hdf5_ins[ds]

            _parse_hdf5_to_derived_type(hdf5_ins_imd, getattr(derived_type, ds))

        else:
            setattr(derived_type, ds, hdf5_ins[ds][:])

    for attr in hdf5_ins.attrs.keys():
        
        # check if value is equal to "_None_" (None string because hdf5 does not supported)
        if hdf5_ins.attrs[attr] == "_None_":
            setattr(derived_type, attr, None)
        else:
            setattr(derived_type, attr, hdf5_ins.attrs[attr])


def read_hdf5_to_model_object(path: str) -> Model:
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
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization

    Save Model

    >>> smash.save_model(model, "model.hdf5")

    Read Model

    >>> model_rld = smash.read_model("model.hdf5")
    >>> model_rld
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization
    """

    if os.path.isfile(path):
        with h5py.File(path, "r") as f:
            
            if not f.attrs.__contains__('_last_update'):
                raise ValueError(
                    f'The hdf5 file {path} does not contain the full smash object structure and therefore cannot be loaded as a smash model object. The full structure of a smash model object can be saved using smash.save_smash_model_to_hdf5(filename, smash_model, content="full").'
                )
            
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

            instance.input_data = Input_DataDT(instance.setup, instance.mesh)

            instance.parameters = ParametersDT(instance.mesh)

            instance.states = StatesDT(instance.mesh)

            instance.output = OutputDT(instance.setup, instance.mesh)

            for derived_type_key in [
                "input_data",
                "parameters",
                "states",
                "output",
            ]:
                _parse_hdf5_to_derived_type(
                    f[derived_type_key], getattr(instance, derived_type_key)
                )

            instance._last_update = f.attrs["_last_update"]

            return instance


    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
