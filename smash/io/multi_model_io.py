from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

from smash.core._constant import STRUCTURE_PARAMETERS, STRUCTURE_STATES

from smash.io._error import ReadHDF5MethodError


from smash.solver._mwd_setup import SetupDT
from smash.solver._mwd_mesh import MeshDT
from smash.solver._mwd_input_data import Input_DataDT
from smash.solver._mwd_parameters import ParametersDT
from smash.solver._mwd_states import StatesDT
from smash.solver._mwd_output import OutputDT

from smash.core._build_model import _build_mesh



import os
import errno
import warnings
import h5py
import numpy as np
import pandas as pd
import smash



__all__ = ["open_hdf5", "add_hdf5_sub_group", "generate_light_smash_object_structure", "generate_medium_smash_object_structure", "generate_object_structure",  "generate_smash_object_structure", "dump_object_to_hdf5_from_list_attribute", "dump_object_to_hdf5_from_dict_attribute", "dump_object_to_hdf5_from_str_attribute", "dump_object_to_hdf5_from_iteratable", "dump_object_to_hdf5", "save_smash_model_to_hdf5", "load_hdf5_file", "read_hdf5_to_dict"]



def open_hdf5(path, replace=False):
    
    if not path.endswith(".hdf5"):
        
        path = path + ".hdf5"
    
    if replace==True:
        
        f= h5py.File(path, "w")
        
    else:
        
        f= h5py.File(path, "a")
        
    return f



def add_hdf5_sub_group(hdf5, subgroup=None):
    
    if subgroup is not None:
        
        loc_path=os.path.dirname(subgroup)
        
        if loc_path=="":
            
            loc_path="./"
            hdf5.require_group(subgroup)
    
    return hdf5


def generate_light_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    
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



def generate_medium_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    
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


def generate_full_smash_object_structure(instance):
    
    key_data=generate_object_structure(instance)
    
    key_list=list()
    key_list.append(key_data)
    key_list.append("_last_update")
    
    return key_list


def generate_object_structure(instance):
    
    key_data={}
    key_list=list()
    return_list=False
    
    for attr in dir(instance):
        
        if not attr.startswith("_") and not attr in ["from_handle", "copy"]:
            
            try:
                
                value = getattr(instance, attr)
                
                if isinstance(value, np.ndarray):
                    
                    if value.dtype == "object" or value.dtype.char == "U":
                        value = value.astype("S")
                        
                    #key_data.update({attr:value})
                    key_list.append(attr)
                    return_list=True
                    
                elif isinstance(value,(str,float,int)):
                    
                    #key_data.update({attr:value})
                    key_list.append(attr)
                    return_list=True
                    
                else: 
                    
                    depp_key_data=generate_object_structure(value)
                    
                    if (len(depp_key_data)>0):
                        key_data.update({attr:depp_key_data})
            
            except:
                
                pass
    
    if return_list:
        
        for attr, value in key_data.items():
            key_list.append({attr:value})
        
        return key_list
        
    else:
        
        return key_data



def generate_smash_object_structure(instance,typeofstructure="medium"):
    
    structure=instance.setup.structure
    
    if typeofstructure=="light":
        
        key_data=generate_light_smash_object_structure(structure)
        
    elif typeofstructure=="medium":
        
        key_data=generate_medium_smash_object_structure(structure)
        
    elif typeofstructure=="full":
        
        key_data=generate_full_smash_object_structure(instance)
    
    return key_data



def dump_object_to_hdf5_from_list_attribute(hdf5,instance,list_attr):
    
    if isinstance(list_attr,list):
    
        for attr in list_attr:
            
            if isinstance(attr, str):
                
                dump_object_to_hdf5_from_str_attribute(hdf5, instance, attr)
            
            elif isinstance(attr,list):
                
                dump_object_to_hdf5_from_list_attribute(hdf5, instance, attr)
            
            elif isinstance(attr,dict):
                
                dump_object_to_hdf5_from_dict_attribute(hdf5, instance, attr)
                
            else:
                
                raise ValueError(
                    f"unconsistant {attr} in {list_attr}. {attr} must be a an instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{list_attr} must be a instance of list."
                )


def dump_object_to_hdf5_from_dict_attribute(hdf5,instance,dict_attr):
    
    if isinstance(dict_attr,dict):
    
        for attr, value in dict_attr.items():
            
            hdf5=add_hdf5_sub_group(hdf5, subgroup=attr)
            
            try:
            
                sub_instance=getattr(instance, attr)
                
            except:
                
                sub_instance=instance
            
            if isinstance(value,dict):
            
                dump_object_to_hdf5_from_dict_attribute(hdf5[attr], sub_instance, value)
            
            if isinstance(value,list):
            
                dump_object_to_hdf5_from_list_attribute(hdf5[attr], sub_instance, value)
            
            elif isinstance(value,str):
                
                dump_object_to_hdf5_from_str_attribute(hdf5[attr], sub_instance, value)
            
            else :
                
                raise ValueError(
                    f"Bad type of '{attr}' in '{dict_attr}'. Dict({attr}) must be a instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{dict_attr} must be a instance of dict."
                )


def dump_object_to_hdf5_from_str_attribute(hdf5,instance,str_attr):
    
    if isinstance(str_attr, str):
        
        try:
            
            value = getattr(instance, str_attr)
            
            if isinstance(value, np.ndarray):
                
                if value.dtype == "object" or value.dtype.char == "U":
                    value = value.astype("S")
                
                hdf5.create_dataset(
                    str_attr,
                    shape=value.shape,
                    dtype=value.dtype,
                    data=value,
                    compression="gzip",
                    chunks=True,
                )
                
            else:
                
                hdf5.attrs[str_attr] = value
                
        except:
            
            raise ValueError(
                f"Unable to get attribute {str_attr} in {instance}"
            )
    
    else:
        
        raise ValueError(
                    f"{str_attr} must be a instance of str."
                )


def dump_object_to_hdf5_from_iteratable(hdf5, instance, iteratable):
    
    if isinstance(iteratable,list):
        
        dump_object_to_hdf5_from_list_attribute(hdf5,instance,iteratable)
        
    elif isinstance(iteratable,dict):
        
        dump_object_to_hdf5_from_dict_attribute(hdf5,instance,iteratable)
    
    else :
        
        raise ValueError(
                    f"{iteratable} must be a instance of list or dict."
                )


def dump_object_to_hdf5(f_hdf5, instance, keys_data, location="./", replace=False):
    
    hdf5=open_hdf5(f_hdf5, replace=replace)
    hdf5=add_hdf5_sub_group(hdf5, subgroup=location)
    dump_object_to_hdf5_from_iteratable(hdf5[location], instance, keys_data)
    hdf5.close()


def save_smash_model_to_hdf5(path_to_hdf5, instance, keys_data=None, content="medium", location="./", replace=True):
    
    if content == "light":
        
        keys_data=generate_light_smash_object_structure(instance.setup.structure)
        
    elif content == "medium":
        
        keys_data=generate_medium_smash_object_structure(instance.setup.structure)
        
    elif content == "full":
        
        keys_data=generate_full_smash_object_structure(instance)
    
    if isinstance(keys_data,(dict,list)):
        
        dump_object_to_hdf5(path_to_hdf5, instance, keys_data, location=location, replace=replace)
        
    else: 
        
        raise ValueError(
                    f"{keys_data} must be a instance of list or dict."
                )
    


def load_hdf5_file(f_hdf5,as_model=False):
    
    if as_model:
        
        instance=read_hdf5_to_model_object(f_hdf5)
        return instance
        
    else:
        
        hdf5=open_hdf5(f_hdf5, replace=False)
        dictionary=read_hdf5_to_dict(hdf5)
        hdf5.close()
        return dictionary


def read_hdf5_to_dict(hdf5):
    
    dictionary={}
    
    for key,item in hdf5.items():
        
        if str(type(item)).find("group") != -1:
            
            dictionary.update({key:read_hdf5_to_dict(item)})
            
            list_attr=list(item.attrs.keys())
            
            for key_attr in list_attr:
                
                dictionary[key].update({key_attr:item.attrs[key_attr]})
            
        if str(type(item)).find("dataset") != -1:
            
            values = [
                        item[:].astype("U") if item[:].dtype.char == "S" else item[:]
                    ]
            dictionary.update({key:values})
            
            list_attr=list(item.attrs.keys())
            
            for key_attr in list_attr:
                
                dictionary.update({key_attr:item.attrs[key_attr]})
    
    
    list_attr=list(hdf5.attrs.keys())
    
    for key_attr in list_attr:
        
        dictionary.update({key_attr:hdf5.attrs[key_attr]})
                
    return dictionary



def _parse_hdf5_to_derived_type(hdf5_ins, derived_type):
    for ds in hdf5_ins.keys():
        if isinstance(hdf5_ins[ds], h5py.Group):
            hdf5_ins_imd = hdf5_ins[ds]

            _parse_hdf5_to_derived_type(hdf5_ins_imd, getattr(derived_type, ds))

        else:
            setattr(derived_type, ds, hdf5_ins[ds][:])

    for attr in hdf5_ins.attrs.keys():
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
