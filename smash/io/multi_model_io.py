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

__all__ = ["open_hdf5", "add_hdf5_sub_group", "default_model_data", "light_model_data", "dump_object_to_hdf5_from_list_attribute", "dump_object_to_hdf5_from_dict_attribute", "dump_object_to_hdf5_from_str_attribute", "dump_object_to_hdf5_from_iteratable", "dump_object_to_hdf5", "save_smash_model_to_hdf5", "load_hdf5_file", "read_hdf5_to_dict"]



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



def default_model_data(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    
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




def light_model_data(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
    
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
    
        for key, attr in dict_attr.items():
            
            hdf5=add_hdf5_sub_group(hdf5, subgroup=key)
            
            try:
            
                sub_instance=getattr(instance, key)
                
            except:
                
                sub_instance=instance
            
            if isinstance(attr,dict):
            
                dump_object_to_hdf5_from_dict_attribute(hdf5[key], sub_instance, attr)
            
            if isinstance(attr,list):
            
                dump_object_to_hdf5_from_list_attribute(hdf5[key], sub_instance, attr)
            
            elif isinstance(attr,str):
                
                dump_object_to_hdf5_from_str_attribute(hdf5[key], sub_instance, attr)
            
            else :
                
                raise ValueError(
                    f"unconsistant {attr} in {dict_attr}. {attr} must be a instance of dict, list or str"
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


def save_smash_model_to_hdf5(path_to_hdf5, instance, keys_data="default", location="./", replace=True):
    
    if isinstance(keys_data,str):
        
        if keys_data == "default":
            
            keys_data=default_model_data(instance.setup.structure)
            
        elif keys_data == "full":
            
            #to do
            keys_data=default_model_data(instance.setup.structure)
            
        elif keys_data == "light":
            
            keys_data=light_model_data(instance.setup.structure)
    
    if isinstance(keys_data,dict):
        
        dump_object_to_hdf5(path_to_hdf5, instance, keys_data, location=location, replace=replace)
        
    else: 
        
        raise ValueError(
                    f"{keys_data} must be a instance of str or dict."
                )
    




def load_hdf5_file(f_hdf5):
    
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

