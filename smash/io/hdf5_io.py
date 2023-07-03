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



__all__ = ["save_object_to_hdf5", "save_dict_to_hdf5", "save_smash_model_to_hdf5", "load_hdf5_file", "read_object_as_dict"]



def open_hdf5(path, read_only=False, replace=False):
    """
    Open or create an HDF5 file.

    Parameters
    ----------
    path : str
        The file path.
    read_only : boolean
        If true the access to the hdf5 fil is in read-only mode. Multi process can read the same hdf5 file simulteneously. This is not possible when access mode are append 'a' or write 'w'.
    replace: Boolean
        If true, the existing hdf5file is erased

    Returns
    -------
    f :
        A HDF5 object.

    Examples
    --------
    >>> hdf5=smash.io.multi_model_io.open_hdf5("./my_hdf5.hdf5")
    >>> hdf5.keys()
    >>> hdf5.attrs.keys()
    """
    if not path.endswith(".hdf5"):
        
        path = path + ".hdf5"
    
    if read_only:
        
        if os.path.isfile(path):
            
            f= h5py.File(path, "r")
            
        else:
            
            raise ValueError(
                    f"File {path} does not exist."
                )
            
    else:
    
        if replace:
            
            f= h5py.File(path, "w")
            
        else:
            
            if os.path.isfile(path):
            
                f= h5py.File(path, "a")
            
            else:
                
                f= h5py.File(path, "w")
    
    return f



def add_hdf5_sub_group(hdf5, subgroup=None):
    """
    Create a new subgroup in a HDF5 object

    Parameters
    ----------
    hdf5 : object
        An hdf5 object opened with open_hdf5()
    subgroup: str
        Path to a subgroub that must be created

    Returns
    -------
    hdf5 :
        the HDF5 object.

    Examples
    --------
    >>> hdf5=smash.io.multi_model_io.open_hdf5("./model_subgroup.hdf5", replace=True)
    >>> hdf5=smash.io.multi_model_io.add_hdf5_sub_group(hdf5, subgroup="mygroup")
    >>> hdf5.keys()
    >>> hdf5.attrs.keys()
    """
    if subgroup is not None:
        
        if subgroup=="":
            
            subgroup="./"
        
        hdf5.require_group(subgroup)
    
    return hdf5


def generate_light_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
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



def generate_medium_smash_object_structure(structure: str,structure_parameters=STRUCTURE_PARAMETERS,structure_states=STRUCTURE_STATES):
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


def generate_full_smash_object_structure(instance):
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
    key_data=generate_object_structure(instance)
    
    key_list=list()
    key_list.append(key_data)
    key_list.append("_last_update")
    
    return key_list




def generate_object_structure(instance):
    """
    this function create a full dictionnary containing all the structure of an object in order to save it to an hdf5

    Parameters
    ----------
    instance : object
        a custom python object.
    
    Returns
    -------
    list or dict :
        A list or dictionary matching the structure of the python object.
    """
    key_data={}
    key_list=list()
    return_list=False
    
    for attr in dir(instance):
        
        if not attr.startswith("_") and not attr in ["from_handle", "copy"]:
            
            try:
                
                value = getattr(instance, attr)
                
                if isinstance(value, (np.ndarray,list)):
                    
                    if isinstance(value,list):
                        value=np.array(value)
                    
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
    """
    this function create a dictionnary containing a complete ar partial structure of an object in order to save it to an hdf5

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
        
        key_data=generate_light_smash_object_structure(structure)
        
    elif typeofstructure=="medium":
        
        key_data=generate_medium_smash_object_structure(structure)
        
    elif typeofstructure=="full":
        
        key_data=generate_full_smash_object_structure(instance)
    
    return key_data




def dump_object_to_hdf5_from_list_attribute(hdf5,instance,list_attr):
    """
    dump a object to a hdf5 file from a list of attributes

    Parameters
    ----------
    hdf5 : object
        an hdf5 object
    instance : object
        a custom python object.
    list_attr : list
        a list of attribute
    """
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
                    f"inconsistent {attr} in {list_attr}. {attr} must be a an instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{list_attr} must be a instance of list."
                )



def dump_object_to_hdf5_from_dict_attribute(hdf5,instance,dict_attr):
    """
    dump a object to a hdf5 file from a dictionary of attributes

    Parameters
    ----------
    hdf5 : object
        an hdf5 object
    instance : object
        a custom python object.
    dict_attr : dict
        a dictionary of attribute
    """
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
                    f"inconsistent '{attr}' in '{dict_attr}'. Dict({attr}) must be a instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{dict_attr} must be a instance of dict."
                )



def dump_object_to_hdf5_from_str_attribute(hdf5,instance,str_attr):
    """
    dump a object to a hdf5 file from a string attribute

    Parameters
    ----------
    hdf5 : object
        an hdf5 object
    instance : object
        a custom python object.
    str_attr : str
        a string attribute
    """
    if isinstance(str_attr, str):
        
        try:
            
            value = getattr(instance, str_attr)
            
            if isinstance(value, (np.ndarray,list)):
                
                if isinstance(value,list):
                    value=np.array(value)
                
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
                
            elif value is None:
                    
                    hdf5.attrs[str_attr] = "_None_"
            
            else:
                
                hdf5.attrs[str_attr] = value
                
        except:
            
            raise ValueError(
                f"Unable to dump attribute {str_attr} with value {value} from {instance}"
            )
    
    else:
        
        raise ValueError(
                    f"{str_attr} must be a instance of str."
                )



def dump_object_to_hdf5_from_iteratable(hdf5, instance, iteratable=None):
    """
    dump a object to a hdf5 file from a iteratable object list or dict

    Parameters
    ----------
    hdf5 : object
        an hdf5 object
    instance : object
        a custom python object.
    iteratable : list | dict
        a list or a dict of attribute
    
    Examples
    --------
    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)
    model.run(inplace=True)
    
    hdf5=smash.io.multi_model_io.open_hdf5("./model.hdf5", replace=True)
    hdf5=smash.io.multi_model_io.add_hdf5_sub_group(hdf5, subgroup="model1")
    keys_data=smash.io.multi_model_io.generate_smash_object_structure(model,typeofstructure="medium")
    smash.io.multi_model_io.dump_object_to_hdf5_from_iteratable(hdf5["model1"], model, keys_data)

    hdf5=smash.io.multi_model_io.open_hdf5("./model.hdf5", replace=False)
    hdf5=smash.io.multi_model_io.add_hdf5_sub_group(hdf5, subgroup="model2")
    keys_data=smash.io.multi_model_io.generate_smash_object_structure(model,typeofstructure="light")
    smash.io.multi_model_io.dump_object_to_hdf5_from_iteratable(hdf5["model2"], model, keys_data)
    """
    if isinstance(iteratable,list):
        
        dump_object_to_hdf5_from_list_attribute(hdf5,instance,iteratable)
        
    elif isinstance(iteratable,dict):
        
        dump_object_to_hdf5_from_dict_attribute(hdf5,instance,iteratable)
    
    else :
        
        raise ValueError(
                    f"{iteratable} must be a instance of list or dict."
                )



def dump_dict_to_hdf5(hdf5,dictionary):
    """
    dump a dictionary to an hdf5 file

    Parameters
    ----------
    hdf5 : object
        an hdf5 object
    dictionary : dict
        a custom python dictionary
    """
    if isinstance(dictionary,dict):
    
        for attr, value in dictionary.items():
            
            try:
            
                if isinstance(value,(dict)):
                    
                    hdf5=add_hdf5_sub_group(hdf5, subgroup=attr)
                    dump_dict_to_hdf5(hdf5[attr],value)
                    
                elif isinstance(value, (np.ndarray,list)):
                    
                    if isinstance(value,(list)):
                        value=np.array(value)
                    
                    if value.dtype == "object" or value.dtype.char == "U":
                        value = value.astype("S")
                    
                    #remove dataset if exist
                    if attr in hdf5.keys():
                        del hdf5[attr]
                    
                    hdf5.create_dataset(
                        attr,
                        shape=value.shape,
                        dtype=value.dtype,
                        data=value,
                        compression="gzip",
                        chunks=True,
                    )
                
                elif value is None:
                    
                    hdf5.attrs[attr] = "_None_"
                    
                else:
                    
                    hdf5.attrs[attr] = value
                
            except:
                
                raise ValueError(
                    f"Unable to save attribute {attr} with value {value}"
                )
    
    else:
        
        raise ValueError(
                    f"{dictionary} must be a instance of dict."
                )



def save_dict_to_hdf5(path_to_hdf5,dictionary=None,location="./",replace=False):
    """
    dump a dictionary to an hdf5 file

    Parameters
    ----------
    path_to_hdf5 : str
        path to the hdf5 file
    dictionary : dict | None
        a dictionary containing the data to be saved
    location : str
        path location or subgroup where to write data in the hdf5 file
    replace : Boolean
        replace an existing hdf5 file. Default is False
    
    Examples
    --------
    setup, mesh = smash.load_dataset("cance")
    model = smash.Model(setup, mesh)
    model.run(inplace=True)
    
    smash.io.multi_model_io.save_dict_to_hdf5("saved_dictionary.hdf5",mesh)
    """
    if isinstance(dictionary,dict):
        
        hdf5=open_hdf5(path_to_hdf5, replace=replace)
        hdf5=add_hdf5_sub_group(hdf5, subgroup=location)
        dump_dict_to_hdf5(hdf5[location], dictionary)
        
    else:
        
        raise ValueError(
                    f"The input {dictionary} must be a instance of dict."
                )



def save_object_to_hdf5(f_hdf5, instance, keys_data=None, location="./", sub_data=None, replace=False):
    """
    dump an object to an hdf5 file

    Parameters
    ----------
    f_hdf5 : str
        path to the hdf5 file
    instance : object
        python object
    keys_data : list | dict
        a list or a dictionary of the attribute to be saved
    location : str
        path location or subgroup where to write data in the hdf5 file
    sub_data : dict | None
        a dictionary containing extra-data to be saved
    replace : Boolean
        replace an existing hdf5 file. Default is False
    """
    
    if keys_data is None:
        keys_data=generate_object_structure(instance)
    
    hdf5=open_hdf5(f_hdf5, replace=replace)
    hdf5=add_hdf5_sub_group(hdf5, subgroup=location)
    dump_object_to_hdf5_from_iteratable(hdf5[location], instance, keys_data)
    
    if isinstance(sub_data,dict):
        
        dump_dict_to_hdf5(hdf5[location], sub_data)
    
    hdf5.close()



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
    
    keys_data=smash.generate_smash_object_structure(model,typeofstructure="medium")
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
        
        keys_data=generate_light_smash_object_structure(instance.setup.structure)
        
    elif content == "medium":
        
        keys_data=generate_medium_smash_object_structure(instance.setup.structure)
        
    elif content == "full":
        
        keys_data=generate_full_smash_object_structure(instance)
    
    if isinstance(keys_data,(dict,list)):
        
        save_object_to_hdf5(path_to_hdf5, instance, keys_data, location=location, sub_data=sub_data,replace=replace)
        
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
        
        hdf5=open_hdf5(f_hdf5, read_only=True, replace=False)
        dictionary=read_hdf5_to_dict(hdf5)
        hdf5.close()
        return dictionary




def read_object_as_dict(instance):
    """
    create a dictionary from a custom python object

    Parameters
    ----------
    instance : object
        an custom python object
    
    Return
    ----------
    key_data: dict
        an dictionary containing all keys and atributes of the object
    """
    key_data={}
    key_list=list()
    return_list=False
    
    for attr in dir(instance):
        
        if not attr.startswith("_") and not attr in ["from_handle", "copy"]:
            
            try:
                
                value = getattr(instance, attr)
                
                if isinstance(value, (np.ndarray,list)):
                    
                    if isinstance(value,list):
                        value=np.array(value)
                    
                    if value.dtype == "object" or value.dtype.char == "U":
                        value = value.astype("S")
                    
                    key_data.update({attr:value})
                    
                elif isinstance(value,(str,float,int)):
                    
                    key_data.update({attr:value})
                    
                else: 
                    
                    depp_key_data=read_object_as_dict(value)
                    
                    if (len(depp_key_data)>0):
                        key_data.update({attr:depp_key_data})
                    
            except:
                
                pass
    
    return key_data




def read_hdf5_to_dict(hdf5):
    """
    Load an hdf5 file

    Parameters
    ----------
    hdf5 : str
        path to the hdf5 file
    
    Return
    --------
    dictionary : dict, a dictionary of all keys and attribute included in the hdf5 file
    
    Examples
    --------
    #read only a part of an hdf5 file
    hdf5=smash.io.multi_model_io.open_hdf5("./multi_model.hdf5")
    dictionary=smash.io.multi_model_io.read_hdf5_to_dict(hdf5["model1"])
    dictionary.keys()
    """
    dictionary={}
    
    for key,item in hdf5.items():
        
        if str(type(item)).find("group") != -1:
            
            dictionary.update({key:read_hdf5_to_dict(item)})
            
            list_attr=list(item.attrs.keys())
            
            for key_attr in list_attr:
                
                # check if value is equal to "_None_" (None string because hdf5 does not supported)
                if item.attrs[key_attr] == "_None_":
                    
                    dictionary[key].update({key_attr:None})
                    
                else:
                    
                    dictionary[key].update({key_attr:item.attrs[key_attr]})
            
        if str(type(item)).find("dataset") != -1:
            
            if item[:].dtype.char == "S":
                
                values=item[:].astype("U")
                
            else:
                
                values=item[:]
            
            dictionary.update({key:values})
            
            list_attr=list(item.attrs.keys())
            
            for key_attr in list_attr:
                
                # check if value is equal to "_None_" (None string because hdf5 does not supported)
                if item.attrs[key_attr] == "_None_":
                    dictionary[key].update({key_attr:None})
                else:
                    dictionary.update({key_attr:item.attrs[key_attr]})
    
    list_attr=list(hdf5.attrs.keys())
    
    for key_attr in list_attr:
        
        # check if value is equal to "_None_" (None string because hdf5 does not supported)
        if hdf5.attrs[key_attr] == "_None_":
            dictionary.update({key_attr:None})
        else:
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
