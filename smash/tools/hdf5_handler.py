from __future__ import annotations

import os
import h5py
import numpy as np

from smash.tools import object_handler

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
    >>> hdf5=smash.tools.hdf5_handler.open_hdf5("./my_hdf5.hdf5")
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
    >>> hdf5=smash.tools.hdf5_handler.open_hdf5("./model_subgroup.hdf5", replace=True)
    >>> hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="mygroup")
    >>> hdf5.keys()
    >>> hdf5.attrs.keys()
    """
    if subgroup is not None:
        
        if subgroup=="":
            
            subgroup="./"
        
        hdf5.require_group(subgroup)
    
    return hdf5



def _dump_object_to_hdf5_from_list_attribute(hdf5,instance,list_attr):
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
                
                _dump_object_to_hdf5_from_str_attribute(hdf5, instance, attr)
            
            elif isinstance(attr,list):
                
                _dump_object_to_hdf5_from_list_attribute(hdf5, instance, attr)
            
            elif isinstance(attr,dict):
                
                _dump_object_to_hdf5_from_dict_attribute(hdf5, instance, attr)
                
            else:
                
                raise ValueError(
                    f"inconsistent {attr} in {list_attr}. {attr} must be a an instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{list_attr} must be a instance of list."
                )



def _dump_object_to_hdf5_from_dict_attribute(hdf5,instance,dict_attr):
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
            
                _dump_object_to_hdf5_from_dict_attribute(hdf5[attr], sub_instance, value)
            
            if isinstance(value,list):
            
                _dump_object_to_hdf5_from_list_attribute(hdf5[attr], sub_instance, value)
            
            elif isinstance(value,str):
                
                _dump_object_to_hdf5_from_str_attribute(hdf5[attr], sub_instance, value)
            
            else :
                
                raise ValueError(
                    f"inconsistent '{attr}' in '{dict_attr}'. Dict({attr}) must be a instance of dict, list or str"
                )
    
    else:
        
        raise ValueError(
                    f"{dict_attr} must be a instance of dict."
                )



def _dump_object_to_hdf5_from_str_attribute(hdf5,instance,str_attr):
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



def _dump_object_to_hdf5_from_iteratable(hdf5, instance, iteratable=None):
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
    
    hdf5=smash.tools.hdf5_handler.open_hdf5("./model.hdf5", replace=True)
    hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="model1")
    keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="medium")
    smash.tools.hdf5_handler._dump_object_to_hdf5_from_iteratable(hdf5["model1"], model, keys_data)

    hdf5=smash.tools.hdf5_handler.open_hdf5("./model.hdf5", replace=False)
    hdf5=smash.tools.hdf5_handler.add_hdf5_sub_group(hdf5, subgroup="model2")
    keys_data=smash.io.hdf5_io.generate_smash_object_structure(model,typeofstructure="light")
    smash.tools.hdf5_handler._dump_object_to_hdf5_from_iteratable(hdf5["model2"], model, keys_data)
    """
    if isinstance(iteratable,list):
        
        _dump_object_to_hdf5_from_list_attribute(hdf5,instance,iteratable)
        
    elif isinstance(iteratable,dict):
        
        _dump_object_to_hdf5_from_dict_attribute(hdf5,instance,iteratable)
    
    else :
        
        raise ValueError(
                    f"{iteratable} must be a instance of list or dict."
                )



def _dump_dict_to_hdf5(hdf5,dictionary):
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
                    _dump_dict_to_hdf5(hdf5[attr],value)
                    
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
    
    smash.tools.hdf5_handler.save_dict_to_hdf5("saved_dictionary.hdf5",mesh)
    """
    if isinstance(dictionary,dict):
        
        hdf5=open_hdf5(path_to_hdf5, replace=replace)
        hdf5=add_hdf5_sub_group(hdf5, subgroup=location)
        _dump_dict_to_hdf5(hdf5[location], dictionary)
        
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
        keys_data=smash.tools.object_handler.generate_object_structure(instance)
    
    hdf5=open_hdf5(f_hdf5, replace=replace)
    hdf5=add_hdf5_sub_group(hdf5, subgroup=location)
    _dump_object_to_hdf5_from_iteratable(hdf5[location], instance, keys_data)
    
    if isinstance(sub_data,dict):
        
        _dump_dict_to_hdf5(hdf5[location], sub_data)
    
    hdf5.close()



def read_hdf5_as_dict(hdf5):
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
    hdf5=smash.tools.hdf5_handler.open_hdf5("./multi_model.hdf5")
    dictionary=smash.tools.hdf5_handler.read_hdf5_as_dict(hdf5["model1"])
    dictionary.keys()
    """
    dictionary={}
    
    for key,item in hdf5.items():
        
        if str(type(item)).find("group") != -1:
            
            dictionary.update({key:read_hdf5_as_dict(item)})
            
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

