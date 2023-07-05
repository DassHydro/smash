from __future__ import annotations

import os
import numpy as np


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

