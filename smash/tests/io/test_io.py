from __future__ import annotations

import smash

from smash.io._error import ReadHDF5MethodError

import os
import numpy as np
import pytest

def generic_mesh_io(**kwargs) -> dict:
    setup, mesh = smash.load_dataset("Cance")
    
    smash.save_mesh(mesh, "tmp_mesh.hdf5")
    
    mesh_rld = smash.read_mesh("tmp_mesh.hdf5")
    
    res = {"mesh_io." + k: np.array(v, ndmin=1) for (k, v) in mesh_rld.items()}

    return res

def test_mesh_io():
    
    res = generic_mesh_io()
    
    for key, value in res.items():
        
        if value.dtype.char == "U":
            value = value.astype("S")
        
        # % Check all values read from mesh
        assert np.array_equal(value, pytest.baseline[key][:]), key
            
    smash.save_model_ddt(pytest.model, "tmp_model_ddt.hdf5")
    
    # % Check ReadHDF5MethodError
    with pytest.raises(ReadHDF5MethodError):
        smash.read_mesh("tmp_model_ddt.hdf5")
