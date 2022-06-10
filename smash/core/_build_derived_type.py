from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.m_setup import SetupDT
    from smash.solver.m_mesh import MeshDT
    
from smash.solver.m_mesh import compute_mesh_path, compute_global_active_cell
    
from pandas import Timestamp
import numpy as np

def _derived_type_parser(derived_type, data: dict):

    """
    Derived type parser
    """

    for key, value in data.items():

        if hasattr(derived_type, key):
            setattr(derived_type, key, value)

        else:
            warnings.warn(
                f"'{key}' key does not belong to the derived type {type(derived_type)}",
                UserWarning,
            )

def _standardize_setup(setup: SetupDT):
    
    """
    Check every SetupDT error/warning exception
    """

    if setup.dt < 0:
        raise ValueError("argument dt of SetupDT is lower than 0")

    if not setup.dt in [900, 3_600, 86_400]:
        warnings.warn(
            "argument dt of SetupDT is not set to a classical value (900, 3600, 86400 seconds)",
            UserWarning,
        )

    if setup.dx < 0:
        raise ValueError("argument dx of SetupDT is lower than 0")

    if setup.start_time.decode() == "":
        raise ValueError("argument start_time of SetupDT is not defined")

    if setup.end_time.decode() == "":
        raise ValueError("argument end_time of SetupDT is not defined")

    try:
        st = Timestamp(setup.start_time.decode())
    except:
        raise ValueError("argument start_time of SetupDT is not a valid date")

    try:
        et = Timestamp(setup.end_time.decode())
    except:
        raise ValueError("argument end_time of SetupDT is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time of SetupDT corresponds to an earlier date than start_time"
        )

    if setup.optim_start_time.decode() == "":
        setup.optim_start_time = setup.start_time
        warnings.warn(
            "argument optim_start_time of SetupDT is not defined. Value set to start_time",
            UserWarning,
        )

    try:
        ost = Timestamp(setup.optim_start_time.decode())
    except:
        raise ValueError("argument optim_start_time of SetupDT is not a valid date")

    if (ost - st).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to an earlier date than start_time"
        )

    if (et - ost).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to a later date than end_time"
        )

def _standardize_mesh(setup: SetupDT, mesh: MeshDT):
    
    """
    Check every MeshDT error/warning exception
    """
    
    if mesh.ncol < 0:
        raise ValueError("argument ncol of MeshDT is lower than 0")
        
    if mesh.nrow < 0:
        raise ValueError("argument nrow of MeshDT is lower than 0")
        
    if mesh.ng < 0:
        raise ValueError("argument ng of MeshDT is lower than 0")
        
    if mesh.xll < 0:
        raise ValueError("argument xll of MeshDT is lower than 0")
        
    if mesh.yll < 0:
        raise ValueError("argument yll of MeshDT is lower than 0")
        
    if np.any(mesh.area < 0):
        raise ValueError("argument area of MeshDT contains at least one value lower than 0")
        
    if np.all(mesh.flow == -99):
        raise ValueError("argument flow of MeshDT contains only NaN value")
        
    if np.all(mesh.drained_area == -99):
        raise ValueError("argument drained_area of MeshDT contains only NaN value")
        
def _build_setup(setup: SetupDT):
    
    """
    Build setup
    """
    
    _standardize_setup(setup)

    st = Timestamp(setup.start_time.decode())
    ost = Timestamp(setup.optim_start_time.decode())
    et = Timestamp(setup.end_time.decode())

    setup.nb_time_step = (et - st).total_seconds() / setup.dt

    setup.optim_start_step = (ost - st).total_seconds() / setup.dt + 1

def _build_mesh(setup: SetupDT, mesh: MeshDT):
    
    """
    Build mesh
    """
    
    _standardize_mesh(setup, mesh)
    
    compute_mesh_path(mesh)
    
    compute_global_active_cell(setup, mesh)
    
