from __future__ import annotations

import warnings
import glob
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.m_setup import SetupDT
    from smash.solver.m_mesh import MeshDT
    from smash.solver.m_input_data import Input_DataDT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    if setup.start_time.decode() == "...":
        raise ValueError("argument start_time of SetupDT is not defined")

    if setup.end_time.decode() == "...":
        raise ValueError("argument end_time of SetupDT is not defined")

    try:
        st = pd.Timestamp(setup.start_time.decode())
    except:
        raise ValueError("argument start_time of SetupDT is not a valid date")

    try:
        et = pd.Timestamp(setup.end_time.decode())
    except:
        raise ValueError("argument end_time of SetupDT is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time of SetupDT corresponds to an earlier date than start_time"
        )

    if setup.optim_start_time.decode() == "...":
        setup.optim_start_time = setup.start_time
        warnings.warn(
            "argument optim_start_time of SetupDT is not defined. Value set to start_time",
            UserWarning,
        )

    try:
        ost = pd.Timestamp(setup.optim_start_time.decode())
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

    if not setup.active_cell_only and setup.sparse_forcing:
        raise ValueError(
            "argument sparse_forcing of SetupDT can not be True if active_cell_only of SetupDT is False"
        )

    if setup.active_cell_only and not setup.sparse_forcing:
        warnings.warn(
            "argument sparse_forcing of SetupDT is False but active_cell_only of SetupDT is True"
        )

    if setup.simulation_only:
        setup.read_qobs = False

    else:
        if setup.read_qobs and setup.qobs_directory.decode() == "...":
            raise ValueError(
                "argument simulation_only of SetupDT is False, read_qobs of SetupDT is True and qobs_directory of SetupDT is not defined"
            )

    # TODO, check for better warning/error callbacks


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
        raise ValueError(
            "argument area of MeshDT contains at least one value lower than 0"
        )

    if np.all(mesh.flow == -99):
        raise ValueError("argument flow of MeshDT contains only NaN value")

    if np.all(mesh.drained_area == -99):
        raise ValueError("argument drained_area of MeshDT contains only NaN value")

    # TODO add check for remaining MeshDT attributes


def _build_setup(setup: SetupDT):

    """
    Build setup
    """

    _standardize_setup(setup)

    st = pd.Timestamp(setup.start_time.decode())
    ost = pd.Timestamp(setup.optim_start_time.decode())
    et = pd.Timestamp(setup.end_time.decode())

    setup.nb_time_step = (et - st).total_seconds() / setup.dt

    setup.optim_start_step = (ost - st).total_seconds() / setup.dt + 1


def _compute_mesh_path(mesh: MeshDT):
    
    ind = np.unravel_index(np.argsort(mesh.drained_area, axis=None), mesh.drained_area.shape)
    
    # Transform from Python to Fortran array indexing
    mesh.path[0,:] = ind[0][:] + 1
    mesh.path[1,:] = ind[1][:] + 1
    
def _build_mesh(setup: SetupDT, mesh: MeshDT):

    """
    Build mesh
    """

    _standardize_mesh(setup, mesh)
    
    _compute_mesh_path(mesh)

def _read_qobs(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    st = pd.Timestamp(setup.start_time.decode())

    code = mesh.code.tobytes(order="F").decode("utf-8").split()

    for i, c in enumerate(code):

        path = glob.glob(
            f"{setup.qobs_directory.decode()}/**/*{c}*.csv", recursive=True
        )

        if len(path) == 0:
            warnings.warn(
                f"No observed discharge file for catchment {c} in recursive root directory {setup.qobs_directory.decode()}"
            )

        elif len(path) > 1:
            raise ValueError(
                f"There is more than one file containing the name of the catchment {c}"
            )

        else:

            with open(path[0], "r") as f:

                header = pd.Timestamp(f.readline())

                time_diff = int((st - header).total_seconds() / setup.dt) + 1

                if time_diff > 0:

                    k = 0

                    for j, line in enumerate(f):

                        if j >= time_diff:

                            try:
                                input_data.qobs[i, k] = float(line)

                                k += 1

                            except:
                                break
                else:

                    k = -time_diff

                    for line in f:

                        try:
                            input_data.qobs[i, k] = float(line)

                            k += 1

                        except:
                            break


def _build_input_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    """
    Build input_data
    """

    if setup.read_qobs:

        _read_qobs(setup, mesh, input_data)
