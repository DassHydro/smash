from __future__ import annotations

from smash.solver._mw_routine import (
    compute_rowcol_to_ind_sparse,
    compute_mean_forcing,
    compute_prcp_indice,
)

from smash.core._read_input_data import _read_qobs, _read_prcp, _read_pet, _read_descriptor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import warnings
import os
import errno
import pandas as pd
import numpy as np

def _parse_derived_type(derived_type, data: dict):

    """
    Derived type parser
    """

    for key, value in data.items():

        if hasattr(derived_type, key):

            setattr(derived_type, key, value)

        else:
            warnings.warn(
                f"'{key}' key does not belong to the derived type '{type(derived_type)}'"
            )


def _standardize_setup(setup: SetupDT):

    """
    Check every SetupDT error/warning exception
    """

    if setup.dt < 0:
        raise ValueError("argument dt is lower than 0")

    if not setup.dt in [900, 3_600, 86_400]:
        warnings.warn(
            "argument dt is not set to a classical value (900, 3600, 86400 seconds)",
            UserWarning,
        )

    if setup.start_time == "...":
        raise ValueError("argument start_time is not defined")

    if setup.end_time == "...":
        raise ValueError("argument end_time is not defined")

    try:
        st = pd.Timestamp(setup.start_time)
    except:
        raise ValueError("argument start_time is not a valid date")

    try:
        et = pd.Timestamp(setup.end_time)
    except:
        raise ValueError("argument end_time is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time corresponds to an earlier date than start_time"
        )

    if setup.read_qobs and setup.qobs_directory == "...":
        raise ValueError(
            "argument read_qobs is True and qobs_directory is not defined"
        )

    if setup.read_qobs and not os.path.exists(setup.qobs_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.qobs_directory,
        )

    if setup.read_prcp and setup.prcp_directory == "...":
        raise ValueError(
            "argument read_prcp is True and prcp_directory is not defined"
        )

    if setup.read_prcp and not os.path.exists(setup.prcp_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.prcp_directory,
        )

    if not setup.prcp_format in ["tif", "nc"]:
        raise ValueError(
            f"Unknown prcp_format '{setup.prcp_format}'. Choices: {['tif', 'nc']}"
        )

    if setup.prcp_conversion_factor < 0:
        raise ValueError("argument prcp_conversion_factor is lower than 0")

    if setup.read_pet and setup.pet_directory == "...":
        raise ValueError(
            "argument read_pet is True and pet_directory is not defined"
        )

    if setup.read_pet and not os.path.exists(setup.pet_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.pet_directory,
        )

    if not setup.pet_format in ["tif", "nc"]:
        raise ValueError(
            f"Unknown pet_format '{setup.pet_format}'. Choices: {['tif', 'nc']}"
        )

    if setup.pet_conversion_factor < 0:
        raise ValueError("argument pet_conversion_factor is lower than 0")

    if setup.read_descriptor and setup.descriptor_directory == "...":
        raise ValueError(
            "argument read_descriptor is True and descriptor_directory is not defined"
        )

    if setup.read_descriptor and not os.path.exists(setup.descriptor_directory):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            setup.descriptor_directory,
        )

    if setup.read_descriptor and setup._nd == 0:
        raise ValueError(
            "argument read_descriptor is True and descriptor_name is not defined"
        )

    if not setup.descriptor_format in ["tif", "nc"]:
        raise ValueError(
            f"Unknown descriptor_format '{setup.descriptor_format}'. Choices: {['tif', 'nc']}"
        )

    if setup.interception_module < 0 or setup.interception_module > 1:
        raise ValueError(
            f"argument interception_module must be in [0, 1] not {setup.interception_module}"
        )

    if setup.production_module < 0 or setup.production_module > 1:
        raise ValueError(
            f"argument production_module must be in [0, 1] not {setup.production_module}"
        )

    if setup.exchange_module < 0 or setup.exchange_module > 2:
        raise ValueError(
            f"argument exchange_module must be in [0, 2] not {setup.exchange_module}"
        )

    if setup.transfer_module < 0 or setup.transfer_module > 3:
        raise ValueError(
            f"argument transfer_module must be in [0, 3] not {setup.transfer_module}"
        )

    if setup.routing_module < 0 or setup.routing_module > 3:
        raise ValueError(
            f"argument routing_module must be in [0, 3] not {setup.routing_module}"
        )


def _build_setup(setup: SetupDT):

    """
    Build setup
    """

    _standardize_setup(setup)

    st = pd.Timestamp(setup.start_time)

    et = pd.Timestamp(setup.end_time)

    setup._ntime_step = (et - st).total_seconds() / setup.dt


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

    if mesh.xmin < 0:
        raise ValueError("argument xmin of MeshDT is lower than 0")

    if mesh.ymax < 0:
        raise ValueError("argument ymax of MeshDT is lower than 0")

    if np.all(mesh.flwdir == -99):
        raise ValueError("argument flwdir of MeshDT contains only NaN value")

    if np.all(mesh.drained_area == -99):
        raise ValueError("argument drained_area of MeshDT contains only NaN value")


def _build_mesh(setup: SetupDT, mesh: MeshDT):

    """
    Build mesh
    """

    _standardize_mesh(setup, mesh)

    if setup.sparse_storage:

        compute_rowcol_to_ind_sparse(mesh)  #% Fortran subroutine mw_routine


def _build_input_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    """
    Build input_data
    """

    if setup.read_qobs:

        _read_qobs(setup, mesh, input_data)

    if setup.read_prcp:

        _read_prcp(setup, mesh, input_data)

    if setup.read_pet:

        _read_pet(setup, mesh, input_data)

    if setup.mean_forcing:

        compute_mean_forcing(setup, mesh, input_data)  #% Fortran subroutine mw_routine

    if setup.prcp_indice:

        compute_prcp_indice(setup, mesh, input_data)  #% Fortran subroutine mw_routine

    if setup.read_descriptor:

        _read_descriptor(setup, mesh, input_data)
