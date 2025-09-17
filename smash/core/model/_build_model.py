from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import rasterio
from f90wrap.runtime import FortranDerivedType

from smash._constant import (
    DEFAULT_RR_INITIAL_STATES,
    DEFAULT_RR_PARAMETERS,
    DEFAULT_SERR_MU_PARAMETERS,
    DEFAULT_SERR_SIGMA_PARAMETERS,
    OPTIMIZABLE_NN_PARAMETERS,
    SERR_MU_MAPPING_PARAMETERS,
    SERR_SIGMA_MAPPING_PARAMETERS,
    STRUCTURE_ADJUST_CI,
    STRUCTURE_RR_PARAMETERS,
    STRUCTURE_RR_STATES,
)
from smash.core.model._read_input_data import (
    _read_descriptor,
    _read_imperviousness,
    _read_pet,
    _read_prcp,
    _read_qobs,
    _read_snow,
    _read_temp,
)
from smash.fcore._mw_atmos_statistic import (
    compute_mean_atmos as wrap_compute_mean_atmos,
)
from smash.fcore._mw_atmos_statistic import (
    compute_prcp_partitioning as wrap_compute_prcp_partitioning,
)
from smash.fcore._mw_interception_capacity import (
    adjust_interception_capacity as wrap_adjust_interception_capacity,
)
from smash.fcore._mwd_sparse_matrix_manipulation import (
    compute_rowcol_to_ind_ac as wrap_compute_rowcol_to_ind_ac,
)

if TYPE_CHECKING:
    from smash.fcore._mwd_input_data import Input_DataDT
    from smash.fcore._mwd_mesh import MeshDT
    from smash.fcore._mwd_output import OutputDT
    from smash.fcore._mwd_parameters import ParametersDT
    from smash.fcore._mwd_setup import SetupDT
    from smash.util._typing import ListLike


# % TODO: Move this function to a generic common function file
def _map_dict_to_fortran_derived_type(dct: dict, fdt: FortranDerivedType, skip: ListLike | None = None):
    if skip is None:
        skip = []
    for key, value in dct.items():
        if key in skip:
            continue

        if isinstance(value, dict):
            if hasattr(fdt, key):
                sub_fdt = getattr(fdt, key)
                if isinstance(sub_fdt, FortranDerivedType):
                    _map_dict_to_fortran_derived_type(value, sub_fdt)

                # % Should be unreachable
                else:
                    pass

            # % Same FortranDerivedType if key does not exist
            # % Recursive call
            else:
                _map_dict_to_fortran_derived_type(value, fdt)

        else:
            if hasattr(fdt, key):
                setattr(fdt, key, value)


def _build_mesh(setup: SetupDT, mesh: MeshDT):
    wrap_compute_rowcol_to_ind_ac(mesh)  # % Fortran subroutine
    mesh.local_active_cell = mesh.active_cell.copy()


def _build_input_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    if setup.read_qobs:
        _read_qobs(setup, mesh, input_data)

    with rasterio.Env():
        if setup.read_prcp:
            _read_prcp(setup, mesh, input_data)

        if setup.read_pet:
            _read_pet(setup, mesh, input_data)

        if setup.read_snow:
            _read_snow(setup, mesh, input_data)

        if setup.read_temp:
            _read_temp(setup, mesh, input_data)

        if setup.read_descriptor:
            _read_descriptor(setup, mesh, input_data)

        if setup.read_imperviousness:
            _read_imperviousness(setup, mesh, input_data)

    if setup.prcp_partitioning:
        wrap_compute_prcp_partitioning(setup, mesh, input_data)  # % Fortran subroutine

    if setup.compute_mean_atmos:
        print("</> Computing mean atmospheric data")
        wrap_compute_mean_atmos(setup, mesh, input_data)  # % Fortran subroutine


def _adjust_interception(
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
    parameters: ParametersDT,
    active_cell_only: bool = True,
):
    # % If structure contains ci and is at sub-daily time step and if user wants the capacity to be adjusted
    if STRUCTURE_ADJUST_CI[setup.structure] and setup.dt < 86_400:
        print("</> Adjusting GR interception capacity")
        # % Date
        day_index = pd.date_range(start=setup.start_time, end=setup.end_time, freq=f"{int(setup.dt)}s")[
            1:
        ].to_series()

        # % Date to proleptic Gregorian ordinal
        day_index = day_index.apply(lambda x: x.toordinal()).to_numpy()

        # % Scale to 1 (Fortran indexing)
        day_index = day_index - day_index[0] + 1

        ind = np.argwhere(parameters.rr_parameters.keys == "ci").item()

        if not active_cell_only:
            # % Backup active cell and local active cell
            active_cell_bak = mesh.active_cell.copy()
            local_active_cell_bak = mesh.local_active_cell.copy()
            # % Set all cells as active
            mesh.active_cell[:] = 1
            mesh.local_active_cell[:] = 1

        wrap_adjust_interception_capacity(
            setup,
            mesh,
            input_data,
            day_index,
            day_index[-1],
            parameters.rr_parameters.values[..., ind],
        )  # % Fortran subroutine

        if not active_cell_only:
            # % Restore active cell and local active cell if not adjusting for active cells only
            mesh.active_cell = active_cell_bak
            mesh.local_active_cell = local_active_cell_bak


def _build_parameters(
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
    parameters: ParametersDT,
):
    # % Build parameters
    parameters.rr_parameters.keys = STRUCTURE_RR_PARAMETERS[setup.structure]

    # % TODO: May be change this with a constant containing lambda functions such as
    # % SCALE_RR_PARAMETERS = [
    # % ...
    # % "llr": lambda x, dt, dx: x * dt / 3_600
    # % ...
    # % ]
    # % parameters.rr_parameters.values = SCALE_RR_PARAMETRS[key](DEFAULT_RR_PARAMETERS[key], setup.dt,
    # % mesh.dx)
    for i, key in enumerate(parameters.rr_parameters.keys):
        value = DEFAULT_RR_PARAMETERS[key]
        if key == "llr":
            value *= setup.dt / 3_600
        parameters.rr_parameters.values[..., i] = value

    # % Build initial states
    parameters.rr_initial_states.keys = STRUCTURE_RR_STATES[setup.structure]

    for i, key in enumerate(parameters.rr_initial_states.keys):
        value = DEFAULT_RR_INITIAL_STATES[key]
        parameters.rr_initial_states.values[..., i] = value

    if setup.adjust_interception:
        _adjust_interception(setup, mesh, input_data, parameters, active_cell_only=True)

    # % Build structural error mu parameters
    parameters.serr_mu_parameters.keys = SERR_MU_MAPPING_PARAMETERS[setup.serr_mu_mapping]

    for i, key in enumerate(parameters.serr_mu_parameters.keys):
        value = DEFAULT_SERR_MU_PARAMETERS[key]
        parameters.serr_mu_parameters.values[..., i] = value

    # % Build structural error sigma parameters
    parameters.serr_sigma_parameters.keys = SERR_SIGMA_MAPPING_PARAMETERS[setup.serr_sigma_mapping]

    for i, key in enumerate(parameters.serr_sigma_parameters.keys):
        value = DEFAULT_SERR_SIGMA_PARAMETERS[key]
        parameters.serr_sigma_parameters.values[..., i] = value

    # % Initalize weights and biases of ANN if hybrid model structure is used
    for key in OPTIMIZABLE_NN_PARAMETERS[max(0, setup.n_layers - 1)]:
        # zero init
        setattr(parameters.nn_parameters, key, 0)


def _build_output(
    setup: SetupDT,
    output: OutputDT,
):
    # % Build final states
    output.rr_final_states.keys = STRUCTURE_RR_STATES[setup.structure]
