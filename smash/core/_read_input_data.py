from __future__ import annotations

from smash.tools._common_function import (
    _index_containing_substring,
    _adjust_left_files_by_date,
)

from smash.core._constant import RATIO_PET_HOURLY

from smash.solver._mw_sparse_storage import matrix_to_sparse_vector_r

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import warnings
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from osgeo import gdal


def _read_windowed_raster(path: str, mesh: MeshDT) -> np.ndarray:
    ds = gdal.Open(path)

    transform = ds.GetGeoTransform()

    xmin = transform[0]
    ymax = transform[3]
    xres = transform[1]
    yres = -transform[5]

    col_off = (mesh.xmin - xmin) / xres
    row_off = (ymax - mesh.ymax) / yres

    band = ds.GetRasterBand(1)

    nodata = band.GetNoDataValue()

    arr = band.ReadAsArray(col_off, row_off, mesh.ncol, mesh.nrow)

    arr = np.where(arr == nodata, -99.0, arr)

    return arr


def _read_qobs(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    start_time = pd.Timestamp(setup.start_time)
    end_time = pd.Timestamp(setup.end_time)
    ind_miss = np.zeros(shape=mesh.ng, dtype=np.int32)

    for i, c in enumerate(mesh.code):
        f = glob.glob(f"{setup.qobs_directory}/**/*{c}*.csv", recursive=True)

        if f:
            ind_miss[i] = 1

            dat = pd.read_csv(f[0])
            try:
                file_start_time = pd.Timestamp(dat.columns[0])
            except:
                raise ValueError(
                    f"Column header '{dat.columns[0]}' in the observed discharge file for catchment {c} is not a valid date"
                )

            file_end_time = file_start_time + pd.Timedelta(seconds=setup.dt * len(dat))
            start_diff = (
                int((start_time - file_start_time).total_seconds() / setup.dt) + 1
            )
            end_diff = int((end_time - file_start_time).total_seconds() / setup.dt) + 1

            # % Check if observed discharge file contains data for corresponding simulation period
            if start_time > file_end_time or end_time < file_start_time:
                raise ValueError(
                    f"The provided observed discharge file for catchment {c} does not contain data for the selected simulation period ['{start_time}', '{end_time}']. The file covers the period ['{file_start_time}', '{file_end_time}']"
                )

            ind_start_dat = max(0, start_diff)
            ind_end_dat = min(dat.index.max(), end_diff)
            ind_start_arr = max(0, -start_diff)
            ind_end_arr = ind_start_arr + ind_end_dat - ind_start_dat

            input_data.obs_response.q[i, ind_start_arr:ind_end_arr] = dat.iloc[
                ind_start_dat:ind_end_dat, 0
            ]

    miss = [mesh.code[i] for i, im in enumerate(ind_miss) if not im]

    if miss:
        warnings.warn(f"Missing {len(miss)} observed discharge file(s): {miss}")


def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.prcp_format == "tif":
        files = sorted(glob.glob(f"{setup.prcp_directory}/**/*tif*", recursive=True))

        files = _adjust_left_files_by_date(files, date_range)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading precipitation")):
            date_strf = date.strftime("%Y%m%d%H%M")

            ind = _index_containing_substring(files, date_strf)

            if ind == -1:
                if setup.sparse_storage:
                    input_data.atmos_data.sparse_prcp[:, i] = -99.0

                else:
                    input_data.atmos_data.prcp[..., i] = -99.0

                warnings.warn(f"Missing precipitation file for date {date}")

            else:
                matrix = (
                    _read_windowed_raster(files[ind], mesh)
                    * setup.prcp_conversion_factor
                )

                if setup.sparse_storage:
                    matrix_to_sparse_vector_r(
                        mesh, matrix, input_data.atmos_data.sparse_prcp[:, i]
                    )

                else:
                    input_data.atmos_data.prcp[..., i] = matrix

                files.pop(ind)

    # % WIP
    elif setup.prcp_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")


def _read_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.pet_format == "tif":
        files = sorted(glob.glob(f"{setup.pet_directory}/**/*tif*", recursive=True))

        if not setup.daily_interannual_pet:
            files = _adjust_left_files_by_date(files, date_range)

        if setup.daily_interannual_pet:
            leap_year_days = pd.date_range(
                start="202001010000", end="202012310000", freq="1D"
            )
            nstep_per_day = int(86_400 / setup.dt)
            hourly_ratio = 3_600 / setup.dt

            if hourly_ratio >= 1:
                ratio = np.repeat(RATIO_PET_HOURLY, hourly_ratio) / hourly_ratio

            else:
                ratio = np.sum(
                    RATIO_PET_HOURLY.reshape(-1, int(1 / hourly_ratio)), axis=1
                )

            for i, day in enumerate(
                tqdm(leap_year_days, desc="</> Reading daily interannual pet")
            ):
                if day.day_of_year in date_range.day_of_year:
                    day_strf = day.strftime("%m%d")

                    ind = _index_containing_substring(files, day_strf)

                    ind_day = np.where(day.day_of_year == date_range.day_of_year)

                    if ind == -1:
                        if setup.sparse_storage:
                            input_data.sparse_pet[:, ind_day] = -99.0

                        else:
                            input_data.pet[..., ind_day] = -99.0

                        warnings.warn(
                            f"Missing daily interannual pet file for date {day_strf}"
                        )

                    else:
                        subset_date_range = date_range[ind_day]

                        matrix = (
                            _read_windowed_raster(files[ind], mesh)
                            * setup.pet_conversion_factor
                        )

                        if setup.sparse_storage:
                            vector = np.zeros(
                                shape=mesh.nac, dtype=np.float32, order="F"
                            )
                            matrix_to_sparse_vector_r(mesh, matrix, vector)

                        for j in range(nstep_per_day):
                            step = day + j * datetime.timedelta(seconds=setup.dt)

                            ind_step = subset_date_range.indexer_at_time(step)

                            if setup.sparse_storage:
                                input_data.atmos_data.sparse_pet[
                                    :, ind_day[0][ind_step]
                                ] = (
                                    np.repeat(
                                        vector[:, np.newaxis], len(ind_step), axis=1
                                    )
                                    * ratio[j]
                                )

                            else:
                                input_data.atmos_data.pet[..., ind_day[0][ind_step]] = (
                                    np.repeat(
                                        matrix[..., np.newaxis], len(ind_step), axis=2
                                    )
                                    * ratio[j]
                                )

                        files.pop(ind)

        else:
            for i, date in enumerate(tqdm(date_range, desc="</> Reading pet")):
                date_strf = date.strftime("%Y%m%d%H%M")

                ind = _index_containing_substring(files, date_strf)

                if ind == -1:
                    if setup.sparse_storage:
                        input_data.atmos_data.sparse_pet[:, i] = -99.0

                    else:
                        input_data.atmos_data.pet[..., i] = -99.0

                    warnings.warn(f"Missing pet file for date {date}")

                else:
                    matrix = (
                        _read_windowed_raster(files[ind], mesh)
                        * setup.pet_conversion_factor
                    )

                    if setup.sparse_storage:
                        matrix_to_sparse_vector_r(
                            mesh, matrix, input_data.atmos_data.sparse_pet[:, i]
                        )

                    else:
                        input_data.atmos_data.pet[..., i] = matrix

                    files.pop(ind)

    # % WIP
    elif setup.pet_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")


def _read_descriptor(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    for i, name in enumerate(setup.descriptor_name):
        path = glob.glob(
            f"{setup.descriptor_directory}/**/{name}.tif*",
            recursive=True,
        )

        if len(path) == 0:
            warnings.warn(
                f"No descriptor file '{name}.tif' in recursive root directory '{setup.descriptor_directory}'"
            )

        else:
            input_data.physio_data.descriptor[..., i] = _read_windowed_raster(
                path[0], mesh
            )
            input_data.physio_data._l_descriptor[i] = np.min(
                input_data.physio_data.descriptor[..., i]
            )
            input_data.physio_data._u_descriptor[i] = np.max(
                input_data.physio_data.descriptor[..., i]
            )
