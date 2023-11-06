from __future__ import annotations

from smash._constant import RATIO_PET_HOURLY

from smash.fcore._mwd_sparse_matrix_manipulation import (
    matrix_to_sparse_matrix as wrap_matrix_to_sparse_matrix,
)

import warnings
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from osgeo import gdal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import ListLike, FilePath
    from smash.fcore._mwd_setup import SetupDT
    from smash.fcore._mwd_mesh import MeshDT
    from smash.fcore._mwd_input_data import Input_DataDT


def _index_files_containing_date(
    files: ListLike, date: pd.Timestamp, dt: float, daily_interannual: bool = False
):
    for i, s in enumerate(files):
        if date.strftime("%Y%m%d%H%M") in s:
            return i
        # % Allow %Y%m%d at daily time step
        elif dt == 86_400 and date.strftime("%Y%m%d") in s:
            return i
        # % Allow %m%d when reading daily interannual
        elif daily_interannual and date.strftime("%m%d") in s:
            return i

    return -1


def _adjust_left_files_by_date(
    files: ListLike, date_range: pd.DatetimeIndex, dt: float
):
    n = 0
    ind = -1
    while ind == -1 and n < len(date_range):
        ind = _index_files_containing_date(files, date_range[n], dt)

        n += 1

    return files[ind:]


def _read_windowed_raster(path: FilePath, mesh: MeshDT) -> np.ndarray:
    gdal.UseExceptions()
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
                    f"Column header '{dat.columns[0]}' in the observed discharge file for catchment '{c}' is not a valid date"
                )

            file_end_time = file_start_time + pd.Timedelta(seconds=setup.dt * len(dat))
            start_diff = (
                int((start_time - file_start_time).total_seconds() / setup.dt) + 1
            )
            end_diff = int((end_time - file_start_time).total_seconds() / setup.dt) + 1

            # % Check if observed discharge file contains data for corresponding simulation period
            if start_time > file_end_time or end_time < file_start_time:
                raise ValueError(
                    f"The provided observed discharge file for catchment '{c}' does not contain data for the selected simulation period ['{start_time}', '{end_time}']. The file covers the period ['{file_start_time}', '{file_end_time}']"
                )

            ind_start_dat = max(0, start_diff)
            ind_end_dat = min(dat.index.max(), end_diff)
            ind_start_arr = max(0, -start_diff)
            ind_end_arr = ind_start_arr + ind_end_dat - ind_start_dat

            input_data.response_data.q[i, ind_start_arr:ind_end_arr] = dat.iloc[
                ind_start_dat:ind_end_dat, 0
            ]

    miss = [mesh.code[i] for i, im in enumerate(ind_miss) if not im]

    if miss:
        warnings.warn(f"Missing {len(miss)} observed discharge file(s): {miss}")


# TODO: Refactorize each read atmos
def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.prcp_format == "tif":
        files = sorted(glob.glob(f"{setup.prcp_directory}/**/*tif*", recursive=True))

        files = _adjust_left_files_by_date(files, date_range, setup.dt)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading precipitation")):
            ind = _index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                if setup.sparse_storage:
                    matrix = np.zeros(
                        shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F"
                    )
                    matrix.fill(np.float32(-99))
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(-99),
                        input_data.atmos_data.sparse_prcp[i],
                    )

                else:
                    input_data.atmos_data.prcp[..., i] = np.float32(-99)

                warnings.warn(f"Missing precipitation file for date {date}")

            else:
                matrix = (
                    _read_windowed_raster(files[ind], mesh)
                    * setup.prcp_conversion_factor
                )

                if setup.sparse_storage:
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(0),
                        input_data.atmos_data.sparse_prcp[i],
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
            files = _adjust_left_files_by_date(files, date_range, setup.dt)

        if setup.daily_interannual_pet:
            leap_year_days = pd.date_range(
                start="202001010000", end="202012310000", freq="1D"
            )
            step_offset = int(
                (date_range[0] - date_range[0].floor("D")).total_seconds() / setup.dt
            )
            nstep_per_day = int(86_400 / setup.dt)
            hourly_ratio = 3_600 / setup.dt

            if hourly_ratio >= 1:
                ratio = np.repeat(RATIO_PET_HOURLY, hourly_ratio) / hourly_ratio

            else:
                ratio = np.sum(
                    RATIO_PET_HOURLY.reshape(-1, int(1 / hourly_ratio)), axis=1
                )

            matrix_dip = np.zeros(
                shape=(mesh.nrow, mesh.ncol, len(leap_year_days)), dtype=np.float32
            )
            missing_day = np.empty(shape=0, dtype=np.int32)

            for i, day in enumerate(
                tqdm(leap_year_days, desc="</> Reading daily interannual pet")
            ):
                if day.day_of_year in date_range.day_of_year:
                    ind = _index_files_containing_date(
                        files, day, setup.dt, setup.daily_interannual_pet
                    )

                    if ind == -1:
                        missing_day = np.append(missing_day, day.day_of_year)

                        warnings.warn(
                            f"Missing daily interannual pet file for date {day.strftime('%m%d')}"
                        )

                    else:
                        matrix_dip[..., i] = (
                            _read_windowed_raster(files[ind], mesh)
                            * setup.pet_conversion_factor
                        )

                        files.pop(ind)

            for i, date in enumerate(
                tqdm(date_range, desc="</> Disaggregating daily interannual pet")
            ):
                day = date.day_of_year
                ratio_ind = (i + step_offset) % nstep_per_day

                if day in missing_day:
                    if setup.sparse_storage:
                        matrix = np.zeros(
                            shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F"
                        )
                        matrix.fill(np.float32(-99))
                        wrap_matrix_to_sparse_matrix(
                            mesh,
                            matrix,
                            np.float32(-99),
                            input_data.atmos_data.sparse_pet[i],
                        )
                    else:
                        input_data.atmos_data.pet[..., i] = np.float32(-99)

                else:
                    matrix = matrix_dip[..., day - 1] * ratio[ratio_ind]
                    if setup.sparse_storage:
                        wrap_matrix_to_sparse_matrix(
                            mesh,
                            matrix,
                            np.float32(0),
                            input_data.atmos_data.sparse_pet[i],
                        )
                    else:
                        input_data.atmos_data.pet[..., i] = matrix

        else:
            for i, date in enumerate(tqdm(date_range, desc="</> Reading pet")):
                ind = _index_files_containing_date(files, date, setup.dt)

                if ind == -1:
                    if setup.sparse_storage:
                        matrix = np.zeros(
                            shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F"
                        )
                        matrix.fill(np.float32(-99))
                        wrap_matrix_to_sparse_matrix(
                            mesh,
                            matrix,
                            np.float32(-99),
                            input_data.atmos_data.sparse_pet[i],
                        )

                    else:
                        input_data.atmos_data.pet[..., i] = np.float32(-99)

                    warnings.warn(f"Missing pet file for date {date}")

                else:
                    matrix = (
                        _read_windowed_raster(files[ind], mesh)
                        * setup.pet_conversion_factor
                    )

                    if setup.sparse_storage:
                        wrap_matrix_to_sparse_matrix(
                            mesh,
                            matrix,
                            np.float32(0),
                            input_data.atmos_data.sparse_pet[i],
                        )

                    else:
                        input_data.atmos_data.pet[..., i] = matrix

                    files.pop(ind)

    # % WIP
    elif setup.pet_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")


def _read_snow(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.snow_format == "tif":
        files = sorted(glob.glob(f"{setup.snow_directory}/**/*tif*", recursive=True))

        files = _adjust_left_files_by_date(files, date_range, setup.dt)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading snow")):
            ind = _index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                if setup.sparse_storage:
                    matrix = np.zeros(
                        shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F"
                    )
                    matrix.fill(np.float32(-99))
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(-99),
                        input_data.atmos_data.sparse_snow[i],
                    )

                else:
                    input_data.atmos_data.snow[..., i] = np.float32(-99)

                warnings.warn(f"Missing snow file for date {date}")

            else:
                matrix = (
                    _read_windowed_raster(files[ind], mesh)
                    * setup.snow_conversion_factor
                )

                if setup.sparse_storage:
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(0),
                        input_data.atmos_data.sparse_snow[i],
                    )

                else:
                    input_data.atmos_data.snow[..., i] = matrix

                files.pop(ind)

    # % WIP
    elif setup.snow_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")


def _read_temp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.temp_format == "tif":
        files = sorted(glob.glob(f"{setup.temp_directory}/**/*tif*", recursive=True))

        files = _adjust_left_files_by_date(files, date_range, setup.dt)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading temperature")):
            ind = _index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                if setup.sparse_storage:
                    matrix = np.zeros(
                        shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F"
                    )
                    # We can assume that -99 is too cold
                    matrix.fill(np.float32(-99))
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(-99),
                        input_data.atmos_data.sparse_temp[i],
                    )

                else:
                    input_data.atmos_data.temp[..., i] = np.float32(-99)

                warnings.warn(f"Missing temperature file for date {date}")

            else:
                matrix = _read_windowed_raster(files[ind], mesh)

                if setup.sparse_storage:
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(0),
                        input_data.atmos_data.sparse_temp[i],
                    )

                else:
                    input_data.atmos_data.temp[..., i] = matrix

                files.pop(ind)

    # % WIP
    elif setup.temp_format == "nc":
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
            input_data.physio_data.l_descriptor[i] = np.min(
                input_data.physio_data.descriptor[..., i]
            )
            input_data.physio_data.u_descriptor[i] = np.max(
                input_data.physio_data.descriptor[..., i]
            )
