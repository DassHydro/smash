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
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.util._typing import ListLike, FilePath
    from smash.fcore._mwd_setup import SetupDT
    from smash.fcore._mwd_mesh import MeshDT
    from smash.fcore._mwd_input_data import Input_DataDT


def _get_date_regex_pattern(dt: float, daily_interannual: bool) -> str:
    if daily_interannual:
        # % Match %m%d
        pattern = 2 * r"\d{2}"

    else:
        # % Match %Y%m%d%H%M
        if dt < 86_400:
            pattern = r"\d{4}" + 4 * r"\d{2}"
        # % Match %Y%m%d
        elif dt == 86_400:
            pattern = r"\d{4}" + 2 * r"\d{2}"
        # % Should be unreachable
        else:
            pass

    return pattern


def _find_index_files_containing_date(
    files: ListLike, date: pd.Timestamp, dt: float, daily_interannual: bool = False
) -> int:
    ind = -1
    regex_pattern = _get_date_regex_pattern(dt, daily_interannual)
    for i, f in enumerate(files):
        re_match = re.search(regex_pattern, f)
        if daily_interannual:
            fdate = pd.Timestamp(f"{date.year}{re_match.group()}")
        else:
            fdate = pd.Timestamp(re_match.group())
        if fdate < date:
            continue
        elif fdate == date:
            ind = i
        elif fdate > date:
            break
    return ind


def _get_atmos_files(
    dir: str,
    fmt: str,
    access: str,
    date_range: pd.DatetimeIndex,
    daily_interannual: bool = False,
) -> list[str]:
    # Set to drop duplicates after strftime
    date_range_strftime_access = (
        set(date_range.strftime(access)) if not access == "" else {""}
    )

    files = []
    for date_strftime_access in date_range_strftime_access:
        files.extend(
            glob.glob(f"{dir}/{date_strftime_access}/**/*{fmt}", recursive=True)
        )

    files = sorted(files)

    # Return all files if daily interannual (336 files max)
    if daily_interannual:
        return files

    else:
        # % Adjust list by removing files that are ealier than start_time
        regex_pattern = _get_date_regex_pattern(date_range.freq.n, daily_interannual)
        for i, f in enumerate(files):
            re_match = re.search(regex_pattern, f)
            fdate = pd.Timestamp(re_match.group())
            if fdate >= date_range[0]:
                ind = i
                break

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
    miss = []

    for i, c in enumerate(mesh.code):
        f = glob.glob(f"{setup.qobs_directory}/**/*{c}*.csv", recursive=True)

        if f:
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
        else:
            miss.append(c)

    if miss:
        warnings.warn(f"Missing {len(miss)} observed discharge file(s): {miss}")


# TODO: Refactorize each read atmos
def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]
    miss = []

    if setup.prcp_format == "tif":
        files = _get_atmos_files(
            setup.prcp_directory, setup.prcp_format, setup.prcp_access, date_range
        )

        for i, date in enumerate(tqdm(date_range, desc="</> Reading precipitation")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                miss.append(date.strftime("%Y-%m-%d %H:%M"))
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

                files = files[ind + 1 :]

    # % WIP
    elif setup.prcp_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    if miss:
        warnings.warn(f"Missing {len(miss)} precipitation file(s) for date(s): {miss}")


def _read_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]
    miss = []

    if setup.pet_format == "tif":
        files = _get_atmos_files(
            setup.pet_directory,
            setup.pet_format,
            setup.pet_access,
            date_range,
            setup.daily_interannual_pet,
        )

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
                    ind = _find_index_files_containing_date(
                        files, day, setup.dt, setup.daily_interannual_pet
                    )

                    if ind == -1:
                        miss.append(day.strftime("%m-%d"))
                        missing_day = np.append(missing_day, day.day_of_year)

                    else:
                        matrix_dip[..., i] = (
                            _read_windowed_raster(files[ind], mesh)
                            * setup.pet_conversion_factor
                        )

                        files = files[ind + 1 :]

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
                ind = _find_index_files_containing_date(
                    files, date, setup.dt, setup.daily_interannual_pet
                )

                if ind == -1:
                    miss.append(date.strftime("%Y-%m-%d %H:%M"))
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

                    files = files[ind + 1 :]

    # % WIP
    elif setup.pet_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    if miss:
        pet_kind = "daily interannual" if setup.daily_interannual_pet else ""
        warnings.warn(f"Missing {len(miss)} {pet_kind} pet file(s) for date(s): {miss}")


def _read_snow(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]
    miss = []

    if setup.snow_format == "tif":
        files = _get_atmos_files(
            setup.snow_directory, setup.snow_format, setup.snow_access, date_range
        )

        for i, date in enumerate(tqdm(date_range, desc="</> Reading snow")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                miss.append(date.strftime("%Y-%m-%d %H:%M"))
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

                files = files[ind + 1 :]

    # % WIP
    elif setup.snow_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    if miss:
        warnings.warn(f"Missing {len(miss)} snow file(s) for date(s): {miss}")


def _read_temp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]
    miss = []

    if setup.temp_format == "tif":
        files = _get_atmos_files(
            setup.temp_directory, setup.temp_format, setup.temp_access, date_range
        )

        for i, date in enumerate(tqdm(date_range, desc="</> Reading temperature")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                miss.append(date.strftime("%Y-%m-%d %H:%M"))
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

                files = files[ind + 1 :]

    # % WIP
    elif setup.temp_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    if miss:
        warnings.warn(f"Missing {len(miss)} temperature file(s) for date(s): {miss}")


def _read_descriptor(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    miss = []
    for i, name in enumerate(setup.descriptor_name):
        path = glob.glob(
            f"{setup.descriptor_directory}/**/{name}.tif*",
            recursive=True,
        )

        if len(path) == 0:
            miss.append(name)

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

    if miss:
        warnings.warn(f"Missing {len(miss)} descriptor file(s): {miss}")
