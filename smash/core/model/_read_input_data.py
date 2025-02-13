from __future__ import annotations

import glob
import os
import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from smash._constant import RATIO_PET_HOURLY
from smash.fcore._mwd_sparse_matrix_manipulation import (
    matrix_to_sparse_matrix as wrap_matrix_to_sparse_matrix,
)

if TYPE_CHECKING:
    from smash.fcore._mwd_input_data import Input_DataDT
    from smash.fcore._mwd_mesh import MeshDT
    from smash.fcore._mwd_setup import SetupDT
    from smash.util._typing import FilePath, ListLike


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
        re_match = re.search(regex_pattern, os.path.basename(f))
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
    date_range_strftime_access = set(date_range.strftime(access)) if access else {""}

    files = []
    for date_strftime_access in date_range_strftime_access:
        files.extend(glob.glob(f"{dir}/{date_strftime_access}/**/*{fmt}", recursive=True))

    files = sorted(files)

    # Return all files if daily interannual (336 files max)
    if daily_interannual:
        return files

    else:
        # % Adjust list by removing files that are ealier than start_time
        regex_pattern = _get_date_regex_pattern(date_range.freq.n, daily_interannual)
        for i, f in enumerate(files):
            re_match = re.search(regex_pattern, os.path.basename(f))
            fdate = pd.Timestamp(re_match.group())
            if fdate >= date_range[0]:
                return files[i:]

    # % Return an empty list in case we did not return ealier. It means that we do not have files
    # % that match the time period. It will return a missing warning in the corresponding read atmos function.
    return []


def _read_windowed_raster(path: FilePath, mesh: MeshDT) -> tuple[np.ndarray, dict[str, int]]:
    warning = {"res": 0, "overlap": 0, "outofbound": 0}

    # % Get raster information
    with rasterio.open(path) as ds:
        transform = ds.get_transform()
        xmin = transform[0]
        ymax = transform[3]
        xres = transform[1]
        yres = -transform[5]

        # % Manage absolute tolerance wrt to xres or yres value
        atol = 1e-2
        xatol = atol * 10 ** min(0, np.floor(np.log10(np.abs(xres))))
        yatol = atol * 10 ** min(0, np.floor(np.log10(np.abs(yres))))

        # % Resolution missmatch
        if not np.isclose(mesh.xres, xres, atol=xatol) or not np.isclose(mesh.yres, yres, atol=yatol):
            warning["res"] = 1

        # % Overlap missmatch
        dxmin = mesh.xmin - xmin
        dymax = ymax - mesh.ymax
        xol_match = np.abs(dxmin / xres - np.round(dxmin / xres))
        yol_match = np.abs(dymax / yres - np.round(dymax / yres))
        if not np.isclose(xol_match, 0, atol=xatol) or not np.isclose(yol_match, 0, atol=yatol):
            warning["overlap"] = 1

        # # % Allocate buffer
        arr = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32)
        arr.fill(np.float32(-99))

        # % Pad offset to the nearest integer
        xoff = np.rint((mesh.xmin - xmin) / xres)
        yoff = np.rint((ymax - mesh.ymax) / yres)

        # % Resolution ratio
        xres_ratio = mesh.xres / xres
        yres_ratio = mesh.yres / yres

        # Reading window
        win_xsize = np.rint(mesh.ncol * xres_ratio)
        win_ysize = np.rint(mesh.nrow * yres_ratio)

        # % Totally out of bound
        # % Return the buffer with no data
        if xoff >= ds.width or yoff >= ds.height or xoff + win_xsize <= 0 or yoff + win_ysize <= 0:
            warning["outofbound"] = 2
            return arr, warning

        # % Partially out of bound
        if xoff < 0 or yoff < 0 or xoff + win_xsize > ds.width or yoff + win_ysize > ds.height:
            warning["outofbound"] = 1

        # % Readjust offset
        pxoff = max(0, xoff)
        pyoff = max(0, yoff)

        # % Readjust reading window
        pwin_xsize = min(win_xsize + min(0, xoff), ds.width - pxoff)
        pwin_ysize = min(win_ysize + min(0, yoff), ds.height - pyoff)

        # % Buffer slices
        xs = pxoff - xoff
        ys = pyoff - yoff
        xe = xs + pwin_xsize
        ye = ys + pwin_ysize
        arr_xslice = slice(int(np.rint(xs * 1 / xres_ratio)), int(np.rint(xe * 1 / xres_ratio)))
        arr_yslice = slice(int(np.rint(ys * 1 / yres_ratio)), int(np.rint(ye * 1 / yres_ratio)))

        # % Reading and writting into buffer
        ds.read(
            1,
            window=rasterio.windows.Window(pxoff, pyoff, pwin_xsize, pwin_ysize),
            out=arr[arr_yslice, arr_xslice],
        )

        # % Manage NoData
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, -99.0, arr)

        return arr, warning


def _get_reading_warning_message(reading_warning: dict[str, int | list[str]]) -> str:
    msg = []

    if reading_warning.get("miss"):
        msg.append(
            f"Missing warning: missing {len(reading_warning['miss'])} file(s): {reading_warning['miss']}"
        )

    if reading_warning.get("res"):
        msg.append(
            "Resolution warning: resolution missmatch between mesh and at least one raster file. Nearest "
            "neighbour resampling algorithm has been used to match mesh resolution"
        )

    if reading_warning.get("overlap"):
        msg.append(
            "Overlap warning: overlap missmatch between mesh and at least one raster file. Cropping domain "
            "has been updated to the nearest overlapping cell"
        )

    if reading_warning.get("outofbound"):
        kind = "partially" if reading_warning["outofbound"] == 1 else "totally"
        msg.append(
            f"Out of bound warning: mesh is {kind} out of bound for at least one raster file. Out of bound "
            f"domain has been filled in with no data"
        )

    return "\n".join(msg)


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
            except Exception:
                raise ValueError(
                    f"Column header '{dat.columns[0]}' in the observed discharge file for catchment '{c}' "
                    f"is not a valid date"
                ) from None

            file_end_time = file_start_time + pd.Timedelta(seconds=setup.dt * (len(dat) - 1))
            start_diff = int((start_time - file_start_time).total_seconds() / setup.dt) + 1
            end_diff = int((end_time - file_start_time).total_seconds() / setup.dt) + 1

            # % Check if observed discharge file contains data for corresponding simulation period
            if start_diff > dat.index.max() or end_diff < 0:
                warnings.warn(
                    f"The provided observed discharge file for catchment '{c}' does not contain data for the "
                    f"selected simulation period ['{start_time}', '{end_time}']. The file covers the period "
                    f"['{file_start_time}', '{file_end_time}']",
                    stacklevel=2,
                )
            else:
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
        warnings.warn(f"Missing {len(miss)} observed discharge file(s): {miss}", stacklevel=2)


# TODO: Refactorize each read atmos
def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    if setup.prcp_format == "tif":
        files = _get_atmos_files(setup.prcp_directory, setup.prcp_format, setup.prcp_access, date_range)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading precipitation")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                reading_warning["miss"].append(date.strftime("%Y-%m-%d %H:%M"))
                if setup.sparse_storage:
                    matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
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
                matrix, warning = _read_windowed_raster(files[ind], mesh)
                matrix *= setup.prcp_conversion_factor
                reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

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

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to precipitation reading.\n{msg}", stacklevel=2)


def _read_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    if setup.pet_format == "tif":
        files = _get_atmos_files(
            setup.pet_directory,
            setup.pet_format,
            setup.pet_access,
            date_range,
            setup.daily_interannual_pet,
        )

        if setup.daily_interannual_pet:
            leap_year_days = pd.date_range(start="202001010000", end="202012310000", freq="1D")
            step_offset = int((date_range[0] - date_range[0].floor("D")).total_seconds() / setup.dt)
            nstep_per_day = int(86_400 / setup.dt)
            hourly_ratio = 3_600 / setup.dt

            if hourly_ratio >= 1:
                ratio = np.repeat(RATIO_PET_HOURLY, hourly_ratio) / hourly_ratio

            else:
                ratio = np.sum(RATIO_PET_HOURLY.reshape(-1, int(1 / hourly_ratio)), axis=1)

            matrix_dip = np.zeros(shape=(mesh.nrow, mesh.ncol, len(leap_year_days)), dtype=np.float32)
            missing_day = np.empty(shape=0, dtype=np.int32)

            for i, day in enumerate(tqdm(leap_year_days, desc="</> Reading daily interannual pet")):
                if day.day_of_year in date_range.day_of_year:
                    ind = _find_index_files_containing_date(files, day, setup.dt, setup.daily_interannual_pet)

                    if ind == -1:
                        reading_warning["miss"].append(day.strftime("%m-%d"))
                        missing_day = np.append(missing_day, day.day_of_year)

                    else:
                        matrix_dip[..., i], warning = _read_windowed_raster(files[ind], mesh)
                        matrix_dip[..., i] *= setup.pet_conversion_factor
                        reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

                        files = files[ind + 1 :]

            for i, date in enumerate(tqdm(date_range, desc="</> Disaggregating daily interannual pet")):
                day = date.day_of_year
                ratio_ind = (i + step_offset) % nstep_per_day

                if day in missing_day:
                    if setup.sparse_storage:
                        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
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
                ind = _find_index_files_containing_date(files, date, setup.dt, setup.daily_interannual_pet)

                if ind == -1:
                    reading_warning["miss"].append(date.strftime("%Y-%m-%d %H:%M"))
                    if setup.sparse_storage:
                        matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
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
                    matrix, warning = _read_windowed_raster(files[ind], mesh)
                    matrix *= setup.pet_conversion_factor
                    reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

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

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to potential evapotranspiration reading.\n{msg}", stacklevel=2)


def _read_snow(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    if setup.snow_format == "tif":
        files = _get_atmos_files(setup.snow_directory, setup.snow_format, setup.snow_access, date_range)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading snow")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                reading_warning["miss"].append(date.strftime("%Y-%m-%d %H:%M"))
                if setup.sparse_storage:
                    matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
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
                matrix, warning = _read_windowed_raster(files[ind], mesh)
                matrix *= setup.snow_conversion_factor
                reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

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

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to snow reading.\n{msg}", stacklevel=2)


def _read_temp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    if setup.temp_format == "tif":
        files = _get_atmos_files(setup.temp_directory, setup.temp_format, setup.temp_access, date_range)

        for i, date in enumerate(tqdm(date_range, desc="</> Reading temperature")):
            ind = _find_index_files_containing_date(files, date, setup.dt)

            if ind == -1:
                reading_warning["miss"].append(date.strftime("%Y-%m-%d %H:%M"))
                if setup.sparse_storage:
                    matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
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
                matrix, warning = _read_windowed_raster(files[ind], mesh)
                reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

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

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to temperature reading.\n{msg}", stacklevel=2)


def _read_descriptor(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    reading_warning = {"miss": [], "res": 0, "overlap": 0, "outofbound": 0}
    for i, name in enumerate(setup.descriptor_name):
        path = glob.glob(
            f"{setup.descriptor_directory}/**/{name}.tif*",
            recursive=True,
        )

        if len(path) == 0:
            reading_warning["miss"].append(name)

        else:
            desc, warning = _read_windowed_raster(path[0], mesh)
            reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})
            mask = desc != -99.0

            # % Check if descriptor contains only missing values
            if not np.any(mask):
                raise ValueError(
                    f"Invalid descriptor '{name}'. It contains only missing values on the selected domain"
                )

            # % Check if descriptor contains missing values on active cells
            if not np.all(mask[mesh.active_cell == 1]):
                raise ValueError(
                    f"Invalid descriptor '{name}'. It contains missing values on the active cells of the "
                    f"selected domain"
                )

            # % Check if descriptor is uniform
            low = np.min(desc, where=mask, initial=np.inf)
            upp = np.max(desc, where=mask, initial=-np.inf)

            if low == upp:
                raise ValueError(
                    f"Invalid descriptor '{name}'. It contains spatially uniform values on the selected "
                    f"domain"
                )
            # % Assign values
            input_data.physio_data.descriptor[..., i] = desc
            input_data.physio_data.l_descriptor[i] = low
            input_data.physio_data.u_descriptor[i] = upp

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to physiographic descriptor reading.\n{msg}", stacklevel=2)


def _read_imperviousness(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    # reading data
    imperviousness, warning = _read_windowed_raster(setup.imperviousness_file, mesh)

    # check values expected between 0 and 1 here
    mask = imperviousness != -99.0

    low = np.min(imperviousness, where=mask, initial=np.inf)
    upp = np.max(imperviousness, where=mask, initial=-np.inf)

    if low < 0 or upp > 1:
        raise ValueError(
            f"Invalid imperviousness. Values must be between 0 and 1 not between {low} and {upp}"
        )
    # write imperviousness rate into fortran variable
    input_data.physio_data.imperviousness = imperviousness

    msg = _get_reading_warning_message(warning)

    if msg:
        warnings.warn(f"Warning(s) linked to imperviousness reading.\n{msg}", stacklevel=2)

    if np.ma.is_masked(imperviousness):
        warnings.warn(f"{msg} \n Masked values are not applied", stacklevel=2)
