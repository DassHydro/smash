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
    from smash.util._typing import FilePath


# Must be done char by char to take account of various format. Must be documented.
# Can't support all date formatter.
# Avantage: can work even file are not sorted with the date.
def _build_date_regex_pattern(date_pattern) -> str:
    # Supported date formatters
    datefmt = {"%Y": 4, "%m": 2, "%d": 2, "%H": 2, "%M": 2, "%S": 2}

    regex = ""
    i = 0
    while i < len(date_pattern):
        if date_pattern[i : i + 2] in datefmt:
            regex += r"\d{" + str(datefmt[date_pattern[i : i + 2]]) + "}"
            i = i + 2
        else:
            regex += r"\\" + date_pattern[i]
            i = i + 1

    return regex


def _split_date_occurence(date_pattern) -> str:
    re_match = re.search(r"\%[0-9]+$", date_pattern)
    if re_match is not None:
        pattern = date_pattern[0 : int(re_match.span()[0])]
        occurence = int(re_match.group()[1])
        return pattern, occurence
    else:
        return date_pattern, 0


# Test if files are sorted according the date
# Create a n samples of niter contigous date
# return True or false
def _sample_delta_time_in_file(files, date_pattern, dt):
    d_pattern, occurence = _split_date_occurence(date_pattern)
    regex_date = _build_date_regex_pattern(d_pattern)

    nsample = 5
    niter = 100
    list_delta_time = []
    list_date_block = []

    for i in range(0, len(files), int(len(files) / nsample)):
        for j in range(min(niter, len(files) - i - 1)):
            re_match_0 = re.findall(regex_date, os.path.basename(files[i + j]))
            re_match_1 = re.findall(regex_date, os.path.basename(files[i + j + 1]))

            if len(re_match_0) > 0 and len(re_match_1) > 0:
                date_match_0 = pd.to_datetime(re_match_0[occurence], format=d_pattern)
                date_match_1 = pd.to_datetime(re_match_1[occurence], format=d_pattern)
                delta_time = date_match_1 - date_match_0
                list_delta_time.append(delta_time.total_seconds())

                if j == 0:
                    list_date_block.append(date_match_0)

                if delta_time.total_seconds() < 0:
                    return False

    # Test if all sample date block are in ascendent order
    if len(list_date_block) > 1:
        for i in range(len(list_date_block) - 1):
            delta_time = list_date_block[i + 1] - list_date_block[i]

            if delta_time.total_seconds() < 0.0:
                return False

    if len(list_delta_time) > 0:
        if np.min(np.array(list_delta_time)) != dt:
            # sorted but not good time-step
            raise ValueError(
                "Precipitation files are sorted with the date pattern but not at the good time-step:"
                f" prcp time-step={list_delta_time[0]}, model time-step={dt}"
            )

        # Finally test if all date are in ascendent order
        if np.all(np.array(list_delta_time) > 0):
            return True
        else:
            return False

    else:
        raise ValueError(
            "Date Pattern was not found in the list of the precipitation files"
            f" while searching pattern={d_pattern} with corresponding regex={regex_date}"
            f" In filename {files[0]},"
        )


# Search for a pattern in a sorted list of files
# return the first position after or equal at date in files
def _fast_index_search_for_date(files, date, date_pattern):
    # print("search for",date)
    d_pattern, occurence = _split_date_occurence(date_pattern)
    regex_date = _build_date_regex_pattern(d_pattern)

    re_match = re.findall(regex_date, os.path.basename(files[-1]))
    if len(re_match) > 0:
        date_match = pd.to_datetime(re_match[occurence], format=d_pattern)

        # cas1
        if date > date_match:
            return len(files)

    else:
        raise ValueError(
            f"Date formatter {d_pattern} corresponding to regex {regex_date}"
            " not found in filename {os.path.basename(files[i])}"
        )

    re_match = re.findall(regex_date, os.path.basename(files[0]))
    if len(re_match) > 0:
        date_match = pd.to_datetime(re_match[occurence], format=d_pattern)

        # cas2
        if date < date_match:
            return 0

    else:
        raise ValueError(
            f"Date formatter {d_pattern} corresponding to regex {regex_date}"
            " not found in filename {os.path.basename(files[i])}"
        )

    # cas3
    pos = 0
    final_pos = 0
    move = 1
    previous_move = 0
    step = len(files) - 1
    nb_iter = 0
    while nb_iter < 100:
        re_match = re.findall(regex_date, os.path.basename(files[pos]))
        date_match = pd.to_datetime(re_match[occurence], format=d_pattern)

        # print(
        #             "At iteration "
        #             + str(nb_iter)
        #             +f" searching {date},"
        #             + "and match "
        #             + date_match.strftime("%Y-%m-%d %H:%M:%S")
        #             + f" at pos {pos}"
        #         )

        if date_match < date:
            pos = min(pos + step, len(files) - 1)
            move = 1
        elif date_match > date:
            pos = max(pos - step, 0)
            move = -1
        elif date_match == date:
            final_pos = pos
            break

        if step == 0:
            if date_match < date:
                final_pos = pos + 1
                break
            elif date_match > date:
                final_pos = pos
                break

        # print(f"-> Move to pos {pos} with step {step}")

        # reduce step only if we change search direction
        if previous_move != move:
            step = int(step / 2)

        nb_iter = nb_iter + 1
        previous_move = move

    # print(
    #             "At last iteration "
    #             + str(nb_iter)
    #             +f" searching {date},"
    #             + "and match "
    #             + date_match.strftime("%Y-%m-%d %H:%M:%S")
    #             + f" at pos {pos}"
    #         )

    # iter>maxlimit:
    if date_match < date:
        final_pos = pos
    elif date_match > date:
        # back to previous pos
        final_pos = pos - 2 * step

    return final_pos


# Getting  only files for the specified daterange
# work even files are not sorted according the date pattern
def _get_files_list_for_date_range(files, date_pattern, date_range):
    d_pattern, occurence = _split_date_occurence(date_pattern)
    regex_date = _build_date_regex_pattern(d_pattern)

    vec_date = []
    for i, f in enumerate(files):
        re_match = re.findall(regex_date, os.path.basename(f))

        if len(re_match) > 0:
            vec_date.append(re_match[occurence])
        # else:
        #     raise ValueError(
        #         f"Date formatter {d_pattern} corresponding to regex {regex_date}"
        #         " not found in filename {os.path.basename(f)}"
        #     )

    vec_date = pd.to_datetime(vec_date, format=d_pattern)
    vec_date = vec_date.strftime("%Y%m%d%H%M%S")

    # convert to numpy array
    np_lst_date = np.array(vec_date)
    np_lst_files = np.array(files)
    np_date_range = np.array(date_range.strftime("%Y%m%d%H%M%S").to_list())
    # sort but keep indexes
    sorted_indices = np_lst_date.argsort()
    # sort according previous indexes
    np_lst_date_sorted = np_lst_date[sorted_indices]

    # build the list of index only for the daterange
    index_list_for_daterange = []
    for i in range(len(np_date_range)):
        pos = np.where(np_lst_date_sorted == np_date_range[i])
        if len(pos[0]) > 0:
            index_list_for_daterange.append(pos[0][0])
        else:
            index_list_for_daterange.append(-1)

    # find the final list of files for daterange (sorted)
    final_list_files = []
    for index in index_list_for_daterange:
        if index >= 0:
            final_list_files.append(np_lst_files[sorted_indices[index]])
        else:
            final_list_files.append(-1)

    return final_list_files


# We suppose that the atmos file are sorted in ascendent order with the date
# (this a drawback of the method) ! Must be documented !
# Only used for daily internanuel pet
def _check_files_containing_date(files: str, date: pd.Timestamp, date_pattern: str) -> int:
    date_string = date.strftime(date_pattern)
    ind = -1
    for i, f in enumerate(files):
        # print(i,date_string,os.path.basename(f))
        if date_string in os.path.basename(f):
            ind = i

    return ind


# We suppose that the atmos file are sorted in ascendent order with the date
# (this a drawback of the method) ! Must be documented !
def _get_atmos_files(
    dir: str,
    fmt: str,
    access: str,
    dt: float,
    date_range: pd.DatetimeIndex,
    date_pattern: str,
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
        # Check if file are sorted with the date pattern
        test = _sample_delta_time_in_file(files, date_pattern, dt)

        # if sorted we quickly search index of date_start and date_end
        # if not sorted we parse all files et keep the dates
        if test:
            # file seems sorted
            pos_0 = _fast_index_search_for_date(files, date_range[0], date_pattern)
            pos_1 = _fast_index_search_for_date(files, date_range[-1], date_pattern)

            files = files[pos_0 : pos_1 + 1]

        else:
            print(
                "Warnings, precipitation filename are not sorted with date."
                " Reading precipitation may take more time."
            )

        # we build the file list according the date_range (sorted): we have 1 date <=>  1file
        final_list_files = _get_files_list_for_date_range(files, date_pattern, date_range)

        return final_list_files

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
        arr_xslice = slice(int(xs * 1 / xres_ratio), int(xe * 1 / xres_ratio))
        arr_yslice = slice(int(ys * 1 / yres_ratio), int(ye * 1 / yres_ratio))

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

    if reading_warning["got"]:
        if len(reading_warning["got"]) > 1:
            msg.append(
                f"Reading sucess: read {len(reading_warning['got'])} file(s): "
                f"{reading_warning['got'][0]}, ..., {reading_warning['got'][0-1]}"
            )
        else:
            msg.append(
                f"Reading sucess: read {len(reading_warning['got'])} file(s): " f"{reading_warning['got'][0]}"
            )

    if reading_warning["miss"]:
        msg.append(
            f"Missing warning: missing {len(reading_warning['miss'])} file(s): " f"{reading_warning['miss']}"
        )

    if reading_warning["res"]:
        msg.append(
            "Resolution warning: resolution missmatch between mesh and at least one raster file. Nearest "
            "neighbour resampling algorithm has been used to match mesh resolution"
        )

    if reading_warning["overlap"]:
        msg.append(
            "Overlap warning: overlap missmatch between mesh and at least one raster file. Cropping domain "
            "has been updated to the nearest overlapping cell"
        )

    if reading_warning["outofbound"]:
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


# TODO: Unique fun to read prcp, snow and temp
def _read_atmos_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT, atmos_data="prcp"):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"got": [], "miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    atmos_data_directory = getattr(setup, f"{atmos_data}_directory")
    atmos_data_format = getattr(setup, f"{atmos_data}_format")
    atmos_date_pattern = getattr(setup, f"{atmos_data}_date_pattern")
    prcp_access = getattr(setup, "prcp_access")
    prcp_conversion_factor = getattr(setup, "prcp_conversion_factor")
    sparse_storage = getattr(setup, "sparse_storage")
    dt = getattr(setup, "dt")

    sparse_matrix = None
    std_storage = None
    if sparse_storage:
        sparse_matrix = getattr(input_data.atmos_data, f"sparse_{atmos_data}")
    else:
        std_storage = getattr(input_data.atmos_data, f"{atmos_data}")

    if atmos_data_format == "tif":
        files = _get_atmos_files(
            atmos_data_directory, atmos_data_format, prcp_access, dt, date_range, atmos_date_pattern
        )

        for i, date in enumerate(tqdm(date_range, desc=f"</> Reading {atmos_data}")):
            atmos_file = files[i]

            if atmos_file == -1:
                reading_warning["miss"].append(
                    date.strftime("%Y-%m-%d %H:%M")
                    + f", (matching pattern {date.strftime(atmos_date_pattern)})"
                )
                if sparse_storage:
                    matrix = np.zeros(shape=(mesh.nrow, mesh.ncol), dtype=np.float32, order="F")
                    matrix.fill(np.float32(-99))
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(-99),
                        sparse_matrix[i],
                    )

                else:
                    std_storage[..., i] = np.float32(-99)

            else:
                reading_warning["got"].append(
                    date.strftime("%Y-%m-%d %H:%M") + f" ({os.path.basename(atmos_file)})"
                )

                matrix, warning = _read_windowed_raster(atmos_file, mesh)
                matrix *= prcp_conversion_factor
                reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

                if sparse_storage:
                    wrap_matrix_to_sparse_matrix(
                        mesh,
                        matrix,
                        np.float32(0),
                        sparse_matrix[i],
                    )

                else:
                    std_storage[..., i] = matrix

                # files = files[1:]

            if sparse_matrix:
                setattr(input_data.atmos_data, f"sparse_{atmos_data}", sparse_matrix)
            else:
                setattr(input_data.atmos_data, f"{atmos_data}", std_storage)

    # % WIP
    elif atmos_data_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to {atmos_data} reading.\n{msg}", stacklevel=2)


def _read_interannual_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    reading_warning = {"got": [], "miss": [], "res": 0, "overlap": 0, "outofbound": 0}

    if setup.pet_format == "tif":
        files = _get_atmos_files(
            setup.pet_directory,
            setup.pet_format,
            setup.pet_access,
            setup.dt,
            date_range,
            setup.prcp_date_pattern,
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

            list_date_in_daterange = date_range.strftime("%m%d")
            list_date_in_leap_year_days = leap_year_days.strftime("%m%d")

            for i, date in enumerate(tqdm(leap_year_days, desc="</> Reading daily interannual pet")):
                if list_date_in_leap_year_days[i] in list_date_in_daterange:
                    ind = _check_files_containing_date(files, date, "%m%d")

                    if ind == -1:
                        reading_warning["miss"].append(
                            date_range[i].strftime("%Y-%m-%d")
                            + f", (matching pattern {date.strftime('%m%d')})"
                        )
                        missing_day = np.append(missing_day, date)
                    else:
                        reading_warning["got"].append(
                            date.strftime("%Y-%m-%d") + f" ({os.path.basename(files[ind])})"
                        )
                        matrix_dip[..., i], warning = _read_windowed_raster(files[ind], mesh)
                        matrix_dip[..., i] *= setup.pet_conversion_factor
                        reading_warning.update({k: v for k, v in warning.items() if not reading_warning[k]})

            for i, date in enumerate(tqdm(date_range, desc="</> Disaggregating daily interannual pet")):
                ratio_ind = (i + step_offset) % nstep_per_day

                if date in missing_day:
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
                    index = list_date_in_leap_year_days.get_loc(list_date_in_daterange[i])
                    matrix = matrix_dip[..., index] * ratio[ratio_ind]
                    if setup.sparse_storage:
                        wrap_matrix_to_sparse_matrix(
                            mesh,
                            matrix,
                            np.float32(0),
                            input_data.atmos_data.sparse_pet[i],
                        )
                    else:
                        input_data.atmos_data.pet[..., i] = matrix
    # % WIP
    elif setup.pet_format == "nc":
        raise NotImplementedError("NetCDF format not implemented yet")

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to potential evapotranspiration reading.\n{msg}", stacklevel=2)


def _read_descriptor(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    reading_warning = {"got": [], "miss": [], "res": 0, "overlap": 0, "outofbound": 0}
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

            # % Check if descriptor contains only no data
            if not np.any(mask):
                raise ValueError(
                    f"Invalid descriptor '{name}'. It contains only missing values on the selected domain"
                )

            low = np.min(desc, where=mask, initial=np.inf)
            upp = np.max(desc, where=mask, initial=-np.inf)

            # % Check if descriptor is uniform
            if low == upp:
                raise ValueError(
                    f"Invalid descriptor '{name}'. Spatially uniform values on the selected domain. It must "
                    "be removed to perform optimization"
                )
            # % Assign values
            input_data.physio_data.descriptor[..., i] = desc
            input_data.physio_data.l_descriptor[i] = low
            input_data.physio_data.u_descriptor[i] = upp

    msg = _get_reading_warning_message(reading_warning)

    if msg:
        warnings.warn(f"Warning(s) linked to physiographic descriptor reading.\n{msg}", stacklevel=2)
