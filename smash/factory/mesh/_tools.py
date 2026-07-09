from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

if TYPE_CHECKING:
    from collections.abc import Generator

    from smash.fcore._mwd_setup import SetupDT


def _xy_to_rowcol(x: float, y: float, xmin: float, ymax: float, xres: float, yres: float) -> tuple:
    row = int((ymax - y) / yres)
    col = int((x - xmin) / xres)

    return row, col


def _rowcol_to_xy(row: int, col: int, xmin: float, ymax: float, xres: float, yres: float) -> tuple:
    x = int(col * xres + xmin)
    y = int(ymax - row * yres)

    return x, y


def _get_catchment_slice_window(
    nrow: int,
    ncol: int,
    row: int,
    col: int,
    area: float,
    dx: float,
    dy: float,
    max_depth: int,
) -> tuple[slice, slice]:
    n = np.ceil(area / (dx * dy)).astype(np.int32)
    srow = np.maximum(0, row - max_depth - n)
    erow = np.minimum(nrow - 1, row + max_depth + n)
    scol = np.maximum(0, col - max_depth - n)
    ecol = np.minimum(ncol - 1, col + max_depth + n)

    return (slice(srow, erow), slice(scol, ecol))


def _trim_mask_2d(
    array: np.ndarray, slice_win: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[slice, slice]]:
    for ax in [0, 1]:
        mask = ~(array.mask).all(axis=ax)

        inv_mask = mask[::-1]

        # Force cast from np.int64 to np.int32
        start_ind = np.argmax(mask).astype(np.int32)
        end_ind = len(inv_mask) - np.argmax(inv_mask).astype(np.int32)

        if ax == 0:
            slice_col = slice(start_ind, end_ind)
            array = array[:, slice_col]

        else:
            slice_row = slice(start_ind, end_ind)
            array = array[slice_row, :]

    if slice_win:
        return array, (slice_row, slice_col)

    else:
        return array


def _get_array(flwdir_dataset: rasterio.DatasetReader, bbox: np.ndarray | None = None) -> np.ndarray:
    if bbox is None:
        flwdir = flwdir_dataset.read(1)

    else:
        xmin, _, xres, _, ymax, yres = _get_transform(flwdir_dataset)

        col_off = int((bbox[0] - xmin) / xres)
        row_off = int((ymax - bbox[3]) / yres)
        ncol = int((bbox[1] - bbox[0]) / xres)
        nrow = int((bbox[3] - bbox[2]) / yres)

        flwdir = flwdir_dataset.read(1, window=rasterio.windows.Window(col_off, row_off, ncol, nrow))

    return flwdir


def _get_transform(flwdir_dataset: rasterio.DatasetReader) -> tuple:
    nrow = flwdir_dataset.height
    ncol = flwdir_dataset.width

    transform = flwdir_dataset.get_transform()

    xmin = transform[0]
    xres = transform[1]
    xmax = xmin + ncol * xres

    ymax = transform[3]
    yres = -transform[5]
    ymin = ymax - nrow * yres

    return (xmin, xmax, xres, ymin, ymax, yres)


def _get_crs(flwdir_dataset: rasterio.DatasetReader, epsg: int) -> rasterio.CRS:
    crs = flwdir_dataset.crs
    if not crs:
        if epsg:
            crs = rasterio.CRS.from_epsg(epsg)
        else:
            raise ValueError(
                "Flow direction file does not contain spatial reference information. Can be specified with "
                "the 'epsg' argument"
            )
    return crs


def _get_shp_dataset_sql_clauses(code: np.ndarray) -> Generator[str, None, None]:
    # Split the code array into chunks of 4000 elements to avoid SQL querie limit
    if code.size == 1:
        yield f"code = '{code[0]}'"
    else:
        chunk_size = 4000
        for i in range(0, code.size, chunk_size):
            yield f"code IN {tuple(code[i : i + chunk_size].tolist())}"


def _load_shp_dataset(shp_path: str, code: np.ndarray) -> gpd.GeoDataFrame:
    # Fast reading of only one line to retrieve the shapefile columns
    if "code" not in gpd.read_file(shp_path, max_features=1).columns:
        raise ValueError("Shapefile must contain at least a 'code' column")

    # Load the shapefile dataset with only the 'code' and 'geometry' columns with a subset based on the given
    # code.
    shp_dataset = pd.concat(
        [
            gpd.read_file(shp_path, columns=["code", "geometry"], where=clause)
            for clause in _get_shp_dataset_sql_clauses(code)
        ]
    )
    shp_dataset.reset_index(drop=True, inplace=True)

    missing_code = [str(c) for c in np.setdiff1d(code, shp_dataset["code"])]

    if missing_code:
        warnings.warn(
            f"Missing shapefile geometry for the code: {missing_code}. The "
            f"area-based method will be used instead of the contour-based one",
            stacklevel=2,
        )

    return shp_dataset


def _trans_dict_hydraulics_discontinuities_to_discontinuitiesDT(
    setup: SetupDT, hydraulics_discontinuities: dict
):
    discontinuities_type = []
    discontinuities_rank = []
    discontinuities_name = []
    rk_dam = 0
    rk_input = 0
    nmax_val = 0

    for name, hd in hydraulics_discontinuities.items():
        if hd["hd_type"] == "dam":
            rk_dam = +1
            rank = rk_dam
        else:
            rk_input = +1
            rank = rk_input

        discontinuities_name.append(name)
        discontinuities_type.append(hd["hd_type"])
        discontinuities_rank.append(rank)

        nmax_val = max(len(hd["rules"]["rel_hv"]), nmax_val)

    nd = len(hydraulics_discontinuities.keys())
    ntime_step = setup.ntime_step

    dam_hv = np.ndarray(shape=(nd, nmax_val))
    dam_hq = np.ndarray(shape=(nd, nmax_val))
    input_q = np.ndarray(shape=(nd, nmax_val))

    dam_hv = -99.0
    dam_hq = -99.0
    input_q = -99.0

    for i, name in enumerate(discontinuities_name):
        if discontinuities_type[i] == "dam":
            j = hydraulics_discontinuities[name]["rules"]["rel_hv"].shape[0]
            dam_hv[i, 0:j, :] = hydraulics_discontinuities[name]["rules"]["rel_hv"][:, :]
            dam_hq[i, 0:j, :] = hydraulics_discontinuities[name]["rules"]["rel_hq"][:, :]

        elif discontinuities_type[i] == "input_q":
            input_q[i, 0:ntime_step, :] = hydraulics_discontinuities[name]["rules"]["input_q"][:, :]

        else:
            continue

    return {
        "hydraulics_discontinuities": {
            "discontinuities_name": discontinuities_name,
            "discontinuities_type": discontinuities_type,
            "discontinuities_rank": discontinuities_rank,
            "dam_hv": dam_hv,
            "dam_hq": dam_hq,
            "input_q": input_q,
        }
    }
