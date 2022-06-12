from __future__ import annotations

import errno
import os
import numpy as np
import rasterio

from . import _meshing

__all__ = ["generate_meshing"]


def _xy_to_colrow(x, y, xll, yll, xres, yres):

    col = int((x - xll) / xres)
    row = int((yll - y) / yres)

    return col, row


def _colrow_to_xy(col, row, xll, yll, xres, yres):

    x = int(col * xres + xll)
    y = int(yll - row * yres)

    return x, y


def _trim_zeros_2D(array, shift_value=False):

    for ax in [0, 1]:

        mask = ~(array == 0).all(axis=ax)

        inv_mask = mask[::-1]

        start_ind = np.argmax(mask)

        end_ind = len(inv_mask) - np.argmax(inv_mask)

        if ax == 0:

            scol, ecol = start_ind, end_ind
            array = array[:, start_ind:end_ind]

        else:

            srow, erow = start_ind, end_ind
            array = array[start_ind:end_ind, :]

    if shift_value:
        return array, scol, ecol, srow, erow

    else:
        return array


def _array_to_ascii(array, path, xll, yll, cellsize, no_data_val):

    array = np.copy(array)
    array[np.isnan(array)] = no_data_val
    header = (
        f"NCOLS {array.shape[1]} \nNROWS {array.shape[0]}"
        f"\nXLLCENTER {xll} \nYLLCENTER {yll} \nCELLSIZE {cellsize} \nNODATA_value {no_data_val}\n"
    )

    with open(path, "w") as f:

        f.write(header)
        np.savetxt(f, array, "%5.2f")

# TODO update for multi-gauge
def generate_meshing(
    path: str,
    x: float,
    y: float,
    area: float,
    dkind: list[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    max_depth: int = 1,
    name: (None, str) = None,
):

    if os.path.isfile(path):

        ds_flow = rasterio.open(path)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    flow = ds_flow.read(1)

    ncol = ds_flow.width
    nrow = ds_flow.height

    transform = ds_flow.transform

    xll = transform[2]
    yll = transform[5]
    xres = transform[0]
    yres = -transform[4]

    col, row = _xy_to_colrow(x, y, xll, yll, xres, yres)

    logical_dln, col_otl, row_otl = _meshing.catchment_dln(
        flow, col, row, xres, yres, area, dkind, max_depth
    )

    x_otl, y_otl = _colrow_to_xy(col_otl, row_otl, xll, yll, xres, yres)

    flow = np.ma.masked_array(flow, mask=(1 - logical_dln))

    flow, scol, ecol, srow, erow = _trim_zeros_2D(flow, shift_value=True)
    logical_dln = _trim_zeros_2D(logical_dln)

    xll_shifted = xll + scol * xres
    yll_shifted = yll - erow * yres

    col_otl = col_otl - scol
    row_otl = row_otl - srow

    # Check dkind flow dir in case of dkind != [1, 2, 3, 4, 5, 6, 7, 8]
    drained_area = _meshing.drained_area(flow)

    drained_area = np.ma.masked_array(drained_area, mask=(1 - logical_dln))
    
    global_active_cell = logical_dln.astype(np.int8)
    
    code = np.zeros(shape=20, dtype="uint8")

    if isinstance(name, str):

        code[0 : len(name)] = [ord(l) for l in name]

        code[len(name) :] = 32

    else:

        code[0] = ord("_")
        code[1] = ord("c")
        code[2:] = 32

    code = code.reshape(20, 1)

    gauge_pos = np.zeros(shape=(2, 1), dtype=np.int32)
    gauge_pos[:, 0] = np.array([row_otl + 1, col_otl + 1])

    mesh = {
        "ng": 1,
        "nrow": flow.shape[0],
        "ncol": flow.shape[1],
        "xll": xll_shifted,
        "yll": yll_shifted,
        "flow": flow,
        "drained_area": drained_area,
        "global_active_cell": global_active_cell,
        "local_active_cell": global_active_cell.copy(),
        "gauge_pos": gauge_pos,
        "code": code,
        "area": np.amax(drained_area),
    }

    return mesh
