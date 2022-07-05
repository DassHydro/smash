from __future__ import annotations

from smash.mesh import _meshing

import errno
import os
import numpy as np
from osgeo import gdal, osr

__all__ = ["generate_mesh"]


def _xy_to_colrow(x, y, xmin, ymax, xres, yres):

    col = int((x - xmin) / xres)
    row = int((ymax - y) / yres)

    return col, row


def _colrow_to_xy(col, row, xmin, ymax, xres, yres):

    x = int(col * xres + xmin)
    y = int(ymax - row * yres)

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


def _array_to_ascii(array, path, xmin, ymin, cellsize, no_data_val):

    array = np.copy(array)
    array[np.isnan(array)] = no_data_val
    header = (
        f"NCOLS {array.shape[1]} \nNROWS {array.shape[0]}"
        f"\nXLLCENTER {xmin} \nYLLCENTER {ymin} \nCELLSIZE {cellsize} \nNODATA_value {no_data_val}\n"
    )

    with open(path, "w") as f:

        f.write(header)
        np.savetxt(f, array, "%5.2f")


def _standardize_generate_mesh(x, y, area, code):

    x_array = np.array(x, dtype=np.float32, ndmin=1)
    y_array = np.array(y, dtype=np.float32, ndmin=1)
    area_array = np.array(area, dtype=np.float32, ndmin=1)

    # TODO Add check for size

    code_array = np.zeros(shape=(20, len(x_array)), dtype="uint8")

    if code is None:

        for i in range(len(x_array)):

            code_ord = [ord(l) for l in ["_", "c", str(i)]]

            code_array[0:3, i] = code_ord
            code_array[3:, i] = 32

    elif isinstance(code, (str, list)):

        code = np.array(code, ndmin=1)

        for i in range(len(x_array)):

            code_array[0 : len(code[i]), i] = [ord(l) for l in code[i]]
            code_array[len(code[i]) :, i] = 32

    return x_array, y_array, area_array, code_array


def generate_mesh(
    path: str,
    x: (float, list[float]),
    y: (float, list[float]),
    area: (float, list[float]),
    max_depth: int = 1,
    code: (None, str) = None,
) -> dict:

    if os.path.isfile(path):
        ds_flow = gdal.Open(path)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    (x, y, area, code) = _standardize_generate_mesh(x, y, area, code)

    flow = ds_flow.GetRasterBand(1).ReadAsArray()

    ncol = ds_flow.RasterXSize
    nrow = ds_flow.RasterYSize

    transform = ds_flow.GetGeoTransform()

    projection = ds_flow.GetProjection()
    srs = osr.SpatialReference(wkt=projection)

    xmin = transform[0]
    ymax = transform[3]
    xres = transform[1]
    yres = -transform[5]

    #% Approximate area from square meter to square degree
    if srs.GetAttrValue("geogcs") == "WGS 84":

        area = area * (xres * yres)

    col_otl = np.zeros(shape=x.shape, dtype=np.int32)
    row_otl = np.zeros(shape=x.shape, dtype=np.int32)
    area_otl = np.zeros(shape=x.shape, dtype=np.float32)
    global_mask_dln = np.zeros(shape=flow.shape, dtype=np.int32)

    for ind in range(len(x)):

        col, row = _xy_to_colrow(x[ind], y[ind], xmin, ymax, xres, yres)

        mask_dln, col_otl[ind], row_otl[ind] = _meshing.catchment_dln(
            flow, col, row, xres, yres, area[ind], max_depth
        )

        area_otl[ind] = np.count_nonzero(mask_dln == 1)

        global_mask_dln = np.where(mask_dln == 1, 1, global_mask_dln)

    flow = np.ma.masked_array(flow, mask=(1 - global_mask_dln))

    flow, scol, ecol, srow, erow = _trim_zeros_2D(flow, shift_value=True)
    global_mask_dln = _trim_zeros_2D(global_mask_dln)

    xmin_shifted = xmin + scol * xres
    ymax_shifted = ymax - srow * yres

    col_otl = col_otl - scol
    row_otl = row_otl - srow

    drained_area = _meshing.drained_area(flow)

    drained_area = np.ma.masked_array(drained_area, mask=(1 - global_mask_dln))

    ind_path = np.unravel_index(np.argsort(drained_area, axis=None), drained_area.shape)

    path = np.zeros(shape=(2, flow.shape[0] * flow.shape[1]), dtype=np.int32, order="F")

    #% Transform from Python to FORTRAN index
    path[0, :] = ind_path[0] + 1
    path[1, :] = ind_path[1] + 1

    global_active_cell = global_mask_dln.astype(np.int32)

    #% Transform from Python to FORTRAN index
    gauge_pos = np.vstack((row_otl + 1, col_otl + 1))

    mesh = {
        "nrow": flow.shape[0],
        "ncol": flow.shape[1],
        "ng": len(x),
        "nac": np.count_nonzero(global_active_cell),
        "xmin": xmin_shifted,
        "ymax": ymax_shifted,
        "flow": flow,
        "drained_area": drained_area,
        "path": path,
        "gauge_pos": gauge_pos,
        "code": code,
        "area": area_otl,
        "global_active_cell": global_active_cell,
        "local_active_cell": global_active_cell.copy(),
    }

    return mesh
