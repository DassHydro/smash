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

def _standardize_generate_meshing(x, y, area, name):
    
    x_array = np.array(x, dtype=np.float32, ndmin=1)
    y_array = np.array(y, dtype=np.float32, ndmin=1)
    area_array = np.array(area, dtype=np.float32, ndmin=1)

    # TODO Add check for size
    
    name_array = np.zeros(shape=(20, len(x_array)), dtype="uint8")
    
    if name is None:
        
        for i in range(len(x_array)):
            
            name_ord = [ord(l) for l in ["_", "c", str(i)]]
            
            name_array[0:3, i] = name_ord
            name_array[3:, i] = 32
            
    else:
        
        for i in range(len(x_array)):
            
            name_array[0:len(name[i]), i] = [ord(l) for l in name[i]]
            name_array[len(name[i]):, i] = 32
            
    return x_array, y_array, area_array, name_array

# TODO update for multi-gauge
def generate_meshing(
    path: str,
    x: (float, list[float]),
    y: (float, list[float]),
    area: (float, list[float]),
    dkind: list[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    max_depth: int = 1,
    code: (None, str) = None,
):

    if os.path.isfile(path):
        ds_flow = rasterio.open(path)

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        
    (x, y, area, code) = _standardize_generate_meshing(x, y, area, code)
    
    flow = ds_flow.read(1)

    ncol = ds_flow.width
    nrow = ds_flow.height

    transform = ds_flow.transform

    xll = transform[2]
    yll = transform[5]
    xres = transform[0]
    yres = -transform[4]
    
    col_otl = np.zeros(shape=x.shape, dtype=np.int32)
    row_otl = np.zeros(shape=x.shape, dtype=np.int32)
    area_otl = np.zeros(shape=x.shape, dtype=np.float32)
    global_mask_dln = np.zeros(shape=flow.shape, dtype=np.int32)

    for ind in range(len(x)):
        
        col, row = _xy_to_colrow(x[ind], y[ind], xll, yll, xres, yres)
        
        mask_dln, col_otl[ind], row_otl[ind] = _meshing.catchment_dln(flow, col, row, xres, yres, area[ind], dkind, max_depth)
        
        area_otl[ind] = np.count_nonzero(mask_dln == 1)
        
        global_mask_dln = np.where(mask_dln == 1, 1, global_mask_dln)
        
    flow = np.ma.masked_array(flow, mask=(1 - global_mask_dln))
    
    flow, scol, ecol, srow, erow = _trim_zeros_2D(flow, shift_value=True)
    global_mask_dln = _trim_zeros_2D(global_mask_dln)
    
    xll_shifted = xll + scol * xres
    yll_shifted = yll - erow * yres
    
    col_otl = col_otl - scol
    row_otl = row_otl - srow
    
    drained_area = _meshing.drained_area(flow)
    
    drained_area = np.ma.masked_array(drained_area, mask=(1 - global_mask_dln))
    
    global_active_cell = global_mask_dln.astype(np.int32)

    gauge_pos = np.vstack((row_otl + 1, col_otl + 1))
    
    mesh = {
        "ng": len(x),
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
        "area": area_otl,
    }

    return mesh
