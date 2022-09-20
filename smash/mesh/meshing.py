from __future__ import annotations

from smash.mesh._meshing import mw_meshing

import errno
import os
import warnings
import numpy as np
from osgeo import gdal, osr

METER_TO_DEGREE = 0.000008512223965693407
DEGREE_TO_METER = 117478.11195173833
D8_VALUE = np.arange(1, 9)

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


def _get_array(ds_flwdir, bbox=None):

    if bbox:

        xmin, xmax, xres, ymin, ymax, yres = _get_transform(ds_flwdir)

        col_off = int((bbox[0] - xmin) / xres)
        row_off = int((ymax - bbox[3]) / yres)
        ncol = int((bbox[1] - bbox[0]) / xres)
        nrow = int((bbox[3] - bbox[2]) / yres)

        flwdir = ds_flwdir.GetRasterBand(1).ReadAsArray(col_off, row_off, ncol, nrow)

    else:

        flwdir = ds_flwdir.GetRasterBand(1).ReadAsArray()

    if np.any(~np.isin(flwdir, D8_VALUE), where=(flwdir > 0)):

        raise ValueError(f"Flow direction data is invalid. Value must be in {D8_VALUE}")

    return flwdir


def _get_transform(ds_flwdir):

    ncol = ds_flwdir.RasterXSize
    nrow = ds_flwdir.RasterYSize

    transform = ds_flwdir.GetGeoTransform()

    xmin = transform[0]
    xres = transform[1]
    xmax = xmin + ncol * xres

    ymax = transform[3]
    yres = -transform[5]
    ymin = ymax - nrow * yres

    return xmin, xmax, xres, ymin, ymax, yres


def _get_srs(ds_flwdir, epsg):

    projection = ds_flwdir.GetProjection()

    if projection:

        srs = osr.SpatialReference(wkt=projection)

    else:

        if epsg:

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg))

        else:

            raise ValueError(
                "Flow direction file does not contain 'CRS' information. Can be specified with the 'epsg' argument"
            )

    return srs


def _standardize_gauge(ds_flwdir, x, y, area, code):

    x = np.array(x, dtype=np.float32, ndmin=1)
    y = np.array(y, dtype=np.float32, ndmin=1)
    area = np.array(area, dtype=np.float32, ndmin=1)

    if (x.size != y.size) or (y.size != area.size):

        raise ValueError(
            f"Inconsistent size for 'x' ({x.size}), 'y' ({y.size}) and 'area' ({area.size})"
        )

    xmin, xmax, xres, ymin, ymax, yres = _get_transform(ds_flwdir)

    if np.any((x < xmin) | (x > xmax)):

        raise ValueError(f"'x' {x} value(s) out of flow directions bounds {xmin, xmax}")

    if np.any((y < ymin) | (y > ymax)):

        raise ValueError(f"'y' {y} value(s) out of flow directions bounds {ymin, ymax}")

    if np.any(area < 0):

        raise ValueError(f"Negative 'area' value(s) {area}")

    #% Setting _c0, _c1, ... _cN as default codes
    if code is None:

        code = np.zeros(shape=(20, x.size), dtype="uint8") + np.uint8(32)

        for i in range(x.size):

            code[0:3, i] = [ord(l) for l in ["_", "c", str(i)]]

    elif isinstance(code, (str, list)):

        code_imd = np.array(code, ndmin=1)

        #% Only check x (y and area already check above)
        if code_imd.size != x.size:

            raise ValueError(
                f"Inconsistent size for 'code' ({code_imd.size}) and 'x' ({x.size})"
            )

        else:

            code = np.zeros(shape=(20, x.size), dtype="uint8") + np.uint8(32)

            for i in range(x.size):

                code[0 : len(code_imd[i]), i] = [ord(l) for l in code_imd[i]]

    return x, y, area, code


def _standardize_bbox(ds_flwdir, bbox):

    #% Bounding Box (xmin, ymin, xmax, ymax)

    if isinstance(bbox, (list, tuple, set)):

        bbox = list(bbox)

        if len(bbox) != 4:

            raise ValueError(f"'bbox argument must be of size 4 ({len(bbox)})")

        if bbox[0] > bbox[1]:

            raise ValueError(
                f"'bbox' xmin ({bbox[0]}) is greater than xmax ({bbox[1]})"
            )

        if bbox[2] > bbox[3]:

            raise ValueError(
                f"'bbox' ymin ({bbox[2]}) is greater than ymax ({bbox[3]})"
            )

        xmin, xmax, xres, ymin, ymax, yres = _get_transform(ds_flwdir)

        if bbox[0] < xmin:

            warnings.warn(
                f"'bbox' xmin ({bbox[0]}) is out of flow directions bound ({xmin}). 'bbox' is update according to flow directions bound"
            )

            bbox[0] = xmin

        if bbox[1] > xmax:

            warnings.warn(
                f"'bbox' xmax ({bbox[1]}) is out of flow directions bound ({xmax}). 'bbox' is update according to flow directions bound"
            )

            bbox[1] = xmax

        if bbox[2] < ymin:

            warnings.warn(
                f"'bbox' ymin ({bbox[2]}) is out of flow directions bound ({ymin}). 'bbox' is update according to flow directions bound"
            )

            bbox[2] = ymin

        if bbox[3] > ymax:

            warnings.warn(
                f"'bbox' ymax ({bbox[3]}) is out of flow directions bound ({ymax}). 'bbox' is update according to flow directions bound"
            )

            bbox[3] = ymax

    else:

        raise TypeError("'bbox' argument must be list-like object")

    return bbox


def _get_path(drained_area):

    ind_path = np.unravel_index(np.argsort(drained_area, axis=None), drained_area.shape)

    path = np.zeros(shape=(2, drained_area.size), dtype=np.int32, order="F")

    #% Transform from Python to FORTRAN index
    path[0, :] = ind_path[0] + 1
    path[1, :] = ind_path[1] + 1

    return path


def _get_mesh_from_xy(ds_flwdir, x, y, area, code, max_depth, epsg):

    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(ds_flwdir)

    srs = _get_srs(ds_flwdir, epsg)

    (x, y, area, code) = _standardize_gauge(ds_flwdir, x, y, area, code)

    flwdir = _get_array(ds_flwdir)

    #% Convert (approximate) area from square meter to square degree
    if srs.GetAttrValue("UNIT") == "degree":

        area = area * METER_TO_DEGREE**2
        dx = xres * DEGREE_TO_METER

    else:

        dx = xres

    col_ol = np.zeros(shape=x.shape, dtype=np.int32)
    row_ol = np.zeros(shape=x.shape, dtype=np.int32)
    area_ol = np.zeros(shape=x.shape, dtype=np.float32)
    mask_dln = np.zeros(shape=flwdir.shape, dtype=np.int32)

    for ind in range(x.size):

        col, row = _xy_to_colrow(x[ind], y[ind], xmin, ymax, xres, yres)

        mask_dln_imd, col_ol[ind], row_ol[ind] = mw_meshing.catchment_dln(
            flwdir, col, row, xres, yres, area[ind], max_depth
        )

        if srs.GetAttrValue("UNIT") == "degree":

            area_ol[ind] = (
                np.count_nonzero(mask_dln_imd == 1)
                * (xres * yres)
                * DEGREE_TO_METER**2
            )

        else:

            area_ol[ind] = np.count_nonzero(mask_dln_imd == 1) * (xres * yres)

        mask_dln = np.where(mask_dln_imd == 1, 1, mask_dln)

    flwdir = np.ma.masked_array(flwdir, mask=(1 - mask_dln))

    flwdir, scol, ecol, srow, erow = _trim_zeros_2D(flwdir, shift_value=True)
    mask_dln = _trim_zeros_2D(mask_dln)

    xmin_shifted = xmin + scol * xres
    ymax_shifted = ymax - srow * yres

    col_ol = col_ol - scol
    row_ol = row_ol - srow
    
    flwdst = mw_meshing.flow_distance(flwdir, col_ol, row_ol, area_ol, dx)

    drained_area = mw_meshing.drained_area(flwdir)

    path = _get_path(drained_area)
    
    flwdst = np.ma.masked_array(flwdst, mask=(1 - mask_dln))
    
    drained_area = np.ma.masked_array(drained_area, mask=(1 - mask_dln))

    active_cell = mask_dln.astype(np.int32)

    #% Transform from Python to FORTRAN index
    gauge_pos = np.vstack((row_ol + 1, col_ol + 1))

    mesh = {
        "dx": dx,
        "nrow": flwdir.shape[0],
        "ncol": flwdir.shape[1],
        "ng": x.size,
        "nac": np.count_nonzero(active_cell),
        "xmin": xmin_shifted,
        "ymax": ymax_shifted,
        "flwdir": flwdir,
        "flwdst": flwdst,
        "drained_area": drained_area,
        "path": path,
        "gauge_pos": gauge_pos,
        "code": code,
        "area": area_ol,
        "active_cell": active_cell,
    }

    return mesh


def _get_mesh_from_bbox(ds_flwdir, bbox, epsg):

    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(ds_flwdir)

    srs = _get_srs(ds_flwdir, epsg)

    bbox = _standardize_bbox(ds_flwdir, bbox)

    flwdir = _get_array(ds_flwdir, bbox)

    flwdir = np.ma.masked_array(flwdir, mask=(flwdir < 1))

    if srs.GetAttrValue("UNIT") == "degree":

        dx = xres * DEGREE_TO_METER

    else:

        dx = xres

    drained_area = mw_meshing.drained_area(flwdir)

    path = _get_path(drained_area)

    drained_area = np.ma.masked_array(drained_area, mask=(flwdir < 1))

    active_cell = np.zeros(shape=flwdir.shape, dtype=np.int32)

    active_cell = np.where(flwdir > 0, 1, active_cell)

    mesh = {
        "dx": dx,
        "nrow": flwdir.shape[0],
        "ncol": flwdir.shape[1],
        "ng": 0,
        "nac": np.count_nonzero(active_cell),
        "xmin": bbox[0],
        "ymax": bbox[3],
        "flwdir": flwdir,
        "drained_area": drained_area,
        "path": path,
        "active_cell": active_cell,
    }

    return mesh


def generate_mesh(
    path: str,
    bbox: (None, tuple(float)) = None,
    x: (None, float, list[float]) = None,
    y: (None, float, list[float]) = None,
    area: (None, float, list[float]) = None,
    code: (None, str, list[str]) = None,
    max_depth: int = 1,
    epsg: (None, str, int) = None,
) -> dict:

    if os.path.isfile(path):

        ds_flwdir = gdal.Open(path)

    else:

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if bbox:

        return _get_mesh_from_bbox(ds_flwdir, bbox, epsg)

    else:

        if not x or not y or not area:

            raise ValueError(
                "'bbox' argument or 'x', 'y' and 'area' arguments must be defined"
            )

        return _get_mesh_from_xy(ds_flwdir, x, y, area, code, max_depth, epsg)
