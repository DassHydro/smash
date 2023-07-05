from __future__ import annotations

from smash._constant import D8_VALUE

import numpy as np
from osgeo import osr


def _xy_to_rowcol(x, y, xmin, ymax, xres, yres):
    row = int((ymax - y) / yres)
    col = int((x - xmin) / xres)

    return row, col


def _rowcol_to_xy(row, col, xmin, ymax, xres, yres):
    x = int(col * xres + xmin)
    y = int(ymax - row * yres)

    return x, y


def _trim_zeros_2d(array, shift_value=False):
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
        # ~ return array, scol, ecol, srow, erow
        return array, srow, erow, scol, ecol

    else:
        return array


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
    nrow = ds_flwdir.RasterYSize
    ncol = ds_flwdir.RasterXSize

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
                "Flow direction file does not contain spatial reference information. Can be specified with the 'epsg' argument"
            )

    return srs


def _get_path(flwacc):
    ind_path = np.unravel_index(np.argsort(flwacc, axis=None), flwacc.shape)

    path = np.zeros(shape=(2, flwacc.size), dtype=np.int32, order="F")

    path[0, :] = ind_path[0]
    path[1, :] = ind_path[1]

    return path
