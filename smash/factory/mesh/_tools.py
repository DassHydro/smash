from __future__ import annotations

import numpy as np
from osgeo import osr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple
    from osgeo import gdal


def _xy_to_rowcol(
    x: float, y: float, xmin: float, ymax: float, xres: float, yres: float
) -> Tuple[int]:
    row = int((ymax - y) / yres)
    col = int((x - xmin) / xres)

    return row, col


def _rowcol_to_xy(
    row: int, col: int, xmin: float, ymax: float, xres: float, yres: float
) -> Tuple[int]:
    x = int(col * xres + xmin)
    y = int(ymax - row * yres)

    return x, y


def _trim_zeros_2d(array: np.ndarray, shift_value: bool = False) -> np.ndarray:
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
        return array, srow, erow, scol, ecol

    else:
        return array


def _get_array(
    flwdir_dataset: gdal.Dataset, bbox: np.ndarray | None = None
) -> np.ndarray:
    if bbox is not None:
        xmin, xmax, xres, ymin, ymax, yres = _get_transform(flwdir_dataset)

        col_off = int((bbox[0] - xmin) / xres)
        row_off = int((ymax - bbox[3]) / yres)
        ncol = int((bbox[1] - bbox[0]) / xres)
        nrow = int((bbox[3] - bbox[2]) / yres)

        flwdir = flwdir_dataset.GetRasterBand(1).ReadAsArray(
            col_off, row_off, ncol, nrow
        )

    else:
        flwdir = flwdir_dataset.GetRasterBand(1).ReadAsArray()

    return flwdir


def _get_transform(flwdir_dataset: gdal.Dataset) -> Tuple[float]:
    nrow = flwdir_dataset.RasterYSize
    ncol = flwdir_dataset.RasterXSize

    transform = flwdir_dataset.GetGeoTransform()

    xmin = transform[0]
    xres = transform[1]
    xmax = xmin + ncol * xres

    ymax = transform[3]
    yres = -transform[5]
    ymin = ymax - nrow * yres

    return (xmin, xmax, xres, ymin, ymax, yres)


def _get_srs(flwdir_dataset: gdal.Dataset, epsg: int) -> osr.SpatialReference:
    projection = flwdir_dataset.GetProjection()

    if projection:
        srs = osr.SpatialReference(wkt=projection)

    else:
        if epsg:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)

        else:
            raise ValueError(
                "Flow direction file does not contain spatial reference information. Can be specified with the 'epsg' argument"
            )

    return srs


def _get_path(flwacc: np.ndarray) -> np.ndarray:
    ind_path = np.unravel_index(np.argsort(flwacc, axis=None), flwacc.shape)

    path = np.zeros(shape=(2, flwacc.size), dtype=np.int32, order="F")

    path[0, :] = ind_path[0]
    path[1, :] = ind_path[1]

    return path
