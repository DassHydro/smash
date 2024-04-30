from __future__ import annotations

import numpy as np
import rasterio


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

        start_ind = np.argmax(mask)

        end_ind = len(inv_mask) - np.argmax(inv_mask)

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
