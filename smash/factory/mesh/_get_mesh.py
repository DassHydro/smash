from __future__ import annotations

from smash.factory.mesh._mesh import mw_mesh

from smash.factory.mesh._standardize import _standardize_bbox, _standardize_gauge
from smash.factory.mesh._tools import (
    _get_transform,
    _get_srs,
    _get_array,
    _get_path,
    _xy_to_rowcol,
    _trim_zeros_2d,
)

import numpy as np


def _get_mesh_from_xy(ds_flwdir, x, y, area, code, max_depth, epsg):
    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(ds_flwdir)

    srs = _get_srs(ds_flwdir, epsg)

    (x, y, area, code) = _standardize_gauge(ds_flwdir, x, y, area, code)

    flwdir = _get_array(ds_flwdir)

    # % Accepting arrays for dx and dy in case of unstructured meshing
    if srs.GetAttrValue("UNIT") == "degree":
        dx, dy = mw_mesh.latlon_dxdy(*flwdir.shape, xres, yres, ymax)

    else:
        dx = np.zeros(shape=flwdir.shape, dtype=np.float32) + xres
        dy = np.zeros(shape=flwdir.shape, dtype=np.float32) + yres

    row_dln = np.zeros(shape=x.shape, dtype=np.int32)
    col_dln = np.zeros(shape=x.shape, dtype=np.int32)
    area_dln = np.zeros(shape=x.shape, dtype=np.float32)
    mask_dln = np.zeros(shape=flwdir.shape, dtype=np.int32)

    for ind in range(x.size):
        row, col = _xy_to_rowcol(x[ind], y[ind], xmin, ymax, xres, yres)

        mask_dln_imd, row_dln[ind], col_dln[ind] = mw_mesh.catchment_dln(
            flwdir, dx, dy, row, col, area[ind], max_depth
        )

        area_dln[ind] = np.count_nonzero(mask_dln_imd == 1) * (xres * yres)

        mask_dln = np.where(mask_dln_imd == 1, 1, mask_dln)

    flwdir = np.ma.masked_array(flwdir, mask=(1 - mask_dln))
    flwdir, srow, erow, scol, ecol = _trim_zeros_2d(flwdir, shift_value=True)
    dx = dx[srow:erow, scol:ecol]
    dy = dy[srow:erow, scol:ecol]

    xmin_shifted = xmin + scol * xres
    ymax_shifted = ymax - srow * yres

    row_dln = row_dln - srow
    col_dln = col_dln - scol

    flwacc = mw_mesh.flow_accumulation(flwdir, dx, dy)

    flwdst = mw_mesh.flow_distance(flwdir, dx, dy, row_dln, col_dln, area_dln)

    path = _get_path(flwacc)

    flwdst = np.ma.masked_array(flwdst, mask=flwdir.mask)

    flwacc = np.ma.masked_array(flwacc, mask=flwdir.mask)

    active_cell = 1 - flwdir.mask.astype(np.int32)

    gauge_pos = np.column_stack((row_dln, col_dln))

    mesh = {
        "xres": xres,
        "yres": yres,
        "xmin": xmin_shifted,
        "ymax": ymax_shifted,
        "nrow": flwdir.shape[0],
        "ncol": flwdir.shape[1],
        "dx": dx,
        "dy": dy,
        "flwdir": flwdir,
        "flwdst": flwdst,
        "flwacc": flwacc,
        "nac": np.count_nonzero(active_cell),
        "active_cell": active_cell,
        "path": path,
        "ng": x.size,
        "gauge_pos": gauge_pos,
        "code": code,
        "area": area,
        "area_dln": area_dln,
    }

    return mesh


def _get_mesh_from_bbox(ds_flwdir, bbox, epsg):
    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(ds_flwdir)

    srs = _get_srs(ds_flwdir, epsg)

    bbox = _standardize_bbox(ds_flwdir, bbox)

    flwdir = _get_array(ds_flwdir, bbox)

    flwdir = np.ma.masked_array(flwdir, mask=(flwdir < 1))

    # % Accepting arrays for dx and dy in case of unstructured meshing
    if srs.GetAttrValue("UNIT") == "degree":
        dx, dy = mw_mesh.latlon_dxdy(*flwdir.shape, xres, yres, ymax)

    else:
        dx = np.zeros(shape=flwdir.shape, dtype=np.float32) + xres
        dy = np.zeros(shape=flwdir.shape, dtype=np.float32) + yres

    flwacc = mw_mesh.flow_accumulation(flwdir, dx, dy)

    path = _get_path(flwacc)

    flwacc = np.ma.masked_array(flwacc, mask=flwdir.mask)

    active_cell = 1 - flwdir.mask.astype(np.int32)

    mesh = {
        "xres": xres,
        "yres": yres,
        "xmin": bbox[0],
        "ymax": bbox[3],
        "nrow": flwdir.shape[0],
        "ncol": flwdir.shape[1],
        "dx": dx,
        "dy": dy,
        "flwdir": flwdir,
        "flwacc": flwacc,
        "nac": np.count_nonzero(active_cell),
        "active_cell": active_cell,
        "ng": 0,
        "path": path,
    }

    return mesh
