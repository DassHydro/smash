from __future__ import annotations

from smash._constant import D8_VALUE

from smash.factory.mesh._standardize import _standardize_generate_mesh_args
from smash.factory.mesh._tools import (
    _get_transform,
    _get_srs,
    _get_array,
    _get_path,
    _xy_to_rowcol,
    _trim_zeros_2d,
)
from smash.factory.mesh._flib_mesh import mw_mesh

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import FilePath, ListLike, Numeric, AlphaNumeric
    from osgeo import gdal

__all__ = ["generate_mesh"]


def generate_mesh(
    flwdir_path: FilePath,
    bbox: ListLike | None = None,
    x: Numeric | ListLike | None = None,
    y: Numeric | ListLike | None = None,
    area: Numeric | ListLike | None = None,
    code: str | ListLike | None = None,
    max_depth: Numeric = 1,
    epsg: AlphaNumeric | None = None,
) -> dict:
    """
    Automatic mesh generation.

    .. hint::
        See the (TODO: Fill) for more.

    Parameters
    ----------
    flwdir_path : str
        Path to the flow directions file. The file will be opened with `gdal <https://gdal.org/api/python/osgeo.gdal.html>`__.

    bbox : sequence or None, default None
        Bounding box with the following convention:

        ``(xmin, xmax, ymin, ymax)``.
        The bounding box values must respect the CRS of the flow directions file.

        .. note::
            If not given, **x**, **y** and **area** must be filled in.

    x : float, sequence or None, default None
        The x-coordinate(s) of the catchment outlet(s) to mesh.
        The **x** value(s) must respect the CRS of the flow directions file.
        The **x** size must be equal to **y** and area.

    y : float, sequence or None, default None
        The y-coordinate(s) of the catchment outlet(s) to mesh.
        The **y** value(s) must respect the CRS of the flow directions file.
        The **y** size must be equal to **x** and **area**.

    area : float, sequence or None, default None
        The area of the catchment(s) to mesh in m².
        The **area** size must be equal to **x** and **y**.

    code : str, sequence or None, default None
        The code of the catchment(s) to mesh.
        The **code** size must be equal to **x**, **y** and **area**.
        In case of bouding box meshing, the **code** argument is not used.

        .. note::
            If not given, the default code is:

            ``['_c0', '_c1', ..., '_cn-1']`` with :math:`n`, the number of gauges.

    max_depth : int, default 1
        The maximum depth accepted by the algorithm to find the catchment outlet.
        A **max_depth** of 1 means that the algorithm will search among the 2-length combinations in:

        ``(row - 1, row, row + 1, col - 1, col, col + 1)``, the coordinates that minimize the relative error between
        the given catchment area and the modeled catchment area calculated from the flow directions file.
        This can be generalized to :math:`n`.

        .. image:: ../../../_static/max_depth.png
            :align: center
            :width: 350

    epsg : str, int or None, default None
        The EPSG value of the flow directions file. By default, if the projection is well
        defined in the flow directions file. It is not necessary to provide the value of
        the EPSG. On the other hand, if the projection is not well defined in the flow directions file
        (i.e. in ASCII file). The **epsg** argument must be filled in.

    Returns
    -------
    mesh : dict
        A mesh dictionary that can be used to initialize the `smash.Model` object.

    See Also
    --------
    smash.Model : Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset, generate_mesh
    >>> flwdir = load_dataset("flwdir")
    >>> mesh = generate_mesh(
    ... flwdir,
    ... x=[840_261, 826_553, 828_269],
    ... y=[6_457_807, 6_467_115, 6_469_198],
    ... area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
    ... code=["V3524010", "V3515010", "V3517010"],
    ... )
    >>> mesh.keys()
    dict_keys(['xres', 'yres', 'xmin', 'ymax', 'nrow', 'ncol',
               'dx', 'dy', 'flwdir', 'flwdst', 'flwacc', 'nac',
               'active_cell', 'path', 'ng', 'gauge_pos', 'code',
               'area', 'area_dln'])
    """

    args = _standardize_generate_mesh_args(
        flwdir_path, bbox, x, y, area, code, max_depth, epsg
    )

    return _generate_mesh(*args)


def _generate_mesh_from_xy(
    flwdir_dataset: gdal.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    code: np.ndarray,
    max_depth: int,
    epsg: int,
) -> dict:
    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(flwdir_dataset)

    srs = _get_srs(flwdir_dataset, epsg)

    flwdir = _get_array(flwdir_dataset)

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

        area_dln[ind] = np.sum(mask_dln_imd * dx * dy)

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


def _generate_mesh_from_bbox(
    flwdir_dataset: gdal.Dataset, bbox: np.ndarray, epsg: int
) -> dict:
    (xmin, xmax, xres, ymin, ymax, yres) = _get_transform(flwdir_dataset)

    srs = _get_srs(flwdir_dataset, epsg)

    flwdir = _get_array(flwdir_dataset, bbox)

    if np.any(~np.isin(flwdir, D8_VALUE), where=(flwdir > 0)):
        raise ValueError(f"Flow direction data is invalid. Value must be in {D8_VALUE}")

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


def _generate_mesh(
    flwdir_dataset: gdal.Dataset,
    bbox: np.ndarray | None,
    x: np.ndarray | None,
    y: np.ndarray | None,
    area: np.ndarray | None,
    code: np.ndarray | None,
    max_depth: int,
    epsg: int | None,
) -> dict:
    if bbox is not None:
        return _generate_mesh_from_bbox(flwdir_dataset, bbox, epsg)
    else:
        return _generate_mesh_from_xy(flwdir_dataset, x, y, area, code, max_depth, epsg)
