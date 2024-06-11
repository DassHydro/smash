from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smash.factory.mesh._libmesh import mw_mesh
from smash.factory.mesh._standardize import _standardize_generate_mesh_args
from smash.factory.mesh._tools import (
    _get_array,
    _get_catchment_slice_window,
    _get_crs,
    _get_transform,
    _trim_mask_2d,
    _xy_to_rowcol,
)

if TYPE_CHECKING:
    from typing import Any

    import rasterio

    from smash.util._typing import AlphaNumeric, FilePath, ListLike, Numeric

__all__ = ["generate_mesh"]


def generate_mesh(
    flwdir_path: FilePath,
    bbox: ListLike[float] | None = None,
    x: Numeric | ListLike[float] | None = None,
    y: Numeric | ListLike[float] | None = None,
    area: Numeric | ListLike[float] | None = None,
    code: str | ListLike[str] | None = None,
    max_depth: Numeric = 1,
    epsg: AlphaNumeric | None = None,
) -> dict[str, Any]:
    # % TODO FC: Add advanced user guide
    """
    Automatic mesh generation.

    Parameters
    ----------
    flwdir_path : `str`
        Path to the flow directions file. The flow direction convention is the following:

        .. image:: ../../../_static/flwdir_convention.png
            :width: 100
            :align: center

    bbox : `list[float, float, float, float]` or None, default None
        Bounding box with the following convention, ``(xmin, xmax, ymin, ymax)``.
        The bounding box values must respect the ``CRS`` of the flow direction file. If the given bounding
        box does not overlap the flow direction, the bounding box is padded to the nearest overlapping cell.

        .. note::
            If not given, **x**, **y** and **area** must be filled in.

    x : `float`, `list[float, ...]` or None, default None
        The x-coordinate(s) of the catchment outlet(s) to mesh.
        The **x** value(s) must respect the ``CRS`` of the flow directions file.
        The **x** size must be equal to **y** and **area** size.

    y : `float`, `list[float, ...]` or None, default None
        The y-coordinate(s) of the catchment outlet(s) to mesh.
        The **y** value(s) must respect the ``CRS`` of the flow directions file.
        The **y** size must be equal to **x** and **area** size.

    area : `float`, `list[float, ...]` or None, default None
        The area of the catchment(s) to mesh in ``m²``.
        The **area** size must be equal to **x** and **y** size.

    code : `str`, list[str, ...] or None, default None
        The code of the catchment(s) to mesh.
        The **code** size must be equal to **x**, **y** and **area**.
        In case of bouding box meshing, the **code** argument is not used.

        .. note::
            If not given, the default code is:

            ``['_c0', '_c1', ..., '_cn-1']`` with :math:`n`, the number of gauges (i.e. the size of **x**)

    max_depth : `int`, default 1
        The maximum depth accepted by the algorithm to find the catchment outlet.
        A **max_depth** of 1 means that the algorithm will search among the 2-length combinations in
        (``row - 1``, ``row``, ``row + 1``; ``col - 1``, ``col``, ``col + 1``), the coordinates that minimize
        the relative error between the given catchment area and the modeled catchment area calculated from the
        flow directions file. This can be generalized to :math:`n`.

        .. image:: ../../../_static/max_depth.png
            :align: center
            :width: 350

    epsg : `str`, `int` or None, default None
        The ``EPSG`` value of the flow directions file. By default, if the projection is well
        defined in the flow directions file. It is not necessary to provide the value of
        the ``EPSG``. On the other hand, if the projection is not well defined in the flow directions file
        (i.e. in ``ASCII`` file). The **epsg** argument must be filled in.

    Returns
    -------
    mesh : dict[str, Any]
        Model initialization mesh dictionary. The elements are:

        xres : `float`
            X cell size. The unit depends on the projection (meter or degree)

        yres : `float`
            Y cell size. The unit depends on the projection (meter or degree)

        xmin : `float`
            X minimum value. The unit depends on the projection (meter or degree)

        ymax : `float`
            Y maximum value. The unit depends on the projection (meter or degree)

        nrow : `int`
            Number of rows

        ncol : `int`
            Number of columns

        dx : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing X cell size.
            If the projection unit is in degree, the value of ``dx`` is approximated in meter.

        dy : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing Y cell size.
            If the projection unit is in degree, the value of ``dy`` is approximated in meter.

        flwdir : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow direction.

        flwdst : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow distance.
            It corresponds to the distance in meter from each cell to the most downstream gauge.
            If there are multiple non-nested downstream gauges, the flow distance are computed for each
            gauge.

        flwacc : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow accumulation. The unit is the square meter.

        npar : `int`
            Number of partition. A partition delimits a set of independent cells on the drainage network. The
            first partition represents all the most upstream cells and the last partition the gauge(s).
            It is possible to loop in parallel over all the cells in the same partition.

        ncpar : `numpy.ndarray`
            An array of shape *(npar,)* containing the number of cells per partition.

        cscpar : `numpy.ndarray`
            An array of shape *(npar,)* containing the cumulative sum of cells per partition.

        cpar_to_rowcol : `numpy.ndarray`
            An array of shape *(nrow*ncol, 2)* containing the mapping from the sorted partition cells to row,
            col indices.

        flwpar : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing the flow partitions. Values range from 1 to ``npar``.

        nac : `int`
            Number of active cells. A domain cell is considered active if it contributes to the discharge at
            the gauge if gauge coordinates are entered (i.e. **x**, **y**) else the whole bouding box is
            considered active.

        active_cell : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing a mask of active cells.

            - ``0``: Inactive cell
            - ``1``: Active cell

        ng : `int`
            Number of gauge.

        gauge_pos : `numpy.ndarray`
            An array of shape *(ng, 2)* containing the row, col indices of each gauge.

        code : `numpy.ndarray`
            An array of shape *(ng,)* containing the code of each gauge.
            The code is an alias enabling the user to identify the gauges in the mesh.

        area : `numpy.ndarray`
            An array of shape *(ng,)* containing the ``real`` observed draining area for each gauge.

        area_dln : `numpy.ndarray`
            An array of shape *(ng,)* containing the ``delineated`` draining area from the flow direction for
            each gauge. It is the relative error between ``area`` and ``area_dln`` which is minimized when
            trying to find the gauge coordinated on the flow direction file.

        .. note::
            The following variables, ``flwdst``, ``gauge_pos``, ``code``, ``area`` and ``area_dln``
            are only returned if gauge coordinates are entered (i.e. **x**, **y**) and not just a bounding box
            (i.e. **bbox**).

    See Also
    --------
    smash.Model : Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> from smash.factory import load_dataset, generate_mesh

    Retrieve a path to a flow direction file. A pre-processed file is available in the `smash` package (the
    path is updated for each user).

    >>> flwdir = load_dataset("flwdir")
    flwdir

    Generate a mesh from gauge coordinates. The following coordinates used are those of the ``Cance``
    dataset

    >>> mesh = generate_mesh(
        flwdir_path=flwdir,
        x=[840_261, 826_553, 828_269],
        y=[6_457_807, 6_467_115, 6_469_198],
        area=[381.7 * 1e6, 107 * 1e6, 25.3 * 1e6],
        code=["V3524010", "V3515010", "V3517010"],
    )
    >>> mesh.keys()
    dict_keys(['xres', 'yres', 'xmin', 'ymax', 'nrow', 'ncol', 'dx', 'dy', 'flwdir',
    'flwdst', 'flwacc', 'npar', 'ncpar', 'cscpar', 'cpar_to_rowcol', 'flwpar', 'nac',
    'active_cell', 'ng', 'gauge_pos', 'code', 'area', 'area_dln'])

    Access a specific value in the mesh dictionary

    >>> mesh["xres"], mesh["nac"], mesh["ng"]
    (1000.0, 383, 3)

    Generate a mesh from a bounding box ``(xmin, xmax, ymin, ymax)``. The following bounding box used
    correspond to the France boundaries

    >>> mesh = generate_mesh(
        flwdir_path=flwdir,
        bbox=[100_000, 1_250_000, 6_050_000, 7_125_000],
    )
    >>> mesh.keys()
    dict_keys(['xres', 'yres', 'xmin', 'ymax', 'nrow', 'ncol', 'dx', 'dy', 'flwdir',
    'flwacc', 'npar', 'ncpar', 'cscpar', 'cpar_to_rowcol', 'flwpar', 'nac', 'active_cell',
    'ng'])

    Access a specific value in the mesh dictionary

    >>> mesh["xres"], mesh["nac"], mesh["ng"]
    (1000.0, 906044, 0)
    """

    args = _standardize_generate_mesh_args(flwdir_path, bbox, x, y, area, code, max_depth, epsg)

    return _generate_mesh(*args)


def _generate_mesh_from_xy(
    flwdir_dataset: rasterio.DatasetReader,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    code: np.ndarray,
    max_depth: int,
    epsg: int,
) -> dict:
    (xmin, _, xres, _, ymax, yres) = _get_transform(flwdir_dataset)

    crs = _get_crs(flwdir_dataset, epsg)

    flwdir = _get_array(flwdir_dataset)

    # % Can close dataset
    flwdir_dataset.close()

    # % Accepting arrays for dx and dy in case of unstructured meshing
    if crs.units_factor[0].lower() == "degree":
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

        # % Take a window of flow directions. It avoids to fill and clean too large
        # % matrices. The boundaries of the window are obtained by assuming the
        # % worst spatial pattern of draining area (i.e. a rectangular catchment
        # % in x or y direction of 1 cell width).
        slice_win = _get_catchment_slice_window(
            *flwdir.shape, row, col, area[ind], dx[row, col], dy[row, col], max_depth
        )

        row_win = row - slice_win[0].start  # % srow
        col_win = col - slice_win[1].start  # % scol

        dx_win = dx[slice_win]
        dy_win = dy[slice_win]
        flwdir_win = flwdir[slice_win]

        mask_dln_win, row_dln_win, col_dln_win = mw_mesh.catchment_dln(
            flwdir_win, dx_win, dy_win, row_win, col_win, area[ind], max_depth
        )

        row_dln[ind] = row_dln_win + slice_win[0].start  # % srow
        col_dln[ind] = col_dln_win + slice_win[1].start  # % scol

        area_dln[ind] = np.sum(mask_dln_win * dx_win * dy_win)

        mask_dln[slice_win] = np.where(mask_dln_win == 1, 1, mask_dln[slice_win])

    flwdir = np.ma.masked_array(flwdir, mask=(1 - mask_dln))
    flwdir, slice_win = _trim_mask_2d(flwdir, slice_win=True)
    dx = dx[slice_win]
    dy = dy[slice_win]

    ymax_shifted = ymax - slice_win[0].start * yres  # % srow
    xmin_shifted = xmin + slice_win[1].start * xres  # % scol

    row_dln = row_dln - slice_win[0].start  # % srow
    col_dln = col_dln - slice_win[1].start  # % scol

    flwacc, flwpar = mw_mesh.flow_accumulation_partition(flwdir, dx, dy)

    flwdst = mw_mesh.flow_distance(flwdir, dx, dy, row_dln, col_dln, area_dln)

    flwdst = np.ma.masked_array(flwdst, mask=flwdir.mask)

    flwacc = np.ma.masked_array(flwacc, mask=flwdir.mask)

    npar = np.max(flwpar)
    ncpar, cscpar, cpar_to_rowcol = mw_mesh.flow_partition_variable(npar, flwpar)
    flwpar = np.ma.masked_array(flwpar, mask=flwdir.mask)

    nac = np.count_nonzero(~flwdir.mask)
    active_cell = 1 - flwdir.mask.astype(np.int32)

    ng = x.size
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
        "npar": npar,
        "ncpar": ncpar,
        "cscpar": cscpar,
        "cpar_to_rowcol": cpar_to_rowcol,
        "flwpar": flwpar,
        "nac": nac,
        "active_cell": active_cell,
        "ng": ng,
        "gauge_pos": gauge_pos,
        "code": code,
        "area": area,
        "area_dln": area_dln,
    }

    return mesh


def _generate_mesh_from_bbox(flwdir_dataset: rasterio.DatasetReader, bbox: np.ndarray, epsg: int) -> dict:
    (_, _, xres, _, ymax, yres) = _get_transform(flwdir_dataset)

    crs = _get_crs(flwdir_dataset, epsg)

    flwdir = _get_array(flwdir_dataset, bbox)

    # % Can close dataset
    flwdir_dataset.close()

    flwdir = np.ma.masked_array(flwdir, mask=(flwdir < 1))

    # % Accepting arrays for dx and dy in case of unstructured meshing
    if crs.units_factor[0].lower() == "degree":
        dx, dy = mw_mesh.latlon_dxdy(*flwdir.shape, xres, yres, ymax)

    else:
        dx = np.zeros(shape=flwdir.shape, dtype=np.float32) + xres
        dy = np.zeros(shape=flwdir.shape, dtype=np.float32) + yres

    flwacc, flwpar = mw_mesh.flow_accumulation_partition(flwdir, dx, dy)

    flwacc = np.ma.masked_array(flwacc, mask=flwdir.mask)

    npar = np.max(flwpar)
    ncpar, cscpar, cpar_to_rowcol = mw_mesh.flow_partition_variable(npar, flwpar)
    flwpar = np.ma.masked_array(flwpar, mask=flwdir.mask)

    nac = np.count_nonzero(~flwdir.mask)
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
        "npar": npar,
        "ncpar": ncpar,
        "cscpar": cscpar,
        "cpar_to_rowcol": cpar_to_rowcol,
        "flwpar": flwpar,
        "nac": nac,
        "active_cell": active_cell,
        "ng": 0,
    }

    return mesh


def _generate_mesh(
    flwdir_dataset: rasterio.DatasetReader,
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
