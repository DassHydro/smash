from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rasterio.features

from smash.factory.mesh._libmesh import mw_mesh
from smash.factory.mesh._standardize import (
    _standardize_detect_sink_args,
    _standardize_generate_mesh_args,
)
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

    import geopandas as gpd

    from smash.util._typing import AlphaNumeric, FilePath, ListLike, Numeric

__all__ = ["detect_sink", "generate_mesh"]


def detect_sink(flwdir_path: FilePath, output_path: FilePath | None = None) -> np.ndarray:
    """
    Detect the sink cells in a flow direction file.

    Sinks in flow direction lead to numerical issues in the routing scheme and should be removed before
    generating the mesh.

    Parameters
    ----------
    flwdir_path : `str`
        Path to the flow directions file. The flow direction convention is the following:

        .. image:: ../../../_static/flwdir_convention.png
            :width: 100
            :align: center

    output_path : `str` or None, default None
        Path to the output file. If given, a raster file, in ``tif`` format, is written representing the sink
        cells.

    Returns
    -------
    sink : `numpy.ndarray`
        An array of shape *(nrow, ncol)*  containing a mask of sink cells.

        - ``0``: non-sink cell
        - ``1``: sink cell

    See Also
    --------
    smash.factory.generate_mesh : Automatic mesh generation.

    Examples
    --------
    >>> from smash.factory import load_dataset, detect_sink

    Retrieve a path to a flow direction file. A pre-processed file is available in the `smash` package (the
    path is updated for each user).

    >>> flwdir = load_dataset("flwdir")
    flwdir

    Detect the sink cells in the flow direction file.

    >>> sink = detect_sink(flwdir)
    >>> sink
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int32)

    ``sink`` is a mask of sink cells. The value ``1`` represents a sink cell and ``0`` a non-sink cell. The
    given flow direction file in the example does not contain any sink cell. ``sink`` is therefore a matrix
    of zeros.

    >>> np.all(sink == 0)
    np.True_

    In this example, we are going to add a false sink (2 cells) to show how they can be identified.

    >>> sink[10, 10:12] = 1
    >>> np.all(sink == 0), np.count_nonzero(sink == 1)
    (np.False_, 2)

    We can retrieve the indices of the sink cells.

    >>> idx = np.argwhere(sink == 1)
    array([[10, 10],
           [10, 11]])

    The flow direction file can be modified to remove sink cells.

    >>> with rasterio.open(flwdir) as ds_in:
    ...     flwdir_data = ds_in.read(1)
    ...     flwdir_data[sink == 1] = 0
    ...     with rasterio.open("flwdir_wo_sink.tif", "w", **ds_in.profile) as ds_out:
    ...         ds_out.write(flwdir_data, 1)

    .. note::
        Setting ``0`` to sink cells in the flow direction file is a way to remove them. However, it might not
        be the best way to handle sink cells.

    Finally, we can write the sink cells to a raster file by providing the output path.

    >>> detect_sink(flwdir, "./flwdir_sink.tif")

    The sink cells are written to the file ``flwdir_sink.tif`` and can be post-processed with a GIS software.
    """

    args = _standardize_detect_sink_args(flwdir_path, output_path)

    return _detect_sink(*args)


def _detect_sink(flwdir_dataset: rasterio.DatasetReader, output_path: str | None) -> np.ndarray:
    flwdir = _get_array(flwdir_dataset)

    sink = mw_mesh.detect_sink(flwdir)

    if output_path is not None:
        profile = flwdir_dataset.profile
        profile.update(dtype=np.uint8, count=1, nodata=np.iinfo(np.uint8).max)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(sink, 1)

    return sink


def generate_mesh(
    flwdir_path: FilePath,
    bbox: ListLike[float] | None = None,
    x: Numeric | ListLike[float] | None = None,
    y: Numeric | ListLike[float] | None = None,
    area: Numeric | ListLike[float] | None = None,
    code: str | ListLike[str] | None = None,
    shp_path: FilePath | None = None,
    max_depth: Numeric = 1,
    epsg: AlphaNumeric | None = None,
    area_error_th: Numeric | None = None,
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
            If not given, **x**, **y** and **area** must be provided.

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

        .. note::
            If not given, the default code is:

            ``['_c0', '_c1', ..., '_cn-1']`` with :math:`n`, the number of gauges (i.e. the size of **x**)

    shp_path : `str` or None, default None
        Path to the shapefile containing the contours of the catchment(s) to mesh.
        The shapefile must contain a ``code`` field to identify each catchment based on the **code** argument.
        The shapefile must be in the same projection as the flow direction file.
        For all the catchments in the shapefile, the ``contour-based`` method will be applied instead of the
        ``area-based`` one.

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
        defined in the flow directions file, it is not necessary to provide the value of
        the ``EPSG``. On the other hand, if the projection is not well defined in the flow directions file
        (i.e. in ``ASCII`` file), the **epsg** argument must be provided.

    area_error_th: `float` or None, default None
        The threshold of the relative error between the modeled area and the observed area beyond which the
        outlets and the catchment will be excluded from the final mesh. The relative error is computed as
        follows: :math:`error=(area_dln - area)/area`.
        For example, `area_error_th=0.2` means that all outlets where the surface error is higher than 20%
        will be excluded.

        .. note::
            If not given, the computation of the error on the area is ignored and all outlets are included
            in the mesh.

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
            If the projection unit is in degrees, the value of ``dx`` is approximated in meters.

        dy : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing Y cell size.
            If the projection unit is in degrees, the value of ``dy`` is approximated in meters.

        flwdir : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow direction.

        flwdst : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow distance.
            It corresponds to the distance in meters from each cell to the most downstream gauge.
            If there are multiple non-nested downstream gauges, the flow distance are computed for each gauge.

        flwacc : `numpy.ndarray`
            An array of shape *(nrow, ncol)* containing flow accumulation. The unit is the square meter.

        npar : `int`
            Number of partition. A partition delimits a set of independent cells on the drainage network.
            The first partition represents all the most upstream cells and the last partition the gauge(s).
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
    smash.factory.detect_sink : Detect the sink cells in a flow direction file.

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

    args = _standardize_generate_mesh_args(
        flwdir_path, bbox, x, y, area, code, shp_path, max_depth, epsg, area_error_th
    )

    return _generate_mesh(*args)


def _generate_mesh_from_xy(
    flwdir_dataset: rasterio.DatasetReader,
    bbox: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    code: np.ndarray,
    shp_dataset: gpd.GeoDataFrame | None,
    max_depth: int,
    epsg: int,
    area_error_th: float | None,
) -> dict:
    (xmin, _, xres, _, ymax, yres) = _get_transform(flwdir_dataset)

    crs = _get_crs(flwdir_dataset, epsg)
    epsg = crs.to_epsg()

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
    sink_dln = np.zeros(shape=x.shape, dtype=np.bool)
    mask_dln = np.zeros(shape=flwdir.shape, dtype=np.int32)

    deleted_catchment = []

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

        if shp_dataset is not None and code[ind] in shp_dataset["code"].values:
            transform = rasterio.transform.Affine(
                xres, 0, xmin + slice_win[1].start * xres, 0, -yres, ymax - slice_win[0].start * yres
            )
            geometry = shp_dataset.loc[shp_dataset["code"] == code[ind], "geometry"]
            mask = rasterio.features.rasterize(
                [(geom, 1) for geom in geometry], out_shape=flwdir_win.shape, transform=transform, fill=0
            )
            mask_dln_win, row_dln_win, col_dln_win, sink_dln[ind] = mw_mesh.catchment_dln_contour_based(
                flwdir_win, mask, row_win, col_win, max_depth
            )
        else:
            mask_dln_win, row_dln_win, col_dln_win, sink_dln[ind] = mw_mesh.catchment_dln_area_based(
                flwdir_win, dx_win, dy_win, row_win, col_win, area[ind], max_depth
            )

        row_dln[ind] = row_dln_win + slice_win[0].start  # % srow
        col_dln[ind] = col_dln_win + slice_win[1].start  # % scol

        area_dln[ind] = np.sum(mask_dln_win * dx_win * dy_win)

        if bbox is not None:
            flwdir_win_ind = np.ma.masked_array(flwdir_win, mask=(1 - mask_dln_win))
            flwdir_win_ind, slice_win_ind = _trim_mask_2d(flwdir_win_ind, slice_win=True)

            ymax_ind = ymax - (slice_win_ind[0].start + slice_win[0].start) * yres
            ymin_ind = ymax_ind - (slice_win_ind[0].stop - slice_win_ind[0].start) * yres

            xmin_ind = xmin + (slice_win_ind[1].start + slice_win[1].start) * xres
            xmax_ind = xmin_ind + (slice_win_ind[1].stop - slice_win_ind[1].start) * xres

            bbox_ind = np.array([xmin_ind, xmax_ind, ymin_ind, ymax_ind])

            if (
                bbox_ind[0] < bbox[0]
                or bbox_ind[1] > bbox[1]
                or bbox_ind[2] < bbox[2]
                or bbox_ind[3] > bbox[3]
            ):
                warnings.warn(
                    f"The extend of catchment {code[ind]} with bbox {bbox_ind} exceed"
                    f" the input bounding box {bbox}."
                    f" This catchment is removed from the mesh.",
                    stacklevel=2,
                )

                mask_dln_win = 0
                deleted_catchment.append(ind)

        if area_error_th is not None:
            if abs((area_dln[ind] - area[ind]) / area[ind]) > area_error_th:
                warnings.warn(
                    f"The error of the modeled area for catchment {code[ind]} exceed the"
                    " threshold {area_error_th}. This catchment is removed",
                    stacklevel=2,
                )
                mask_dln_win = 0
                deleted_catchment.append(ind)

        mask_dln[slice_win] = np.where(mask_dln_win == 1, 1, mask_dln[slice_win])

    row_dln = np.delete(row_dln, deleted_catchment)
    col_dln = np.delete(col_dln, deleted_catchment)
    area_dln = np.delete(area_dln, deleted_catchment)
    area = np.delete(area, deleted_catchment)
    x = np.delete(x, deleted_catchment)
    y = np.delete(y, deleted_catchment)
    code = np.delete(code, deleted_catchment)

    if np.any(sink_dln):
        warnings.warn(
            f"One or more sinks were detected when trying to delineate the catchment(s): "
            f"'{code[sink_dln == 1]}'. The catchment(s) might not be correctly delineated avoiding the sink "
            f"cells. See also the 'smash.factory.detect_sink' function "
            f"(https://smash.recover.inrae.fr/api_reference/sub-packages/smash/"
            f"smash.factory.detect_sink.html) to identify and correct sinks beforehand",
            stacklevel=2,
        )

    if bbox is None:
        flwdir = np.ma.masked_array(flwdir, mask=(1 - mask_dln))
        # slice flwdir according the border of the active domain
        flwdir, slice_win = _trim_mask_2d(flwdir, slice_win=True)

    else:
        col_off = int((bbox[0] - xmin) / xres)
        row_off = int((ymax - bbox[3]) / yres)
        ncol = int((bbox[1] - bbox[0]) / xres)
        nrow = int((bbox[3] - bbox[2]) / yres)

        slice_win = (slice(row_off, row_off + nrow), slice(col_off, col_off + ncol))

        flwdir = np.ma.masked_array(flwdir[slice_win], mask=(1 - mask_dln[slice_win]))

    dx = dx[slice_win]
    dy = dy[slice_win]

    ymax_shifted = ymax - slice_win[0].start * yres  # % srow
    xmin_shifted = xmin + slice_win[1].start * xres  # % scol

    row_dln = row_dln - slice_win[0].start  # % srow
    col_dln = col_dln - slice_win[1].start  # % scol

    flwacc, flwpar = mw_mesh.flow_accumulation_partition(flwdir, dx, dy)

    if x.size > 0:
        flwdst = mw_mesh.flow_distance(flwdir, dx, dy, row_dln, col_dln, area_dln)
    else:
        flwdst = np.zeros(shape=flwdir.shape)

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
        "epsg": epsg,
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
    epsg = crs.to_epsg()

    flwdir = _get_array(flwdir_dataset, bbox)

    # % Can close dataset
    flwdir_dataset.close()

    sink = mw_mesh.detect_sink(flwdir)
    # Check if sinks are detected on active cells and remove them
    if np.any(sink[flwdir > 0]):
        flwdir[sink == 1] = 0
        warnings.warn(
            "One or more sinks were detected in the given bounding box. The sink cells will be considered as "
            "non-active cells. See also the 'smash.factory.detect_sink' function "
            "(https://smash.recover.inrae.fr/api_reference/sub-packages/smash/"
            "smash.factory.detect_sink.html) to identify and correct sinks beforehand",
            stacklevel=2,
        )

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
        "epsg": epsg,
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
    shp_dataset: gpd.GeoDataFrame | None,
    max_depth: int,
    epsg: int | None,
    area_error_th: float | None,
) -> dict:
    if bbox is not None and x is None:
        return _generate_mesh_from_bbox(flwdir_dataset, bbox, epsg)
    else:
        return _generate_mesh_from_xy(
            flwdir_dataset, bbox, x, y, area, code, shp_dataset, max_depth, epsg, area_error_th
        )
