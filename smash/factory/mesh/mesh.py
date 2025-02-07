from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import rasterio

import rasterio.features

import geopandas as gpd

from smash.factory.mesh._libmesh import mw_mesh
from smash.factory.mesh._standardize import (
    _standardize_detect_sink_args,
    _standardize_generate_hy1d_mesh_args,
    _standardize_generate_mesh_args,
    _standardize_generate_rr_mesh_args,
)
from smash.factory.mesh._tools import (
    _get_array,
    _get_catchment_slice_window,
    _get_cross_sections_and_segments,
    _get_crs,
    _get_river_line,
    _get_transform,
    _trim_mask_2d,
    _xy_to_rowcol,
)

if TYPE_CHECKING:
    from typing import Any

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


def _detect_sink(flwdir_path: str, output_path: str | None) -> np.ndarray:
    with rasterio.open(flwdir_path) as flwdir_dataset:
        flwdir = _get_array(flwdir_dataset)

    sink = mw_mesh.detect_sink(flwdir)

    if output_path is not None:
        profile = flwdir_dataset.profile
        profile.update(dtype=np.uint8, count=1, nodata=np.iinfo(np.uint8).max)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(sink, 1)

    return sink


def _generate_rr_mesh_from_xy(
    flwdir_path: str,
    x: np.ndarray,
    y: np.ndarray,
    area: np.ndarray,
    code: np.ndarray,
    shp_dataset: gpd.GeoDataFrame | None,
    max_depth: int,
    epsg: int,
) -> dict:
    with rasterio.open(flwdir_path) as flwdir_dataset:
        (xmin, _, xres, _, ymax, yres) = _get_transform(flwdir_dataset)

        crs = _get_crs(flwdir_dataset, epsg)

        flwdir = _get_array(flwdir_dataset)

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

        mask_dln[slice_win] = np.where(mask_dln_win == 1, 1, mask_dln[slice_win])

    if np.any(sink_dln):
        warnings.warn(
            f"One or more sinks were detected when trying to delineate the catchment(s): "
            f"'{code[sink_dln == 1]}'. The catchment(s) might not be correctly delineated avoiding the sink "
            f"cells. See also the 'smash.factory.detect_sink' function "
            f"(https://smash.recover.inrae.fr/api_reference/sub-packages/smash/"
            f"smash.factory.detect_sink.html) to identify and correct sinks beforehand",
            stacklevel=2,
        )

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
        "epsg": crs.to_epsg(),
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


def _generate_rr_mesh_from_bbox(flwdir_path: str, bbox: np.ndarray, epsg: int) -> dict:
    with rasterio.open(flwdir_path) as flwdir_dataset:
        (_, _, xres, _, ymax, yres) = _get_transform(flwdir_dataset)

        crs = _get_crs(flwdir_dataset, epsg)

        flwdir = _get_array(flwdir_dataset, bbox)

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
        "epsg": crs.to_epsg(),
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


def _generate_rr_mesh(
    flwdir_path: FilePath,
    bbox: ListLike[float] | None = None,
    x: Numeric | ListLike[float] | None = None,
    y: Numeric | ListLike[float] | None = None,
    area: Numeric | ListLike[float] | None = None,
    code: str | ListLike[str] | None = None,
    shp_path: FilePath | None = None,
    max_depth: Numeric = 1,
    epsg: AlphaNumeric | None = None,
) -> dict[str, Any]:
    flwdir_path, bbox, x, y, area, code, shp_dataset, max_depth, epsg = _standardize_generate_rr_mesh_args(
        flwdir_path, bbox, x, y, area, code, shp_path, max_depth, epsg
    )

    if bbox is not None:
        return _generate_rr_mesh_from_bbox(flwdir_path, bbox, epsg)
    else:
        return _generate_rr_mesh_from_xy(flwdir_path, x, y, area, code, shp_dataset, max_depth, epsg)

def _generate_hy1d_mesh(
    rr_mesh: dict[str, Any], 
    river_line_path: FilePath, 
    bbox: ListLike[float] | None = None,
    w_coef_a: float = 1.54,
    w_coef_b: float = 0.44
) -> dict[str, Any]:
    (river_line_path,) = _standardize_generate_hy1d_mesh_args(river_line_path)

    if bbox is not None:
        print("debug: reading river line without preprocessing.")
        river_line = gpd.read_file(river_line_path)
    else:
        print("debug: using _get_river_line for river line preprocessing.")
        river_line = _get_river_line(river_line_path, rr_mesh)

    cross_sections, segments = _get_cross_sections_and_segments(river_line, rr_mesh,w_coef_a, w_coef_b)

    return {
        "ncs": len(cross_sections),
        "cross_sections": cross_sections,
        "nseg": len(segments),
        "segments": segments,
    }


def _generate_mesh(
    rr_options: dict[str, Any],
    hy1d_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rr_mesh = _generate_rr_mesh(**rr_options)

    if hy1d_options:
        rr_mesh.update(_generate_hy1d_mesh(rr_mesh, **hy1d_options))
    else:
        rr_mesh.update({"ncs": 0, "nseg": 0})
    return rr_mesh


def generate_mesh(
    rr_options: dict[str, Any],
    hy1d_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # % TODO: Add documentation here
    args = _standardize_generate_mesh_args(rr_options, hy1d_options)

    return _generate_mesh(*args)
