from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from smash.factory.mesh._libmesh import mw_mesh

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import rasterio.features
from shapely.geometry import Point, LineString, Polygon
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.features import shapes
from pyflwdir.gis_utils import degree_metres_x, degree_metres_y
from pyflwdir.core import confluence_indices


if TYPE_CHECKING:
    from collections.abc import Generator


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

        # Force cast from np.int64 to np.int32
        start_ind = np.argmax(mask).astype(np.int32)
        end_ind = len(inv_mask) - np.argmax(inv_mask).astype(np.int32)

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


def _get_shp_dataset_sql_clauses(code: np.ndarray) -> Generator[str, None, None]:
    # Split the code array into chunks of 4000 elements to avoid SQL querie limit
    if code.size == 1:
        yield f"code = '{code[0]}'"
    else:
        chunk_size = 4000
        for i in range(0, code.size, chunk_size):
            yield f"code IN {tuple(code[i:i + chunk_size].tolist())}"


def _load_shp_dataset(shp_path: str, code: np.ndarray) -> gpd.GeoDataFrame:
    # Fast reading of only one line to retrieve the shapefile columns
    if "code" not in gpd.read_file(shp_path, max_features=1).columns:
        raise ValueError("Shapefile must contain at least a 'code' column")

    # Load the shapefile dataset with only the 'code' and 'geometry' columns with a subset based on the given
    # code.
    shp_dataset = pd.concat(
        [
            gpd.read_file(shp_path, columns=["code", "geometry"], where=clause)
            for clause in _get_shp_dataset_sql_clauses(code)
        ]
    )
    shp_dataset.reset_index(drop=True, inplace=True)

    missing_code = [str(c) for c in np.setdiff1d(code, shp_dataset["code"])]

    if missing_code:
        warnings.warn(
            f"Missing shapefile geometry for the code: {missing_code}. The "
            f"area-based method will be used instead of the contour-based one",
            stacklevel=2,
        )

    return shp_dataset


def _switch_flwdir_convention(flwdir: np.ndarray, convention: str) -> np.ndarray:
    if convention == "smash-d8":
        conditions = [
            (flwdir == 1),
            (flwdir == 2),
            (flwdir == 3),
            (flwdir == 4),
            (flwdir == 5),
            (flwdir == 6),
            (flwdir == 7),
            (flwdir == 8),
            (flwdir == 0),  # SMASH nodata
        ]
        values = [64, 128, 1, 2, 4, 8, 16, 32, 247]  # 0 -> 247 for D8 nodata
        flwdir_converted = np.select(conditions, values, 247).astype(np.uint8)

    elif convention == "d8-smash":
        conditions = [
            (flwdir == 64),
            (flwdir == 128),
            (flwdir == 1),
            (flwdir == 2),
            (flwdir == 4),
            (flwdir == 8),
            (flwdir == 16),
            (flwdir == 32),
            (flwdir == 247),  # D8 nodata
        ]
        values = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # 247 -> 0 for SMASH nodata
        flwdir_converted = np.select(conditions, values, 0).astype(np.int32)

    return flwdir_converted


def _d8_idx(idx0: int, shape: tuple[int, int]) -> np.ndarray:
    """Returns linear indices of eight neighboring cells"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idxs_lst = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:  # skip pit -> return empty array
                continue
            r_us, c_us = r + dr, c + dc
            if r_us >= 0 and r_us < nrow and c_us >= 0 and c_us < ncol:  # check bounds
                idx = r_us * ncol + c_us
                idxs_lst.append(idx)
    return np.array(idxs_lst)


def _get_main_river_line(river_line: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def _mark_connected_lines(
        row: gpd.GeoDataFrame, river_line: gpd.GeoDataFrame, current_group: int
    ) -> None:
        connected = river_line[river_line.geometry.touches(row.geometry)]
        for idx, row_connected in connected.iterrows():
            if river_line.at[idx, "group"] == -1:
                river_line.at[idx, "group"] = current_group
                _mark_connected_lines(row_connected, river_line, current_group)

    river_line["group"] = -1
    current_group = 0

    for idx, row in river_line.iterrows():
        if river_line.at[idx, "group"] == -1:
            river_line.at[idx, "group"] = current_group
            _mark_connected_lines(row, river_line, current_group)
            current_group += 1

    main_group = river_line.groupby("group").size().idxmax()

    main_river_line = river_line[river_line["group"] == main_group].copy()
    main_river_line.drop(columns=["group"], inplace=True)

    return main_river_line


def _get_river_line(river_line_path: str, rr_mesh: dict[str, Any]) -> gpd.GeoDataFrame:
    transform = rasterio.transform.from_origin(
        rr_mesh["xmin"], rr_mesh["ymax"], rr_mesh["xres"], rr_mesh["yres"]
    )
    shapes = rasterio.features.shapes(
        rr_mesh["active_cell"], mask=rr_mesh["active_cell"].astype(bool), transform=transform
    )

    lriver_line = []
    for s in shapes:
        rl_mask = Polygon(s[0]["coordinates"][0])
        rl = gpd.read_file(river_line_path, mask=rl_mask, columns=["geometry"])
        rl = _get_main_river_line(rl)
        rl["geometry"] = rl.geometry.explode()
        lriver_line.append(rl)

    return gpd.GeoDataFrame(pd.concat(lriver_line, ignore_index=True))


def _get_cross_sections_and_segments(
    river_line: gpd.GeoDataFrame, 
    rr_mesh: dict[str, Any],
    w_coef_a: float,
    w_coef_b: float
) -> np.ndarray:
    transform = rasterio.transform.from_origin(
        rr_mesh["xmin"], rr_mesh["ymax"], rr_mesh["xres"], rr_mesh["yres"]
    )
    latlon = rr_mesh["epsg"] == 4326

    flwdir = pyflwdir.from_array(
        _switch_flwdir_convention(rr_mesh["flwdir"], "smash-d8"),
        ftype="d8",
        transform=transform,
        latlon=latlon,
        cache=True,
    )

    selected_path = {}
    # Iterate over each line segment in the river GeoDataFrame
    for rl in river_line["geometry"]:
        # Handle MultiLineString geometries when _get_river_line is bypassed
        if rl.geom_type == 'LineString':
            # Extract line coordinates
            line_coords = list(rl.coords)
        else:
            line_coords = [coord for linestring in rl.geoms for coord in linestring.coords] # MultiLineString

        # Determine the starting point of the current line segment
        start_point = Point(line_coords[0][0], line_coords[0][1])

        # Transform the starting point coordinates to raster cell coordinates
        start_cell_col, start_cell_row = map(int, ~flwdir.transform * (start_point.x, start_point.y))

        # Get the linear index of the start cell
        start_cell_idx = start_cell_row * flwdir.shape[1] + start_cell_col

        # Get the indices of the eight neighboring cells of the start cell
        neighbors_cells_idxs = _d8_idx(start_cell_idx, flwdir.shape)
        neighbors_cells_idxs = neighbors_cells_idxs[np.isin(neighbors_cells_idxs, flwdir.idxs_seq)]

        # Define flow path source cells as the start cell and its eight neighbors
        source_cells = np.concatenate(([start_cell_idx], neighbors_cells_idxs))

        # Compute flow paths starting from the source cells
        flwpaths, _ = flwdir.path(idxs=source_cells)

        flwpaths_analysis = {}
        for fp in flwpaths:
            count = sum(
                sum(1 for x, y, *_ in line_coords if x_min <= x < x_max and y_min <= y < y_max)
                for cell in fp
                for row, col in [np.unravel_index(cell, flwdir.shape)]
                for x_min, y_max in [flwdir.transform * (col, row)]
                for x_max, y_min in [flwdir.transform * (col + 1, row + 1)]
            )
            flwpaths_analysis[fp[0]] = count

        # Select the start cell index of the flow path with the highest number of line segment points
        best_cell_idx = max(flwpaths_analysis, key=lambda k: flwpaths_analysis[k])

        # Store the selected path of the best start cell index
        selected_path[best_cell_idx] = flwpaths[np.where(source_cells == best_cell_idx)[0][0]]

    # Compute the flow path
    flwpath = np.unique(np.concatenate(list(selected_path.values())))
    flwpath = flwpath[np.isin(flwpath, flwdir.idxs_seq)]
    flwpath_rowcol = np.unravel_index(flwpath, flwdir.shape)
    flwpath_mask = np.zeros(flwdir.shape, dtype=bool)
    flwpath_mask[flwpath_rowcol] = True

    # Extract inflow cell indices from the final flow path mask
    inflows_idxs = flwdir.inflow_idxs(flwpath_mask)

    # Generate stream segments (defined by flow path between two confluences)
    streams = flwdir.streams(mask=flwpath_mask)

    # Get upstream cell for each stream segment excluding cells in flow path
    downstream_indices = {stream['properties']['idx_ds'] for stream in streams}
    stream_upstream_cells = [
        stream['properties']['idx'] for stream in streams
        if stream['properties']['idx'] not in downstream_indices]

    # Separate inflows into upstream and lateral inflows based on whether their downstream cell is in the
    # flow path upstream cells
    lat_inflows_idxs = [
        inflow_point
        for inflow_point in inflows_idxs
        if flwdir.idxs_ds[inflow_point] not in stream_upstream_cells
    ]
    # lateral_inflows_rowcol = np.unravel_index(lateral_inflows, flwdir.shape)
    up_inflows_idxs = [
        inflow_point
        for inflow_point in inflows_idxs
        if flwdir.idxs_ds[inflow_point] in stream_upstream_cells
    ]

    # Calculate river widths using geomorphological relationship with upstream drainage area
    # W = w_coef_a * (A)^w_coef_b where:
    # W is river width in meters
    # A is upstream drainage area in m² 
    # w_coef_a and w_coef_b are empirical coefficients
    flwdir_upstream_area = flwdir.upstream_area(unit='m2')
    river_cells_widths = np.zeros(flwdir.shape, dtype=np.float32)
    river_cells_widths[flwpath_rowcol] = w_coef_a * (flwdir_upstream_area[flwpath_rowcol] * 10**-6)**w_coef_b
    
    # Get flow distance
    flwdst = flwdir.distnc.ravel()

    for stream in streams:
        coords = np.array(stream["geometry"]["coordinates"])
        stream["properties"]["rowcols"] = [
            tuple(map(np.int32, reversed(~flwdir.transform * coord))) for coord in coords
        ]
        stream["properties"]["idxs"] = [
            np.ravel_multi_index(rowcol, flwdir.shape) for rowcol in stream["properties"]["rowcols"]
        ]

        stream["properties"]["midpoint_coordinates"] = (coords[:-1] + coords[1:]) / 2
        stream["properties"]["midpoint_rowcols"] = stream["properties"]["rowcols"][:-1]
        stream["properties"]["midpoint_idxs"] = stream["properties"]["idxs"][:-1]
        stream["properties"]["nmidpoints"] = len(stream["properties"]["midpoint_coordinates"])

    # Populate cross sections
    cross_sections = []
    for stream in streams:
        for i in range(stream["properties"]["nmidpoints"]):
            if latlon: # calculate curvilinear abscissa based on latlon condition
                centroid_coords = np.array(stream["geometry"]["coordinates"][i])
                midpoint_coords = np.array(stream["properties"]["midpoint_coordinates"][i])
                lat_avg = (centroid_coords[1] + midpoint_coords[1]) / 2.0 # average latitude
                dx = degree_metres_x(lat_avg) * (centroid_coords[0] - midpoint_coords[0]) # horizontal length of a degree in metres at a given latitude
                dy = degree_metres_y(lat_avg) * (centroid_coords[1] - midpoint_coords[1]) # verical length of a degree in metres at a given latitude
                x = flwdst[stream["properties"]["midpoint_idxs"][i]] - np.hypot(dx, dy)
            else:
                x = flwdst[stream["properties"]["midpoint_idxs"][i]] - np.hypot(
                    *(stream["geometry"]["coordinates"][i] - stream["properties"]["midpoint_coordinates"][i])
                )
            
            # Hardcoded bathymetry value with 0.001 m/m slope
            bathy = x * 1e-3

            # Lateral inflows
            cs_lat_idxs = [
                inflow
                for inflow in lat_inflows_idxs
                if flwdir.idxs_ds[inflow] == stream["properties"]["idxs"][i]
            ]
            nlat = len(cs_lat_idxs)
            cs_lat_rowcols = np.empty(shape=(nlat, 2), dtype=np.int32)
            for j, inflow in enumerate(cs_lat_idxs):
                cs_lat_rowcols[j] = np.unravel_index(inflow, flwdir.shape)

            # Upstream inflows
            cs_up_idxs = [
                inflow
                for inflow in up_inflows_idxs
                if flwdir.idxs_ds[inflow] == stream["properties"]["idxs"][i]
            ]
            nup = len(cs_up_idxs)
            cs_up_rowcols = np.empty(shape=(nup, 2), dtype=np.int32)
            for j, inflow in enumerate(cs_up_idxs):
                cs_up_rowcols[j] = np.unravel_index(inflow, flwdir.shape)

            cross_sections.append(
                {
                    "coord": stream["properties"]["midpoint_coordinates"][i],
                    "rowcol": stream["properties"]["midpoint_rowcols"][i],
                    "x": x,
                    "bathy": bathy,
                    "nlevels": 1,
                    "manning": np.full(shape=(1,), fill_value=1 / 100, dtype=np.float32),
                    "level_heights": np.empty(shape=(1,), dtype=np.float32),
                    "level_widths": np.array([river_cells_widths[stream["properties"]["midpoint_rowcols"][i]]], dtype=np.float32),
                    "nlat": nlat,
                    "lat_rowcols": cs_lat_rowcols,
                    "nup": nup,
                    "up_rowcols": cs_up_rowcols,
                }
            )

    # Populate segments
    segments = []
    for i, stream in enumerate(streams):
        ds_seg = np.array(
            [
                j
                for j, other_stream in enumerate(streams)
                if other_stream["properties"]["idx"] == stream["properties"]["idx_ds"] and i != j
            ]
        )
        us_seg = np.array(
            [
                j
                for j, other_stream in enumerate(streams)
                if other_stream["properties"]["idx_ds"] == stream["properties"]["idx"] and i != j
            ]
        )

        cs_idxs = [np.ravel_multi_index(cs["rowcol"], flwdir.shape) for cs in cross_sections]
        first_cs = np.argwhere(cs_idxs == stream["properties"]["midpoint_idxs"][0]).item()
        last_cs = np.argwhere(cs_idxs == stream["properties"]["midpoint_idxs"][-1]).item()

        segments.append(
            {
                "first_cross_section": first_cs,
                "last_cross_section": last_cs,
                "nds_seg": len(ds_seg),
                "ds_segment": ds_seg,
                "nus_seg": len(us_seg),
                "us_segment": us_seg,
            }
        )

    return cross_sections, segments

def _compute_flow_direction(
    input_raster_path: str,
    river_line_path: str,
    is_flwdir_raster: bool,
    outlets: str,
    upscale_factor: int,
    burning_depth: int,
) -> tuple[pyflwdir.FlwdirRaster, pyflwdir.FlwdirRaster, np.ndarray]:
    """
    Compute upscaled flow direction from fine-resolution input raster.
    
    Parameters
    ----------
    input_raster_path : str
        Path to input raster (DEM or flow direction)
    is_flwdir_raster : bool
        Whether input is flow direction (True) or DEM (False)
    upscale_factor : int
        Factor by which to upscale the flow direction
        
    Returns
    -------
    dict
        Dictionary containing:
        - flwdir: Fine resolution flow direction object
        - ups_flwdir: Upscaled flow direction object
        - flwdir_smash_array: Upscaled flow direction array in SMASH convention
        - idxs_out: Linear indices of subgrid outlets (in fine-resolution grid)
        - epsg: EPSG code of input raster
    """
    # Read input raster
    with rasterio.open(input_raster_path, "r") as src:
        raster_data = src.read(1)
        transform = src.transform
        latlon = src.crs.is_geographic
        epsg = src.crs.to_epsg()
        profile = src.profile
        if is_flwdir_raster:
            flwdir = pyflwdir.from_array(
                data=raster_data,
                ftype='d8',
                transform=transform,
                latlon=latlon,
            )
        else:
            profile.update(nodata=-9999.0) # WARNING: set dem nodata to -9999.0 to match pyflwdir nodata (by default -9999.0)
            if burning_depth > 0:
                # Read river line
                river_gdf = gpd.read_file(river_line_path)
                # Generate river mask
                river_mask, _ = mask(
                    src, river_gdf.geometry, all_touched=False, nodata=0)
                # Create a boolean mask
                river_cells_mask = river_mask[0] != 0
                # Create a copy of the raw DEM data
                burned_dem_data = raster_data.copy()
                # Burn river cells with defined offset
                burned_dem_data[river_cells_mask] -= burning_depth
                # Generate flow direction object from burned DEM
                flwdir = pyflwdir.from_dem(
                        data=burned_dem_data,
                        transform=transform,
                        latlon=latlon,
                        outlets=outlets
                    )
            else:
                # Generate flow direction object from raw DEM (no stream burning)
                flwdir = pyflwdir.from_dem(
                    data=raster_data,
                    transform=transform,
                    latlon=latlon,
                    outlets=outlets
                )
            # Adjust elevation using Yamazaki et al., 2012 method
            # Hydrologically Adjusted DEM: downstream cells are not higher than upstream cells
            # WARNING: Adjusted DEM from RAW DEM (no stream burning)
            adjusted_dem = pyflwdir.dem.adjust_elevation(flwdir.idxs_ds, flwdir.idxs_seq, raster_data.flatten())

    # Upscale flow direction
    uparea = flwdir.upstream_area()
    ups_flwdir, idxs_out = flwdir.upscale(
        scale_factor=upscale_factor,
        uparea=uparea,
        method="ihu"
    )
    # Optional: Assess the quality of the upscaling
    flwerr = flwdir.upscale_error(ups_flwdir, idxs_out)
    percentage_error = np.sum(flwerr == 0) / np.sum(flwerr != 255) * 100

    ups_flwdir_smash_array = _switch_flwdir_convention(ups_flwdir.to_array(ftype='d8'), "d8-smash")

    results = {
            # Coarse (Upscaled) Grid properties
            "xres": ups_flwdir.transform.a,
            "yres": abs(ups_flwdir.transform.e),
            "xmin": ups_flwdir.bounds[0],
            "ymax": ups_flwdir.bounds[3],
            "epsg": epsg,

            # Flow direction data
            "fine_flwdir_object": flwdir, 
            "ups_flwdir_object": ups_flwdir,
            "ups_flwdir_smash_array": ups_flwdir_smash_array,
            "idxs_out": idxs_out, 
            "flwerr": flwerr,
            "percentage_error": percentage_error,
            }
    if not is_flwdir_raster:
        results.update(
            {
            "adjusted_dem": adjusted_dem,
            }
        )
    return results


def _generate_masked_mesh(
    flow_dir_data: dict,
    gauge_attributes: dict,
    max_depth: int
) -> dict:
    """
    Generate masked mesh using SMASH's catchment delineation algorithm.
    
    Parameters
    ----------
    flow_dir_data : dict
        Dictionary containing:
        - xres, yres: Grid resolutions
        - xmin, ymax: Grid coordinates
        - epsg: EPSG code
        - flwdir_smash_array: Flow direction array in SMASH convention
    gauge_attributes : dict
        Dictionary containing gauge information:
        - x: x-coordinates
        - y: y-coordinates
        - area: catchment areas
        - code: gauge codes
    max_depth : int
        Maximum search depth for catchment delineation
    
    Returns
    -------
    dict
        Mesh dictionary containing grid and flow properties
    """
    # Get flow direction array in smash convention
    flwdir = flow_dir_data["ups_flwdir_smash_array"]
    
    # Get CRS
    crs = rasterio.CRS.from_epsg(flow_dir_data["epsg"])
    
    # Calculate dx, dy based on coordinate system
    if crs.units_factor[0].lower() == "degree":
        dx, dy = mw_mesh.latlon_dxdy(
            flwdir.shape[0],
            flwdir.shape[1],
            flow_dir_data["xres"],
            flow_dir_data["yres"],
            flow_dir_data["ymax"]
        )
    else:
        dx = np.zeros(shape=flwdir.shape, dtype=np.float32) + flow_dir_data["xres"]
        dy = np.zeros(shape=flwdir.shape, dtype=np.float32) + flow_dir_data["yres"]
    
    # Extract gauge information
    x = np.array(gauge_attributes["x"])
    y = np.array(gauge_attributes["y"])
    area = np.array(gauge_attributes["area"])
    code = np.array(gauge_attributes["code"])
    
    # Initialize delineation arrays
    row_dln = np.zeros(shape=x.shape, dtype=np.int32)
    col_dln = np.zeros(shape=x.shape, dtype=np.int32)
    area_dln = np.zeros(shape=x.shape, dtype=np.float32)
    sink_dln = np.zeros(shape=x.shape, dtype=np.bool)
    mask_dln = np.zeros(shape=flwdir.shape, dtype=np.int32)
    
    # Delineate catchments
    for ind in range(x.size):
        row, col = _xy_to_rowcol(
            x[ind], y[ind],
            flow_dir_data["xmin"], flow_dir_data["ymax"],
            flow_dir_data["xres"], flow_dir_data["yres"]
        )
        slice_win = _get_catchment_slice_window(
            *flwdir.shape, row, col,
            area[ind], dx[row, col], dy[row, col],
            max_depth
        )
        row_win = row - slice_win[0].start
        col_win = col - slice_win[1].start
        
        dx_win = dx[slice_win]
        dy_win = dy[slice_win]
        flwdir_win = flwdir[slice_win]
        
        mask_dln_win, row_dln_win, col_dln_win, sink_dln[ind] = mw_mesh.catchment_dln_area_based(
            flwdir_win, dx_win, dy_win, row_win, col_win, area[ind], max_depth
        )

        row_dln[ind] = row_dln_win + slice_win[0].start
        col_dln[ind] = col_dln_win + slice_win[1].start
        area_dln[ind] = np.sum(mask_dln_win * dx_win * dy_win)
        mask_dln[slice_win] = np.where(mask_dln_win == 1, 1, mask_dln[slice_win])
    
    # Handle sinks warning
    if np.any(sink_dln):
        warnings.warn(
            f"One or more sinks were detected when trying to delineate the catchment(s): "
            f"'{code[sink_dln == 1]}'. The catchment(s) might not be correctly delineated avoiding the sink cells.",
            stacklevel=2,
        )
    
    # Create masked arrays and compute properties
    flwdir = np.ma.masked_array(flwdir, mask=(1 - mask_dln))
    flwdir, slice_win = _trim_mask_2d(flwdir, slice_win=True)
    dx = dx[slice_win]
    dy = dy[slice_win]
    
    # Update coordinates
    ymax_shifted = flow_dir_data["ymax"] - slice_win[0].start * flow_dir_data["yres"]
    xmin_shifted = flow_dir_data["xmin"] + slice_win[1].start * flow_dir_data["xres"]
    row_dln = row_dln - slice_win[0].start
    col_dln = col_dln - slice_win[1].start
    
    # Compute flow properties
    flwacc, flwpar = mw_mesh.flow_accumulation_partition(flwdir, dx, dy)
    flwdst = mw_mesh.flow_distance(flwdir, dx, dy, row_dln, col_dln, area_dln)
    
    # Mask flow properties
    flwdst = np.ma.masked_array(flwdst, mask=flwdir.mask)
    flwacc = np.ma.masked_array(flwacc, mask=flwdir.mask)
    
    # Compute partitioning
    npar = np.max(flwpar)
    ncpar, cscpar, cpar_to_rowcol = mw_mesh.flow_partition_variable(npar, flwpar)
    flwpar = np.ma.masked_array(flwpar, mask=flwdir.mask)
    
    # Compute active cells
    nac = np.count_nonzero(~flwdir.mask)
    active_cell = 1 - flwdir.mask.astype(np.int32)
    
    # Prepare gauge information
    ng = x.size
    gauge_pos = np.column_stack((row_dln, col_dln))
    
    return {
        "xmin": xmin_shifted,
        "ymax": ymax_shifted,
        "xres": flow_dir_data["xres"],
        "yres": flow_dir_data["yres"],
        "epsg": flow_dir_data["epsg"],
        "nrow": flwdir.shape[0],
        "ncol": flwdir.shape[1],
        "dx": dx,
        "dy": dy,
        "flwdir": flwdir,
        "flwdst": flwdst,
        "flwacc": flwacc,
        "flwpar": flwpar,
        "npar": npar,
        "ncpar": ncpar,
        "cscpar": cscpar,
        "cpar_to_rowcol": cpar_to_rowcol,
        "nac": nac,
        "active_cell": active_cell,
        "ng": ng,
        "gauge_pos": gauge_pos,
        "code": code,
        "area": area,
        "area_dln": area_dln
    }


def _compute_subgrid_network(
    masked_mesh: dict,
    flow_dir_data: dict,
    river_line_path: str,
    width_coef_a: float,
    width_coef_b: float,
    depth_coef_a: float,
    depth_coef_b: float,
    use_subgrid_area: bool = False,
    return_analysis_data: bool = False
) -> dict: 

    # Get flow direction data from masked mesh at smash convention
    masked_flwdir_data = _switch_flwdir_convention(masked_mesh['flwdir'].data, "smash-d8")

    # Create transform from masked mesh
    smash_transform = Affine(masked_mesh['xres'],  0, masked_mesh['xmin'], 0, -masked_mesh['yres'],masked_mesh['ymax'])

    # Create masked upscaled flow direction object using masked_mesh extent
    ups_flwdir = pyflwdir.from_array(
        data=masked_flwdir_data,
        ftype='d8',
        mask=~masked_mesh['flwdir'].mask, # WARNING: need to invert smash active cells mask
        transform=smash_transform,
        latlon=flow_dir_data["ups_flwdir_object"].latlon,
    )

    # Get fine resolution flow direction object, WARNING: at the initial extent !
    flwdir = flow_dir_data["fine_flwdir_object"]
    
    # Get linear indices of subgrid outlets (WARNING: at the initial extent !) and clip to masked mesh extent
    # Calculate the offset between full grid and masked grid
    x_offset = int((masked_mesh['xmin'] - flow_dir_data["ups_flwdir_object"].bounds[0]) / masked_mesh['xres'])
    y_offset = int((flow_dir_data["ups_flwdir_object"].bounds[3] - masked_mesh['ymax']) / masked_mesh['yres'])
    # Extract the slice based on these offsets
    idxs_out = flow_dir_data["idxs_out"][y_offset:y_offset+masked_mesh['nrow'], 
                           x_offset:x_offset+masked_mesh['ncol']]
    
    
    ### ----- CLIP RIVER NETWORK TO SMASH MASK -----------------------------------------------------------
    river_line = gpd.read_file(river_line_path)
    mask = ~masked_mesh['flwdir'].mask    
    mask_shapes = list(shapes(mask.astype(np.uint8), mask=mask, transform=smash_transform))
    polygons = [Polygon(s[0]["coordinates"][0]) for s in mask_shapes if s[1] == 1]
    mask_gdf = gpd.GeoDataFrame(geometry=polygons, crs=str(masked_mesh['epsg'])).dissolve()
    mask_geom = mask_gdf.union_all()
    # Check if all rivers are fully within mask
    within_mask = river_line.geometry.within(mask_geom)
    if not within_mask.all():
        # some river features fall outside the flow direction mask, clipping...
        cropped_river_gdf = gpd.clip(river_line, mask_geom)
        cropped_river_gdf = cropped_river_gdf.sort_index() # clipping can change row order.
        river_line = cropped_river_gdf  # overwrite original river network with clipped

    ### ----- COMPUTE COARSE RESOLUTION FLOW PATH -----------------------------------------------------------

    selected_paths = {}
    # Iterate over each line segment in the river GeoDataFrame
    for rl_idx, rl in river_line.iterrows():
        if rl["geometry"].geom_type == 'LineString':
            line_coords = list(rl["geometry"].coords)
        else:
            line_coords = [coord for linestring in rl["geometry"].geoms for coord in linestring.coords] # MultiLineString

        # Determine the starting point of the current line segment
        start_point = Point(line_coords[0][0], line_coords[0][1])

        # Transform the starting point coordinates to raster cell coordinates
        start_cell_col, start_cell_row = map(int, ~ups_flwdir.transform * (start_point.x, start_point.y))

        # Get the linear index of the start cell
        start_cell_idx = int(start_cell_row * ups_flwdir.shape[1] + start_cell_col)

        # Get the indices of the eight neighboring cells of the start cell
        neighbors_cells_idxs = _d8_idx(start_cell_idx, ups_flwdir.shape)
        neighbors_cells_idxs = neighbors_cells_idxs[np.isin(neighbors_cells_idxs, ups_flwdir.idxs_seq)]

        # Define flow path source cells as the start cell and its eight neighbors
        # Only add start_cell_idx if it's in valid cells
        if start_cell_idx in ups_flwdir.idxs_seq:
            source_cells = np.concatenate(([start_cell_idx], neighbors_cells_idxs)).astype(np.int32)
        else:
            source_cells = neighbors_cells_idxs.astype(np.int32)
        #source_cells = np.concatenate(([start_cell_idx], neighbors_cells_idxs)).astype(np.int32)

        # Compute flow paths starting from the source cells
        flwpaths, _ = ups_flwdir.path(idxs=source_cells)

        flwpaths_analysis = {}
        for fp in flwpaths:
            count = sum(
                sum(1 for x, y, *_ in line_coords if x_min <= x < x_max and y_min <= y < y_max)
                for cell in fp
                for row, col in [np.unravel_index(cell, ups_flwdir.shape)]
                for x_min, y_max in [ups_flwdir.transform * (col, row)]
                for x_max, y_min in [ups_flwdir.transform * (col + 1, row + 1)]
            )
            flwpaths_analysis[fp[0]] = count
        
        """# Get maximum vertices count
        max_count = max(flwpaths_analysis.values())
        # Define threshold (90% of max count)
        threshold = 0.95 * max_count
        # Check if start_cell_idx is within threshold of max count
        if flwpaths_analysis[start_cell_idx] >= threshold:
            best_cell_idx = start_cell_idx
        else:
            # Otherwise, use the cell with the maximum count
            best_cell_idx = max(flwpaths_analysis, key=lambda k: flwpaths_analysis[k])"""
    
        # Select the start cell index of the flow path with the highest number of line segment points
        best_cell_idx = max(flwpaths_analysis, key=lambda k: flwpaths_analysis[k])

        """# Store the selected path of the best start cell index
        selected_path[best_cell_idx] = flwpaths[np.where(source_cells == best_cell_idx)[0][0]]"""

        # Store the selected path of the best start cell index with the river segment index
        selected_paths[(rl_idx, best_cell_idx)] = flwpaths[np.where(source_cells == best_cell_idx)[0][0]]

    # Compute the flow path
    ups_flwpath = np.unique(np.concatenate(list(selected_paths.values())))
    ups_flwpath = ups_flwpath[np.isin(ups_flwpath, ups_flwdir.idxs_seq)]
    ups_flwpath_rowcol = np.unravel_index(ups_flwpath, ups_flwdir.shape)
    ups_flwpath_mask = np.zeros(ups_flwdir.shape, dtype=bool)
    ups_flwpath_mask[ups_flwpath_rowcol] = True

    # Extract inflow cell indices from the final flow path mask
    inflows_idxs = ups_flwdir.inflow_idxs(ups_flwpath_mask)

    # Generate stream segments (defined by flow path between two confluences)
    streams = ups_flwdir.streams(mask=ups_flwpath_mask)

    # Get upstream cell for each stream segment excluding cells in flow path
    downstream_indices = {stream['properties']['idx_ds'] for stream in streams}
    stream_upstream_cells = [
        stream['properties']['idx'] for stream in streams
        if stream['properties']['idx'] not in downstream_indices]

    # Separate inflows into upstream and lateral inflows based on whether their downstream cell is in the
    # flow path upstream cells
    lat_inflows_idxs = [
        inflow_point
        for inflow_point in inflows_idxs
        if ups_flwdir.idxs_ds[inflow_point] not in stream_upstream_cells
    ]
    # lateral_inflows_rowcol = np.unravel_index(lateral_inflows, flwdir.shape)
    up_inflows_idxs = [
        inflow_point
        for inflow_point in inflows_idxs
        if ups_flwdir.idxs_ds[inflow_point] in stream_upstream_cells
    ]

    ### ----- COMPUTE FINE RESOLUTION FLOW PATH FROM SUBGRID OULETS-----------------------------------------------------------

    # Filter unit catchment outlets using river mask
    river_cells_outlet_pixels = idxs_out[ups_flwpath_mask]
    # Create a mask where only the basin outlet is True
    fine_res_basin_outlet_mask = np.zeros(flwdir.shape, dtype=bool)
    # Get the outlet cell from the last stream segment
    outlet_cell_idx = streams[-1]['properties']['idx']
    # Get outlet pixel of outlet cell
    outlet_pixel_idx = idxs_out.ravel()[outlet_cell_idx] 
    # Set only the basin outlet pixel to True
    fine_res_basin_outlet_mask.flat[outlet_pixel_idx] = True
    # Trace paths, stopping at the basin outlet
    fine_res_flwpath, _ = flwdir.path(
        idxs=river_cells_outlet_pixels, 
        mask=fine_res_basin_outlet_mask
    )
    fine_res_flwpath = np.unique(np.concatenate(fine_res_flwpath))

    fine_res_flwpath_mask = np.zeros(flwdir.shape, dtype=bool)
    fine_res_flwpath_mask[np.unravel_index(fine_res_flwpath, flwdir.shape)] = True
    # Create a masked flow direction object
    masked_fine_flwdir = pyflwdir.from_array(
        data=flwdir.to_array(),
        ftype='d8',
        mask=fine_res_flwpath_mask,  # Only cells in flow path are valid
        transform=flwdir.transform,
        latlon=flwdir.latlon
    )
    # Compute Fine Resolution Stream Segments
    river_fine_streams = masked_fine_flwdir.streams()

    # Compute Confluences of Fine Resolution Stream Segments
    confluences = confluence_indices(masked_fine_flwdir.idxs_ds)

    ### ----- COMPUTE SUBGRID CROSS SECTIONS -----------------------------------------------------------

    # Initialize lists
    unique_combinations = set()
    cell_list, pixel_list, inflow_list = [], [], []
    confluence_distances = {}

    # Process each fine stream except the last one
    for idx, stream in enumerate(river_fine_streams[:-1]):
    #for idx, stream in enumerate(river_fine_streams):
        x_coords, y_coords = np.array(stream['geometry']['coordinates']).T
        stream_line = LineString(zip(x_coords, y_coords))
        pixel_idxs = flwdir.index(x_coords, y_coords)
        outlet_pixel_values = pixel_idxs[np.isin(pixel_idxs, river_cells_outlet_pixels)]
        
        # Handle empty segments first
        if len(outlet_pixel_values) == 0:
            #print(f"\nEmpty segment at index {idx}")
            mid_idx = len(pixel_idxs) // 2
            new_pixel = pixel_idxs[mid_idx]
            combination = (-1, new_pixel) # no cell index, attribute -1 value
            if combination not in unique_combinations:
                unique_combinations.add(combination)
                cell_list.append(-1)
                pixel_list.append(new_pixel)
                inflow_list.append(False)
            continue 
        else:
            for pixel_value in outlet_pixel_values:
                cell_idx = np.where(idxs_out.ravel() == pixel_value)[0][0]
                cell_point = Point(ups_flwdir.xy(cell_idx))
                
                if pixel_value in confluences:
                    #print(f"\nConfluence at pixel {pixel_value}")
                    is_end = pixel_value == pixel_idxs[-1]
                    
                    if is_end:
                        distance = cell_point.distance(stream_line)

                        if len(outlet_pixel_values) > 1:
                            curr_idx = np.where(pixel_idxs == pixel_value)[0][0]
                            prev_idx = np.where(pixel_idxs == outlet_pixel_values[np.where(outlet_pixel_values == pixel_value)[0][0] - 1])[0][0]
                            mid_idx = prev_idx + (curr_idx - prev_idx) // 2
                            new_pixel = pixel_idxs[mid_idx]
                        else:
                            mid_idx = len(pixel_idxs) // 2
                            new_pixel = pixel_idxs[mid_idx]
                        
                        if pixel_value not in confluence_distances:
                            confluence_distances[pixel_value] = {
                                'distance': distance,
                                'segment': idx,
                                'cell': cell_idx,
                                'new_pixel': new_pixel,
                                'list_index': len(cell_list)
                            }
                            inflow_value = True
                        else:
                            if distance < confluence_distances[pixel_value]['distance']:
                                inflow_list[confluence_distances[pixel_value]['list_index']] = False
                                confluence_distances[pixel_value].update({
                                    'distance': distance,
                                    'segment': idx,
                                    'cell': cell_idx,
                                    'new_pixel': new_pixel,
                                    'list_index': len(cell_list)
                                })
                                inflow_value = True
                            else:
                                inflow_value = False
                    else:
                        if len(outlet_pixel_values) > 1:
                            curr_idx = np.where(pixel_idxs == pixel_value)[0][0]
                            next_idx = np.where(pixel_idxs == outlet_pixel_values[np.where(outlet_pixel_values == pixel_value)[0][0] + 1])[0][0]
                            mid_idx = curr_idx + (next_idx - curr_idx) // 2
                            new_pixel = pixel_idxs[mid_idx]
                        else:
                            mid_idx = len(pixel_idxs) // 2
                            new_pixel = pixel_idxs[mid_idx]
                        inflow_value = False
                    
                    combination = (cell_idx, new_pixel)
                else:
                    combination = (cell_idx, pixel_value)
                    inflow_value = True
                    
                if combination not in unique_combinations:
                    unique_combinations.add(combination)
                    cell_list.append(cell_idx)
                    pixel_list.append(combination[1])
                    inflow_list.append(inflow_value)

    
    # Method 1: Using integer division
    nrow, ncol = ups_flwdir.shape
    INVALID_COORDS = (-99, -99)
    cell_rowcols = [(np.int32(idx // ncol), np.int32(idx % ncol)) if idx != -1 
                else INVALID_COORDS for idx in cell_list]
    """# Method 2: Using coordinate transformation
    x, y = ups_flwdir.xy(np.array(cell_list))[0], ups_flwdir.xy(np.array(cell_list))[1]
    rows, cols = pyflwdir.gis_utils.rowcol(ups_flwdir.transform, x, y)
    cell_rowcols = list(zip(np.int32(rows), np.int32(cols)))"""

# generate x, y coords of cross sections (pixels)
    cs_x_coords, cs_y_coords = masked_fine_flwdir.xy(np.array(pixel_list))[0], masked_fine_flwdir.xy(np.array(pixel_list))[1]
    cs_coords =  [(int(x), int(y)) for x, y in zip(cs_x_coords, cs_y_coords)]

    # Create DataFrame
    df_cs = pd.DataFrame({
        'cell': cell_list,
        'rowcol': cell_rowcols,
        'pixel': pixel_list,
        'cs_coords': cs_coords,
        'inflow': inflow_list
    })

    # Compute metrics for each cell-pixel pair
    upstream_area_list = []
    curvilinear_abscissa_list = []
    bathy_list = []
    level_width_list = []
    depth_list = []
    adj_elevtn_list = []

    fine_upstream_area_map = flwdir.upstream_area(unit='km2').ravel()

    # Calculate metrics for each entry in df_cs
    for pixel_value in df_cs['pixel']:
        upstream_area = fine_upstream_area_map[pixel_value]
        """WARNING : use masked_fine_flwdir for curvilinear abcissa calculation instead of fine_flwdir"""
        curvilinear_abscissa = masked_fine_flwdir.distnc.ravel()[pixel_value]
        if "adjusted_dem" not in flow_dir_data:
            bathy = curvilinear_abscissa * 1e-3
            bathy_list.append(bathy)
        level_width = width_coef_a * upstream_area**width_coef_b
        depth = depth_coef_a * upstream_area**depth_coef_b
        
        upstream_area_list.append(upstream_area)
        curvilinear_abscissa_list.append(curvilinear_abscissa)
        level_width_list.append(level_width)
        depth_list.append(depth)
        if "adjusted_dem" in flow_dir_data:
            adj_elevtn_list.append(flow_dir_data["adjusted_dem"][pixel_value])


        # DEBUG: test garonne without dem based bathy
        #bathy = curvilinear_abscissa * 1e-5
        #bathy = curvilinear_abscissa *1e-3
        #bathy_list.append(bathy)

    # Add computed metrics to DataFrame
    df_cs['upstream_area'] = upstream_area_list
    df_cs['curvilinear_abscissa'] = curvilinear_abscissa_list
    df_cs['level_width'] = level_width_list
    df_cs['depth'] = depth_list
    if "adjusted_dem" in flow_dir_data:
        #df_cs['bathy'] = bathy_list # DEBUG: test garonne without dem based bathy
        df_cs['adj_elevtn'] = adj_elevtn_list
        df_cs['bathy'] = df_cs['adj_elevtn'] - df_cs['depth']

    else: # bathy is computed from curvilinear abscissa when using directly flow direction raster
        df_cs['bathy'] = bathy_list

    # Preallocate lists for inflows
    n_cells = len(df_cs)
    lateral_inflows_list = [[] for _ in range(n_cells)]
    lateral_inflows_coords_list = [[] for _ in range(n_cells)]
    nlat_list = np.zeros(n_cells, dtype=np.int32)

    upstream_inflows_list = [[] for _ in range(n_cells)]
    upstream_inflows_coords_list = [[] for _ in range(n_cells)]
    nup_list = np.zeros(n_cells, dtype=np.int32)

    # Process inflows based on inflow boolean
    for i, (cell_idx, has_inflow) in enumerate(zip(df_cs['cell'], df_cs['inflow'])):
        if has_inflow:
            # Get lateral inflows
            cs_lat_idxs = np.array([inflow for inflow in lat_inflows_idxs if ups_flwdir.idxs_ds[inflow] == cell_idx])
            nlat_list[i] = len(cs_lat_idxs)
            #lateral_inflows_list[i] = cs_lat_idxs.tolist()
            #lateral_inflows_coords_list[i] = list(map(tuple, np.column_stack(np.unravel_index(cs_lat_idxs, ups_flwdir.shape)))) if cs_lat_idxs.size > 0 else np.zeros((0,2), dtype=np.float32)
            lateral_inflows_coords_list[i] = (
                np.column_stack(np.unravel_index(cs_lat_idxs, ups_flwdir.shape)).astype(np.int32) 
                if cs_lat_idxs.size > 0 
                else np.zeros((0,2), dtype=np.int32)
            )
            # Get upstream inflows
            cs_up_idxs = np.array([inflow for inflow in up_inflows_idxs if ups_flwdir.idxs_ds[inflow] == cell_idx])
            nup_list[i] = len(cs_up_idxs)
            #upstream_inflows_list[i] = cs_up_idxs.tolist()
            #upstream_inflows_coords_list[i] = list(map(tuple, np.column_stack(np.unravel_index(cs_up_idxs, ups_flwdir.shape)))) if cs_up_idxs.size > 0 else np.zeros((0,2), dtype=np.float32)
            upstream_inflows_coords_list[i] = (
                np.column_stack(np.unravel_index(cs_up_idxs, ups_flwdir.shape)).astype(np.int32)
                if cs_up_idxs.size > 0 
                else np.zeros((0,2), dtype=np.int32)
            )

    # Assign to DataFrame
    df_cs['lat_idxs'] = lateral_inflows_list
    df_cs['nlat'] = nlat_list
    df_cs['lat_rowcols'] = lateral_inflows_coords_list

    df_cs['up_idxs'] = upstream_inflows_list
    df_cs['nup'] = nup_list
    df_cs['up_rowcols'] = upstream_inflows_coords_list

    df_cs['nlevels'] = np.ones(len(df_cs), dtype=np.int32)
    df_cs['manning'] = np.full(len(df_cs), 1/30, dtype=np.float32) # K = 30 m^(1/3)/s (Larnier et al., 2025 --> first param: an arbitrary ”mild” value  352 for large rivers)
    df_cs['level_heights'] = [np.empty(shape=(1,), dtype=np.float32)] * len(df_cs)

    # Convert DataFrame to SMASH format
    cross_sections = [{
        'rowcol': cs['rowcol'],
        'cs_coords': cs['cs_coords'],
        "x": cs['curvilinear_abscissa'],
        "bathy": cs['bathy'],
        "nlevels": cs['nlevels'],
        "manning": np.full(1, cs['manning'], dtype=np.float32),
        "level_heights": cs['level_heights'],
        "level_widths": np.array([cs['level_width']], dtype=np.float32),
        "nlat": cs['nlat'],
        "lat_rowcols": (
            np.array(cs['lat_rowcols'], dtype=np.int32) 
            if cs['nlat'] > 0 
            else np.zeros((0,2), dtype=np.int32)
        ),
        "nup": cs['nup'],
        "up_rowcols": (
            np.array(cs['up_rowcols'], dtype=np.int32) 
            if cs['nup'] > 0 
            else np.zeros((0,2), dtype=np.int32)
        ),
        **({'elevtn': cs['adj_elevtn']} if 'adj_elevtn' in cs else {})
    } for _, cs in df_cs.iterrows()]



    ### ----- COMPUTE SEGMENTS (df_seg) -----------------------------------------------------------

    # Create mapping for quick pixel lookup
    pixel_to_index = {pixel: idx for idx, pixel in enumerate(df_cs['pixel'])}

    # Initialize segments list
    segments = []

    # Process each stream to create segments
    for i, stream in enumerate(river_fine_streams[:-1]):
    #for i, stream in enumerate(river_fine_streams):
        x_coords, y_coords = zip(*stream['geometry']['coordinates'])
        pixel_idxs = flwdir.index(x_coords, y_coords)
        
        # Get all cross-section indices for this stream
        cross_section_indices = [pixel_to_index[pixel] for pixel in pixel_idxs if pixel in pixel_to_index]
        
        ds_seg = np.array([
            j for j, other_stream in enumerate(river_fine_streams[:-1])
            if other_stream['properties']['idx'] == stream['properties']['idx_ds'] and i != j
        ])
        us_seg = np.array([
            j for j, other_stream in enumerate(river_fine_streams[:-1])
            if other_stream['properties']['idx_ds'] == stream['properties']['idx'] and i != j
        ])
        
        # Create segment in SMASH format
        segments.append({
            'first_cross_section': cross_section_indices[0] if cross_section_indices else None,
            'last_cross_section': cross_section_indices[-1] if cross_section_indices else None,
            'nds_seg': len(ds_seg),
            'ds_segment': ds_seg,
            'nus_seg': len(us_seg),
            'us_segment': us_seg,
        })

    ### ----- COMPUTE UNIT CATCHMENT AREAS -----------------------------------------------------------
    
    # coords of the upscaled flow direction cells at full extent
    xs_full, ys_full = flow_dir_data["ups_flwdir_object"].xy(flow_dir_data["ups_flwdir_object"].idxs_seq)

    # coords of the upscaled flow direction cells masked from smash
    xs_masked, ys_masked = ups_flwdir.xy(ups_flwdir.idxs_seq)

    # Stack as (N, 2) arrays
    coords_full = np.column_stack((xs_full, ys_full))
    coords_masked = np.column_stack((xs_masked, ys_masked))

    # Build lookup dictionary
    coord_to_full_idx = {tuple(coord): idx for idx, coord in enumerate(coords_full)}

    # Find indices of masked cells in full extent
    masked_in_full_indices = np.array([coord_to_full_idx.get(tuple(coord), -1) for coord in coords_masked])

    # unit catchment areas for the full extent upscaled flow direction
    _, ucat_are_full = flwdir.ucat_area(flow_dir_data["idxs_out"], unit='m2')

    # Flatten full catchment area array
    ucat_are_full_flat = ucat_are_full.ravel()

    # Get linear indices in the full grid for the masked cells
    full_linear_indices_for_masked = flow_dir_data["ups_flwdir_object"].idxs_seq[masked_in_full_indices]

    # Get masked catchment areas
    masked_ucat_are = ucat_are_full_flat[full_linear_indices_for_masked]

    # reshape the masked grid shape
    masked_ucat_are_2d = np.full(ups_flwdir.shape, np.nan)
    masked_ucat_are_2d.flat[ups_flwdir.idxs_seq] = masked_ucat_are


    result = {
        # Subgrid network properties
        "ncs": len(cross_sections),
        "cross_sections": cross_sections,
        "nseg": len(segments),
        "segments": segments,
        "use_subgrid_area": use_subgrid_area,
    }

    if use_subgrid_area:
        result["ucat_area"] = masked_ucat_are_2d

    if return_analysis_data:
            # Create visualization dictionary
            visu = {
                "df_cs": df_cs,
                "fine_flwdir": flwdir,
                "masked_fine_flwdir": masked_fine_flwdir,
                "ups_flwdir": ups_flwdir,
                "ups_flwdir_full_extent": flow_dir_data["ups_flwdir_object"],
                "idxs_out_full_extent": flow_dir_data["idxs_out"],
                "streams": streams,
                "river_fine_streams": river_fine_streams,
                "river_cells_outlet_pixels": river_cells_outlet_pixels,
                "idxs_out": idxs_out,
                "confluences": confluences,
                "ups_flwpath": ups_flwpath,
                "smash_river_line": river_line,
                "flwerr": flow_dir_data["flwerr"],
                "percentage_error": flow_dir_data["percentage_error"]
            }

            if use_subgrid_area:
                visu["ucat_area"] = masked_ucat_are_2d

            # Add it to result dictionary
            result["visualization"] = visu

            """# Add data for visualization and analysis
            result.update({
                "df_cs": df_cs,
                "fine_flwdir": flwdir,
                "masked_fine_flwdir": masked_fine_flwdir,
                "ups_flwdir": ups_flwdir,
                "streams": streams,
                "river_fine_streams": river_fine_streams,
                "river_cells_outlet_pixels": river_cells_outlet_pixels,
                "confluences": confluences
            })"""
    return result