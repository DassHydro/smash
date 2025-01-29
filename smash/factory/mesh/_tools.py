from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import rasterio.features
from shapely.geometry import Point, Polygon
from pyflwdir.gis_utils import degree_metres_x, degree_metres_y

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


def _get_cross_sections_and_segments(river_line: gpd.GeoDataFrame, rr_mesh: dict[str, Any]) -> np.ndarray:
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
        # Extract line coordinates
        line_coords = list(rl.coords)

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
                    "level_widths": np.full(shape=(1,), fill_value=5, dtype=np.float32),
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
