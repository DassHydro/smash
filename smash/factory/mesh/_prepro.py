import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import pyflwdir



def _d8_idx(idx0, shape):
    """Returns linear indices of eight neighboring cells"""
    nrow, ncol = shape
    # assume c-style row-major
    r = idx0 // ncol
    c = idx0 % ncol
    idxs_lst = list()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:  # skip pit -> return empty array
                continue
            r_us, c_us = r + dr, c + dc
            if r_us >= 0 and r_us < nrow and c_us >= 0 and c_us < ncol:  # check bounds
                idx = r_us * ncol + c_us
                idxs_lst.append(idx)
    return np.array(idxs_lst)

def _extract_inflows(flw: pyflwdir.FlwdirRaster, river_line: str) -> tuple[
    tuple[np.ndarray, np.ndarray],  
    np.ndarray,                      
    np.ndarray,                      
    tuple[np.ndarray, np.ndarray],  
    tuple[np.ndarray, np.ndarray] 
]:
    """
    Compute the flow path and extract upstream and lateral inflow points.
    
    Parameters
    ----------
    flw : pyflwdir.FlwdirRaster
        An actionable flow direction object of pyflwdir.
        
    river_line : str
        Path to the river line shapefile.
        
    Returns
    -------
    flow_path_rows_cols : tuple[np.ndarray, np.ndarray]
        Tuple of arrays containing the row and column coordinates of the cells along the flow path.
        
    final_flow_path : np.ndarray
        Array containing linear indices of the cells along the flow path.
        
    inflows_idxs : np.ndarray
        Array containing linear indices of inflow cells.
    
    upstream_inflows_rows_cols : tuple[np.ndarray, np.ndarray]
        Tuple of arrays containing the row and column coordinates of upstream inflow cells.
        
    lateral_inflows_rows_cols : tuple[np.ndarray, np.ndarray]
        Tuple of arrays containing the row and column coordinates of lateral inflow cells.
    """
    
    # Load the river line shapefile and explode any multi-linestring geometries
    river_gdf = gpd.read_file(river_line).explode(index_parts=True)
    selected_path = dict()

    # Iterate over each line segment in the river GeoDataFrame
    for _, row in tqdm(river_gdf.iterrows(), total=len(river_gdf), desc="Processing rows"):
        
        # Get the coordinates of the current line segment
        line_coords = list(row['geometry'].coords)
        
        # Determine the starting point of the current line segment
        start_point = Point(line_coords[0][0], line_coords[0][1])
        
        # Transform the starting point coordinates to raster cell coordinates
        start_point_col, start_point_row = ~flw.transform * \
            (start_point.x, start_point.y)
        start_cell_row, start_cell_col = int(
            start_point_row), int(start_point_col)
        
        # Get the linear index of the start cell 
        start_cell_idx = start_cell_row * flw.shape[1] + start_cell_col
        
        # Get the indices of the eight neighboring cells of the start cell
        neighbors_cells_idxs = _d8_idx(start_cell_idx, flw.shape)
        neighbors_cells_idxs = neighbors_cells_idxs[np.isin(
            neighbors_cells_idxs, flw.idxs_seq)]
        
        # Define flow path source cells as the start cell and its eight neighbors
        source_cells = np.concatenate(
            ([start_cell_idx], neighbors_cells_idxs))

        # Compute flow paths starting from the source cells
        flow_paths, _ = flw.path(idxs=source_cells)

        flow_paths_analysis = dict()
        for flow_path in flow_paths:
            count = 0
            for cell in flow_path:
                row, col = np.unravel_index(cell, flw.shape)
                x_min, y_max = flw.transform * (col, row)
                x_max, y_min = flw.transform * (col + 1, row + 1)
                # Count the number of line segment points within the flow path cell
                count += sum(
                    1 for x_point, y_point, _ in line_coords
                    if x_min <= x_point < x_max and y_min <= y_point < y_max
                )
                
            #Store the total number of line segment points within each flow path, with the start cell as the key
            flow_paths_analysis[flow_path[0]] = count

        # Select the start cell index of the flow path with the highest number of line segment points
        best_cell_idx = max(flow_paths_analysis,
                            key=lambda k: flow_paths_analysis[k])

        # Store the selected path of the best start cell index
        selected_path[best_cell_idx] = flow_paths[np.where(
            source_cells == best_cell_idx)[0][0]]
    
    # Identify upstream cells from the selected paths, excluding invalid cells and outlet
    flow_path_upstream_cells = [cell for cell in list(
        selected_path.keys()) if cell in flw.idxs_seq and cell != flw.idxs_pit[0]]

    # Compute the final flow path
    final_flow_path = np.unique(
        np.concatenate(list(selected_path.values())))
    final_flow_path = final_flow_path[np.isin(
        final_flow_path, flw.idxs_seq)]

    # Convert final flow path indices to row and column coordinates and create a mask
    final_flow_path_rows, final_flow_path_cols = np.unravel_index(
        final_flow_path, flw.shape)
    final_flow_path_mask = np.zeros(flw.shape, dtype=bool)
    final_flow_path_mask[final_flow_path_rows, final_flow_path_cols] = True
    
    # Extract inflow cell indices from the final flow path mask
    inflows_idxs = flw.inflow_idxs(final_flow_path_mask)

    # Separate inflows into upstream and lateral inflows based on whether their downstream cell is in the flow path upstream cells
    upstream_inflows = [
        inflow_point for inflow_point in inflows_idxs if flw.idxs_ds[inflow_point] in flow_path_upstream_cells]
    lateral_inflows = [
        inflow_point for inflow_point in inflows_idxs if flw.idxs_ds[inflow_point] not in flow_path_upstream_cells]

    # Convert final flow path, upstream inflows, and lateral inflows indices to row and column coordinates
    flow_path_rows_cols = (final_flow_path_rows, final_flow_path_cols)
    upstream_inflows_rows, upstream_inflows_cols = np.unravel_index(
    upstream_inflows, flw.shape)
    upstream_inflows_rows_cols = (upstream_inflows_rows, upstream_inflows_cols)
    lateral_inflows_rows, lateral_inflows_cols = np.unravel_index(
    lateral_inflows, flw.shape)
    lateral_inflows_rows_cols = (lateral_inflows_rows, lateral_inflows_cols)
    
    return flow_path_rows_cols, final_flow_path, inflows_idxs, upstream_inflows_rows_cols, lateral_inflows_rows_cols


def _compute_river_geometry(
    flw: pyflwdir.FlwdirRaster,
    flow_path_rows_cols: tuple[np.ndarray, np.ndarray],
    a: int,
    b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate river widths based on upstream drainage area
    
    Parameters
    ----------
    flw : pyflwdir.FlwdirRaster
        An actionable flow direction object of pyflwdir.
        
    flow_path_rows_cols : tuple[np.ndarray, np.ndarray]
        Tuple of arrays containing the row and column coordinates of the cells along the flow path.
        
    a : int
        Parameter coefficient for river width calculation.
        
    b : int
        Parameter exponent for river width calculation.
        
    Returns
    -------
    flwdir_accu_areas : np.ndarray
        Array containing flow accumulation data (units: [m^2]).
    
    river_drainage_areas : np.ndarray
        Array containing river upstream drainage area data (units: [m^2]).

    river_widths : np.ndarray
        Array containing river width data (units: [m]).
        
    Notes
    -----
    The formula to calculate the river width is adapted from Vatankhah and Easa (2013):
    W = a * A^b
    where W is the river width, A is the upstream drainage area of the cell.
    
    """
        
    rows, cols = flow_path_rows_cols
    
    # Calculate flow accumulation data
    flwdir_accu_areas = flw.accuflux(data=flw.area)
    
    # Select river cells flow accumulation data
    river_accu_areas = flwdir_accu_areas[flow_path_rows_cols]
    
    # Calculate river cells upstream drainage area
    river_drainage_areas = np.zeros(flw.shape, dtype=np.float64)
    river_drainage_areas[rows, cols] = river_accu_areas
    
    
    # Calculate river cells widths 
    widths = a * (river_accu_areas * 10 **-6) ** b
    river_widths = np.zeros(flw.shape, dtype=np.float64)
    river_widths[rows, cols] = widths
    

    return flwdir_accu_areas, river_drainage_areas, river_widths
    

