import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapelysmooth import taubin_smooth
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


def _compute_streams_paths(
    stream_index,
    streams,
    visited=None):
    """
    Compute streams paths from a given stream index.
    """
    if visited is None:
        visited = set()
    
    if stream_index in visited:
        return []
    
    visited.add(stream_index)
    stream = streams[stream_index]
    downstream_segments = stream['ds_seg']
    all_downstream = [stream_index]

    for ds in downstream_segments:
        if ds != -1:
            all_downstream.extend(_compute_streams_paths(ds, streams, visited))
    
    return all_downstream

def _hydro_prepro(
    flw: pyflwdir.FlwdirRaster, 
    river_line: str,
    a: int,
    b: int):
    """
    Preprocessing for coupling hydrological and hydraulic models:
        - Compute the flow path and extract upstream and lateral inflow points.
        - Compute river top widths based on upstream drainage area.
        - ...
    
    
    Parameters
    ----------
    flw : pyflwdir.FlwdirRaster
        An actionable flow direction object of pyflwdir.
        
    river_line : str
        Path to the river line shapefile.
        
    a : int
        Parameter coefficient for river width calculation.
        
    b : int
        Parameter exponent for river width calculation.
        
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
    
        
    # Calculate flow accumulation data
    flwdir_accu_areas = flw.accuflux(data=flw.area)
    
    # Select river cells flow accumulation data
    river_accu_areas = flwdir_accu_areas[flow_path_rows_cols]
    
    # Calculate river cells upstream drainage area
    river_drainage_areas = np.zeros(flw.shape, dtype=np.float64)
    river_drainage_areas[final_flow_path_rows, final_flow_path_cols] = river_accu_areas
    
    # Calculate river cells widths from upstream drainage area ≡ cross-sections top width
    # TODO: use formulation used in the MGB-IPH model, equation 12 in Pontes et al., 2017
    widths = a * (river_accu_areas * 10 **-6) ** b
    river_widths = np.zeros(flw.shape, dtype=np.float64)
    river_widths[final_flow_path_rows, final_flow_path_cols] = widths
    
    # TODO: Calculate river cells depth from top width, equation 13 in Pontes et al., 2017
    
    # TODO: adapt input parameters for the method: "4 fitting parameters, relating the river depth and the drainage area, and the river width and the drainage area"
    
    # Get stream segments from the final flow path mask
    streams = flw.streams(mask = final_flow_path_mask)
    
    # TODO: Remove the last stream segment ≡ segment with the outlet cell: streams = streams[:-1]
    # --> how to include outlet section?
    
    # Get Strahler stream order 
    stream_order = flw.stream_order(mask = final_flow_path_mask)
    for stream in streams:
        # Get start point index of stream
        start_idx = np.array([stream['properties']['idx']])
        # Get start point raster coordinates
        start_row, start_col = np.unravel_index(start_idx, flw.shape)
        # Get the segment stream order
        segment_stream_order = stream_order[start_row[0], start_col[0]]
        # Add the new key-value pair to the stream dictionary
        stream['properties']['strahler_stream_order'] = segment_stream_order
    
    # Get stream cells indices and river widths
    for stream in streams:
        coords = stream['geometry']['coordinates']
        stream_cells_idxs = list()
        stream_widths = list()
        for coord in coords:
            x,y = coord
            col, row = ~flw.transform * (x, y)
            row, col = int(row), int(col)
            
            idx = np.ravel_multi_index((row, col), flw.shape)
            stream_cells_idxs.append(idx)
            
            w = river_widths[row, col]
            stream_widths.append(w)
        stream['properties']['stream_cells_idxs'] = stream_cells_idxs
        stream['properties']['widths'] = stream_widths
    
    
    # Calculate midpoints coordinates on the stream
    intermediate_streams = list()
    for stream_id, stream in enumerate(streams):
        coords = np.array(stream['geometry']['coordinates'])
        midpoints = (coords[:-1] + coords[1:]) / 2    
        intermediate_streams.append({
        'original_pts_coords': stream['geometry']['coordinates'],
        'stream_cells_idx': stream['properties']['stream_cells_idxs'],
        'original_pts_widths': stream['properties']['widths'],
        'mid_pts_coords': midpoints.tolist(),
        'mid_pts_widths': stream['properties']['widths'][:-1], # remove last crosssection width
        'idx': stream['properties']['idx'],
        'idx_ds': stream['properties']['idx_ds'],
        'pit': stream['properties']['pit'],
        'strahler_stream_order': stream['properties']['strahler_stream_order']
        })
        
    # Smooth the river streams generated from flow path with taubin algorithm
    geometries = [LineString(stream['geometry']['coordinates']) for stream in streams]
    streams_gdf = gpd.GeoDataFrame({'geometry': geometries})
    taubin_smooth_geometries = [taubin_smooth(linestring) for linestring in streams_gdf['geometry']]
    taubin_smooth_gdf = gpd.GeoDataFrame(geometry=taubin_smooth_geometries)
    
    # Project the river streams mid-points on the smoothed stream lines
    # TODO : add projections as parameters
    original_crs = 'EPSG:4326'
    target_crs = 'EPSG:2154'
    smoothed_streams = list()
    for index, stream in enumerate(intermediate_streams):
        pts_to_project = [Point(coord) for coord in stream['mid_pts_coords']]
        smoothed_stream_line = taubin_smooth_gdf.loc[index].geometry
        projection = [nearest_points(point, smoothed_stream_line)[1].coords[0] for point in pts_to_project]
        
        # Convert geographic coordinates to planar coordinates
        gdf = gpd.GeoDataFrame(geometry=[Point(coord) for coord in projection], crs=original_crs)
        gdf_planar = gdf.to_crs(target_crs)
        planar_projection = [(point.x, point.y) for point in gdf_planar.geometry]
            
        smoothed_streams.append({'original_centroid_coords': stream['original_pts_coords'][:-1],
                                'stream_cells_idx': stream['stream_cells_idx'][:-1],
                                'geographic_projected_xs_coords': projection,
                                'planar_projected_xs_coords': planar_projection,
                                'projected_xs_width': stream['mid_pts_widths'],
                                'idx':stream['idx'],
                                'idx_ds':stream['idx_ds'],
                                'pit': stream['pit'],
                                'strahler_stream_order': stream['strahler_stream_order']})

    # Create a list of unique projected cross-sections on the smoothed stream lines
    smoothed_xs = list()
    unique_coords = set()
    for stream in smoothed_streams:
        coordinates = stream['planar_projected_xs_coords']
        for i, coord in enumerate(coordinates):
            if coord not in unique_coords:
                unique_coords.add(coord)
                smoothed_xs.append({
                    'planar_projected_xs_coords': coord,
                    'stream_cells_idx': stream['stream_cells_idx'][i],
                    'projected_xs_width': stream['projected_xs_width'][i],
                    'original_centroid_coords': stream['original_centroid_coords'][i]
                })
    
    # Compute segments connectivities
    for stream in smoothed_streams:
        
        # Get upstream segments indices for the current stream
        idx = stream['idx']
        us_seg = list()
        for id, other_stream in enumerate(smoothed_streams):
            if other_stream['idx_ds'] == idx:
                us_seg.append(id)
        if us_seg:
            stream['us_seg'] = us_seg
        else:
            stream['us_seg'] = -1
        
        # Get downstream segments indices for the current stream
        idx_ds = stream['idx_ds']
        ds_seg = list()
        for id, other_stream in enumerate(smoothed_streams):
            if other_stream['idx'] == idx_ds:
                ds_seg.append(id)
        if ds_seg:
            stream['ds_seg'] = ds_seg
        else:
            stream['ds_seg'] = -1
    
    # Get segments first and last cross-sections indexes
    for stream in smoothed_streams:
        first_xs_coords = stream['planar_projected_xs_coords'][0]
        last_xs_coords = stream['planar_projected_xs_coords'][-1]
        
        for xs_idx, xs in enumerate(smoothed_xs):
            if xs['planar_projected_xs_coords'] == first_xs_coords:
                stream['first_cs'] = xs_idx
            if xs['planar_projected_xs_coords'] == last_xs_coords:
                stream['last_cs'] = xs_idx
                
    
    
    # Compute streams paths
    # Populate streams_paths with paths starting from headwater streams 
    # --> us_seg == -1
    streams_paths = dict()
    for i, stream in enumerate(smoothed_streams):
        if stream['us_seg'] == -1:
            streams_paths[i] = _compute_streams_paths(i, smoothed_streams)
    
    
    # Calculate curvilinear abscissas for each cross-section in streams paths
    xs_curvilinear_abscissas = {}
    for path_key, path in streams_paths.items():
        path = path[::-1]  # Reverse the path to start from outlet
        xs_coords = list()
        
        # Collect all coordinates for the path
        for stream_id in path:
            xs_coords.extend(reversed(smoothed_streams[stream_id]['planar_projected_xs_coords']))

        # Calculate cumulative distances along the path
        abscissas = []
        x = 0.0
        outlet_coord = xs_coords[0]
        abscissas.append({'coord': outlet_coord, 'x': x})
        for i in range(1, len(xs_coords)):
            x += np.sqrt((xs_coords[i][0] - xs_coords[i-1][0])**2 + (xs_coords[i][1] - xs_coords[i-1][1])**2)
            abscissas.append({'coord': xs_coords[i], 'x': x})

        xs_curvilinear_abscissas[path_key] = abscissas
    
    
    # Extract unique cross sections abscissas
    unique_xs_abscissas = {
        tuple(item['coord']): item['x']
        for path_abscissas in xs_curvilinear_abscissas.values()
        for item in path_abscissas
    }
    
    
    # Create type Segment fortran: a list of dictionaries with the following keys:
        # - first_cs: index of the first cross-section of the segment
        # - last_cs: index of the last cross-section of the segment
        # - ds_seg: index of the downstream segment
        # - us_seg: index of the upstream segment
    seg = list()
    for stream in smoothed_streams:    
        seg.append({'first_cs': stream['first_cs'],
                                'last_cs': stream['last_cs'],
                                'ds_seg': stream['ds_seg'],
                                'us_seg': stream['us_seg']})
    
    # Create type CrossSection fortran: a list of dictionaries with the following keys:
        # - coord: planar coordinates of the cross-section ≡ xs_coords
        # - nlevels: number of discretization levels
        # - level_widths:
        # ...
    cs = list()
    for xs in smoothed_xs:
        cs.append({
            'coord': [xs['planar_projected_xs_coords'][0], xs['planar_projected_xs_coords'][1]],
            'nlevels': 1,
            'level_heights': [0.0],
            'level_widths': [xs['projected_xs_width']],
            'strickler_params': [0.0],
            'bathy': 0.0,  
            'x': unique_xs_abscissas[tuple(xs['planar_projected_xs_coords'])],  
            'ob_levels': [0, 0] ,
            'delta': 0.0,  
            'deltademi': 0.0,
            'y': [0.0]
        })
    
    return streams, smoothed_streams, smoothed_xs, seg, cs





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

