

from pathlib import Path
import rasterio
from rasterio import features
from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import pandas as pd

import pytest

import pyflwdir
import smash



class OverlapError(Exception):
    """Exception raised when overlapping sub-basins are detected."""

    def __init__(self, message="Sub-basins overlap detected."):
        self.message = message
        super().__init__(self.message)


def vectorize(data, nodata, transform, crs, name="value"):
    """Convenience method for vectorizing a raster"""
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    return gdf


def check_overlap(flwdir_raster, inflows_idxs, output_file):
    """Check if inflows sub-basins overlap."""

    with rasterio.open(flwdir_raster) as src:
        flwdir = src.read(1)
        flw = pyflwdir.from_array(
            flwdir, ftype="d8", transform=src.transform, latlon=src.crs.is_geographic, cache=True)

        flwdir_accu_areas = flw.accuflux(data=flw.area)

        x_inflows, y_inflows = flw.xy(inflows_idxs)
        inflows_coords = [Point(x, y) for x, y in zip(x_inflows, y_inflows)]

        df_inflows_sb = pd.DataFrame()
        for i, centroid in enumerate(inflows_coords):
            subbasins = flw.basins(xy=(centroid.x, centroid.y))
            df = vectorize(subbasins.astype(np.int32),
                           0, src.transform, src.crs)
            df['subbasin'] = inflows_idxs[i]
            df['outlet_lon'] = centroid.x
            df['outlet_lat'] = centroid.y
            outlet_col, outlet_row = ~src.transform * \
                (centroid.x, centroid.y)
            outlet_row, outlet_col = int(
                outlet_row), int(outlet_col)
            outlet_flow_acc = flwdir_accu_areas[outlet_row, outlet_col]
            df['flow_acc'] = outlet_flow_acc
            df_inflows_sb = pd.concat([df_inflows_sb, df], ignore_index=True)

        gdf_inflows_sb = gpd.GeoDataFrame(df_inflows_sb, geometry='geometry')
        # delete column value of def vectorize
        gdf_inflows_sb.drop('value', axis=1, inplace=True)

    # Check sub-basins (polygons) overlapping
    # keep_geom_type=False : produces other geometry types: LineString and MultiLineString due to the intersection operation
    inflows_sb_overlay = gdf_inflows_sb.overlay(
        gdf_inflows_sb, how='intersection', keep_geom_type=False)
    # Filter to keep only Polygon geometries to avoid warning if keep_geom_type=True 
    inflows_sb_overlay = inflows_sb_overlay[inflows_sb_overlay.geometry.type == 'Polygon']
    
    inflows_sb_intersections = inflows_sb_overlay[inflows_sb_overlay['subbasin_1']
                                                  != inflows_sb_overlay['subbasin_2']]
    inflows_sb_intersections.reset_index(drop=True, inplace=True)

    if not inflows_sb_intersections.empty:
        raise OverlapError()
    else:

        inflows_coverage = round(
            gdf_inflows_sb.flow_acc.sum()/np.max(flwdir_accu_areas)*100)
        gdf_inflows_sb.to_file(output_file)

    return inflows_coverage, len(flw.idxs_seq)


def write_raster(flwdir_raster, cells_rows_cols, output_file):

    with rasterio.open(flwdir_raster) as src:
        flwdir = src.read(1)
        metadata = src.meta
        flw = pyflwdir.from_array(
            flwdir, ftype="d8", transform=src.transform, latlon=src.crs.is_geographic, cache=True)

        mask_array = np.zeros(flw.shape, dtype=np.uint8)
        rows, cols = cells_rows_cols
        mask_array[rows, cols] = 1
        metadata.update({"nodata": 0})
        with rasterio.open(output_file, "w", **metadata) as out_ds:
            out_ds.write(mask_array, 1)


@pytest.fixture(scope="module")
def dataset_info():
    """Fixture to set up dataset paths and output directories."""
    base_dir = Path(__file__).parent.parent
    datasets = [
        ("argens_upscale_1km", "dataset/argens_upscale_1km", "HYDRO_ARGENS_BD_CARTHAGE_COMPLETE.shp", "argens_upscaled_1km_flow_directions.tif"),
        ("argens_upscale_500m", "dataset/argens_upscale_500m", "HYDRO_ARGENS_BD_CARTHAGE_COMPLETE.shp", "argens_upscaled_500m_flow_directions.tif"),
        ("gapeau_upscale_1km", "dataset/gapeau_upscale_1km", "HYDRO_GAPEAU_BD_CARTHAGE_COMPLETE.shp", "gapeau_upscaled_1km_flow_directions.tif"),
        ("gapeau_upscale_500m", "dataset/gapeau_upscale_500m", "HYDRO_GAPEAU_BD_CARTHAGE_COMPLETE.shp", "gapeau_upscaled_500m_flow_directions.tif"),
    ]

    dataset_paths = {}
    outputs_dirs = {}

    for dataset_name, dataset_dir, river_line_file, flw_dir_file in datasets:
        dataset_paths[dataset_name] = {
            "path": base_dir / dataset_dir,
            "river_line_file": river_line_file,
            "flw_dir_file": flw_dir_file,
        }
        outputs_dirs[dataset_name] = base_dir / "prepro_test_results" / dataset_name
        outputs_dirs[dataset_name].mkdir(parents=True, exist_ok=True)

    return dataset_paths, outputs_dirs



@pytest.mark.parametrize("dataset_name", [
    "argens_upscale_1km",
    "argens_upscale_500m",
    "gapeau_upscale_1km",
    "gapeau_upscale_500m"
])
def test_inflows_extraction(dataset_name, dataset_info):
    dataset_paths, outputs_dirs = dataset_info

    dataset = dataset_paths[dataset_name]
    outputs_dir = outputs_dirs[dataset_name]

    river_line = dataset["path"] / dataset["river_line_file"]
    flw_dir_file = dataset["path"] / dataset["flw_dir_file"]

    mesh = smash.factory.generate_mesh_grid_vect(
    flw_dir_file,
    river_line,
    2.79,
    0.48)
    
    flow_path = mesh['flow_path_rows_cols']
    flow_path_idxs = mesh['flow_path_idxs']
    inflows_idxs = mesh['inflows_idxs']
    upstream_inflows = mesh['upstream_inflows_rows_cols']
    lateral_inflows = mesh['lateral_inflows_rows_cols']


    inflows_coverage, flw_size = check_overlap(
        flw_dir_file, inflows_idxs, outputs_dir / f"inflows_subbasins_{dataset_name}.shp")

    flow_path_cells_coverage = round(len(flow_path_idxs)/ flw_size * 100)
    
    total_coverage = inflows_coverage + flow_path_cells_coverage

    write_raster(flw_dir_file, flow_path, outputs_dir /
                 f"flow_path_{dataset_name}.tif")
    write_raster(flw_dir_file, upstream_inflows, outputs_dir /
                 f"upstream_inflows_{dataset_name}.tif")
    write_raster(flw_dir_file, lateral_inflows, outputs_dir /
                 f"lateral_inflows_{dataset_name}.tif")

    assert total_coverage == 100