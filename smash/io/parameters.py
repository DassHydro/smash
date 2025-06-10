from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import rasterio

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import FilePath


def export_parameters(model: Model, path: FilePath, parameters: list | None = None):
    """
    Export the smash parameters to tif format in a specified directory.
    Relevant metadata will be embedded within the GeoTIFF files. Additionally,
    the `mesh.active_cell` array will be saved as a separate GeoTIFF.

    Parameters
    ----------
    model : smash.Model
        A SMASH model instance containing setup, mesh, and parameter data.
    path : str
        Directory path where the GeoTIFF files will be saved. If the directory
        does not exist, it will be created.
    """
    if not os.path.exists(path):
        os.mkdir(path)

    if parameters is None:
        param_keys = model.rr_parameters.keys
    else:
        for p in parameters:
            if p not in model.rr_parameters.keys:
                raise ValueError(
                    f"Parameters '{p}' is not a valid parameters name."
                    "Choice are: {list(model.rr_parameters.keys)}."
                )

        param_keys = np.array(parameters)

    for param in param_keys:
        array = model.rr_parameters.values[:, :, np.argwhere(param_keys == param).item()]

        bbox = [
            model.mesh.xmin,
            model.mesh.xmin + array.shape[1] * model.mesh.xres,
            model.mesh.ymax - array.shape[0] * model.mesh.yres,
            model.mesh.ymax,
        ]

        _write_array_to_geotiff(
            filename=os.path.join(path, param + ".tif"),
            array=array,
            xmin=model.mesh.xmin,
            ymax=model.mesh.ymax,
            xres=model.mesh.xres,
            yres=model.mesh.yres,
            epsg=model.mesh.epsg,
            tags={
                "dt": model.setup.dt,
                "hydrological_module": model.setup.hydrological_module,
                "snow_module": model.setup.snow_module,
                "routing_module": model.setup.routing_module,
                "bounding_box": bbox,
                "epsg": model.mesh.epsg,
            },
        )

    _write_array_to_geotiff(
        filename=os.path.join(path, "active_cell.tif"),
        array=model.mesh.active_cell,
        xmin=model.mesh.xmin,
        ymax=model.mesh.ymax,
        xres=model.mesh.xres,
        yres=model.mesh.yres,
        epsg=model.mesh.epsg,
        tags={
            "dt": model.setup.dt,
            "hydrological_module": model.setup.hydrological_module,
            "snow_module": model.setup.snow_module,
            "routing_module": model.setup.routing_module,
            "bounding_box": bbox,
            "epsg": model.mesh.epsg,
        },
    )


def _write_array_to_geotiff(
    filename: FilePath = None,
    array: np.ndarray = None,
    xmin: float = 0,
    ymax: float = 0.0,
    xres: float = 0.0,
    yres: float = 0.0,
    epsg: int = -99,
    tags: dict | None = None,
):
    metadata = {
        "driver": "GTiff",
        "dtype": "float64",
        "nodata": None,
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "crs": rasterio.CRS.from_epsg(epsg),
        "transform": rasterio.Affine(xres, 0.0, xmin, 0.0, -yres, ymax),
    }

    _rasterio_write_tiff(filename=filename, matrix=array, metadata=metadata, tags=tags)


def _rasterio_write_tiff(
    filename: FilePath = "mygeotiff.tif",
    matrix: np.ndarray = np.zeros(10),
    metadata: dict | None = None,
    tags: dict | None = None,
):
    if metadata is None:
        metadata = {}

    if tags is None:
        tags = {}

    with rasterio.Env():
        with rasterio.open(filename, "w", compress="lzw", **metadata) as dst:
            dst.write(matrix, 1)
            dst.update_tags(**tags)
