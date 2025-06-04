from __future__ import annotations

import rasterio
import numpy as np
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import FilePath


def export_parameters(model: Model, path: FilePath):
    """
    Export the smash parameters to tif format in a specified directory. Useful metadata will be saved inside the geotiff. The `mesh.active_cell` array is also saved in a separate geotif.
    Parameters:
    -----------
    model: smash.Model
        A Smash model object with setup, mesh and parameters.
    path: str
        The directory path where to saved the geotif. If not exist, the directory will be created.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    
    param_keys=model.rr_parameters.keys

    for param in param_keys:

        #array=model.rr_parameters.values[:,:,param_keys.index(param)]
        array=model.rr_parameters.values[:,:,np.argwhere(param_keys==param).item()]

        bbox=[model.mesh.xmin, 
              model.mesh.xmin+array.shape[1]*model.mesh.xres,
              model.mesh.ymax-array.shape[0]*model.mesh.yres,
              model.mesh.ymax,
              ]

        _write_array_to_geotiff(
                                filename=os.path.join(path, param+".tif"), 
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
                                    }
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
                                 }
                            )


def _write_array_to_geotiff(filename: FilePath = None,
                           array: np.ndarray = None,
                           xmin: float = 0,
                           ymax: float = 0.,
                           xres: float = 0.,
                           yres: float = 0.,
                           epsg: int = -99,
                           tags: dict = {}
                           ):

    metadata={'driver': 'GTiff', 
              'dtype': 'float64', 
              'nodata': None, 
              'width': array.shape[1], 
              'height': array.shape[0], 
              'count': 1, 
              'crs': rasterio.CRS.from_epsg(epsg),
              'transform': rasterio.Affine(xres, 
                                           0.0, 
                                           xmin, 
                                           0.0, 
                                           -yres,
                                           ymax)}

    _rasterio_write_tiff(filename=filename, matrix=array, metadata=metadata, tags=tags)


def _rasterio_write_tiff(filename: FilePath ="mygeotiff.tif",
                        matrix: np.ndarray =np.zeros(10),
                        metadata: dict ={},
                        tags: dict = {},
                        ):

    with rasterio.Env():
        with rasterio.open(filename, 'w', compress='lzw', **metadata) as dst:
            dst.write(matrix, 1)
            dst.update_tags(**tags)

