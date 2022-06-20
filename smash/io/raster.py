from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.m_mesh import MeshDT

import numpy as np
import rasterio as rio
from osgeo import gdal


def read_windowed_raster(path: str, mesh: MeshDT) -> np.ndarray:

    ds = rio.open(path)

    transform = ds.transform

    xmin = transform[2]
    ymax = transform[5]
    xres = transform[0]
    yres = -transform[4]

    col_off = (mesh.xmin - xmin) / xres
    row_off = (ymax - mesh.ymax) / yres

    window = rio.windows.Window(
        col_off=col_off, row_off=row_off, width=mesh.ncol, height=mesh.nrow
    )

    return ds.read(1, window=window)


def read_windowed_raster_gdal(path: str, mesh: MeshDT) -> np.ndarray:

    ds = gdal.Open(path)
    
    transform = ds.GetGeoTransform()
    
    xmin = transform[0]
    ymax = transform[3]
    xres = transform[1]
    yres = -transform[5]
    
    col_off = (mesh.xmin - xmin) / xres
    row_off = (ymax - mesh.ymax) / yres
    
    return ds.GetRasterBand(1).ReadAsArray(col_off, row_off, mesh.ncol, mesh.nrow)
    
