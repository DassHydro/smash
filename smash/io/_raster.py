from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.mw_mesh import MeshDT

import numpy as np
from osgeo import gdal


def _read_windowed_raster(path: str, mesh: MeshDT) -> np.ndarray:

    ds = gdal.Open(path)

    transform = ds.GetGeoTransform()

    xmin = transform[0]
    ymax = transform[3]
    xres = transform[1]
    yres = -transform[5]

    col_off = (mesh.xmin - xmin) / xres
    row_off = (ymax - mesh.ymax) / yres
    
    band = ds.GetRasterBand(1)
    
    nodata = band.GetNoDataValue()
    
    arr = band.ReadAsArray(col_off, row_off, mesh.ncol, mesh.nrow)
    
    arr = np.where(arr == nodata, -99, arr)

    return arr
