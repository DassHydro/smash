from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_mesh import MeshDT
    from smash.util._typing import FilePath


def import_parameters(model: Model, path_to_parameters: FilePath):
    """
    Description
    -----------
    Read a geotif, resample if necessarry then clip it on the bouning box of the smash mesh

    Parameters
    ----------
    model: object
        SMASH model object
    path_to_parameters: str
        Path to the directory which contain the geotiff files (parameters)

    return
    ------
    np.ndarray
        The data clipped on the SMASH bounding box
    """
    list_param = model.rr_parameters.keys

    for param in list_param:
        if os.path.exists(os.path.join(path_to_parameters, param + ".tif")):
            cropped_param = _rasterio_read_param(
                path=os.path.join(path_to_parameters, param + ".tif"), mesh=model.mesh
            )

            pos = np.argwhere(list_param == param).item()
            model.rr_parameters.values[:, :, pos] = cropped_param

        else:
            raise ValueError(f"Missing parameter {param} in {path_to_parameters}")


def _rasterio_read_param(path: FilePath, mesh: MeshDT):
    """
    Description
    -----------
    Read a geotif, resample if necessarry then clip it on the bouning box of the smash mesh

    Parameters
    ----------
    path: str
        Path to a geotiff file.
    mesh: object
        object of the smash mesh

    return
    ------
    np.ndarray
        The data clipped on the SMASH bounding box
    """
    bounds = _get_bbox_from_smash_mesh(mesh)
    xres = mesh.xres
    yres = mesh.yres

    # Open the larger raster
    with rasterio.open(path) as dataset:
        x_scale_factor = dataset.res[0] / xres
        y_scale_factor = dataset.res[1] / yres

        # resampling first to avoid spatial shifting of the parameters
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * y_scale_factor),
                int(dataset.width * x_scale_factor),
            ),
            resampling=Resampling.nearest,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
        )

        # Get a window that corresponds to the smaller raster's bounds
        window = from_bounds(**bounds, transform=transform)

        # Read the data from the large raster using the bbox of small raster
        output_resampled = data[
            0,
            int(window.row_off) : int(window.row_off + window.height),
            int(window.col_off) : int(window.col_off + window.width),
        ]

    return output_resampled


def _get_bbox_from_smash_mesh(mesh):
    """
    Description
    -----------
    Compute the bbox from a Smash mesh dictionary

    Parameters
    ----------
    mesh: object
        object of the smash mesh

    return
    ------
    dict()
        the bounding box of the smash mesh
    """

    if hasattr(mesh, "xres") and hasattr(mesh, "yres"):
        dx = mesh.xres
        dy = mesh.yres
    else:
        dx = np.mean(mesh.dx)
        dy = np.mean(mesh.dy)

    if hasattr(mesh, "ncol") and hasattr(mesh, "nrow"):
        ncol = mesh.ncol
        nrow = mesh.nrow
    else:
        nrow = mesh.active_cell.shape[0]
        ncol = mesh.active_cell.shape[1]

    left = mesh.xmin
    right = mesh.xmin + ncol * dx
    bottom = mesh.ymax - nrow * dy
    top = mesh.ymax
    bbox = {"left": left, "bottom": bottom, "right": right, "top": top}

    return bbox
