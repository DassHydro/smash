from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.enums import Resampling

from smash._constant import DEFAULT_RR_PARAMETERS, FEASIBLE_RR_PARAMETERS

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
                path=os.path.join(path_to_parameters, param + ".tif"),
                mesh=model.mesh,
                default_value=DEFAULT_RR_PARAMETERS[param],
            )

            # depending how parameters has been written and which no_data value hav been
            # chosen, Smash will raise an error if the parameter are not included in the
            # FEASIBLE_RR_PARAMETERS domain.
            # We just set its value to the default one.
            mask = np.where(cropped_param < FEASIBLE_RR_PARAMETERS[param][0])
            cropped_param[mask] = DEFAULT_RR_PARAMETERS[param]

            mask = np.where(cropped_param > FEASIBLE_RR_PARAMETERS[param][1])
            cropped_param[mask] = DEFAULT_RR_PARAMETERS[param]

            pos = np.argwhere(list_param == param).item()
            model.rr_parameters.values[:, :, pos] = cropped_param

        else:
            raise ValueError(f"Missing parameter {param} in {path_to_parameters}")


def _rasterio_read_param(path: FilePath, mesh: MeshDT, default_value: float = 0.0):
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
    output_bbox = _get_bbox_from_smash_mesh(mesh)

    xres = mesh.xres
    yres = mesh.yres

    # requiring merge #440, as workaround we test if attr epsg exist.
    if hasattr(mesh, "epsg"):
        output_crs = rasterio.CRS.from_epsg(mesh.epsg)
    else:
        output_crs = rasterio.CRS.from_epsg(2154)

    # Open the larger raster
    with rasterio.open(path) as dataset:
        x_scale_factor = dataset.res[0] / xres
        y_scale_factor = dataset.res[1] / yres

        transform = dataset.transform
        height = dataset.height
        width = dataset.width
        crs = dataset.crs
        input_bbox = dataset.bounds

        # resampling first to avoid spatial shifting of the parameters
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * y_scale_factor),
                int(dataset.width * x_scale_factor),
            ),
            resampling=Resampling.nearest,
        )

    if rasterio.coords.disjoint_bounds(
        (output_bbox.left, output_bbox.bottom, output_bbox.right, output_bbox.top),
        (input_bbox.left, input_bbox.bottom, input_bbox.right, input_bbox.top),
    ):
        raise ValueError(
            "The domain of the mesh and the domain of the Geotiff parameters are disjoint."
            f"{output_bbox} / {input_bbox}"
        )

    # check if ouput bbox are included in bounds, otherwise print a warning
    if (
        output_bbox.left < input_bbox.left
        or output_bbox.right > input_bbox.right
        or output_bbox.bottom < input_bbox.bottom
        or output_bbox.top > input_bbox.top
    ):
        print("</> The boundaries of the Smash domain exceed the boundaries of the parameters domain.")

    # Use a memory dataset
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs=crs,
        ) as dataset:
            dataset.write(data[0, :, :], 1)

            new_width = int((output_bbox.right - output_bbox.left) / yres)
            new_height = int((output_bbox.top - output_bbox.bottom) / xres)

            new_transform = rasterio.transform.from_bounds(
                west=output_bbox.left,
                south=output_bbox.bottom,
                east=output_bbox.right,
                north=output_bbox.top,
                width=new_width,
                height=new_height,
            )

            # Target array
            new_array = np.empty((new_height, new_width), dtype=np.float32)

            # reproject dataset
            rasterio.warp.reproject(
                source=rasterio.band(dataset, 1),
                destination=new_array,
                src_transform=transform,
                src_crs=crs,
                dst_transform=new_transform,
                dst_crs=output_crs,
                dst_nodata=default_value,
                resampling=Resampling.nearest,
            )

    return new_array


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

    output_bbox = rasterio.coords.BoundingBox(bbox["left"], bbox["bottom"], bbox["right"], bbox["top"])

    return output_bbox
