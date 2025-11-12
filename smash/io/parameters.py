from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio

from smash._constant import (
    DEFAULT_RR_INITIAL_STATES,
    DEFAULT_RR_PARAMETERS,
    FEASIBLE_RR_INITIAL_STATES,
    FEASIBLE_RR_PARAMETERS,
)

if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.fcore._mwd_mesh import MeshDT
    from smash.util._typing import FilePath, ListLike

__all__ = ["read_grid_parameters", "save_grid_parameters"]


def _write_grid_to_geotiff(
    filename: FilePath,
    data: np.ndarray,
    mesh: MeshDT,
    tags: dict[str, Any],
) -> None:
    metadata = {
        "driver": "GTiff",
        "dtype": "float64",
        "nodata": None,
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": rasterio.CRS.from_epsg(mesh.epsg),
        "transform": rasterio.Affine(mesh.xres, 0.0, mesh.xmin, 0.0, -mesh.yres, mesh.ymax),
    }
    with rasterio.open(filename, "w", compress="lzw", **metadata) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)


def _read_grid_from_geotiff(
    path: str,
    mesh: MeshDT,
    default_value: float,
) -> np.ndarray:
    dst_bbox = rasterio.coords.BoundingBox(
        left=mesh.xmin,
        bottom=mesh.ymax - mesh.nrow * mesh.yres,
        right=mesh.xmin + mesh.ncol * mesh.xres,
        top=mesh.ymax,
    )
    dst_crs = mesh.epsg
    with rasterio.open(path) as ds:
        x_scale_factor = ds.res[0] / mesh.xres
        y_scale_factor = ds.res[1] / mesh.yres
        src_transform = ds.transform
        src_height = ds.height
        src_width = ds.width
        src_crs = ds.crs
        src_bbox = ds.bounds

        # Error if domains are disjoints
        if rasterio.coords.disjoint_bounds(
            (dst_bbox.left, dst_bbox.bottom, dst_bbox.right, dst_bbox.top),
            (src_bbox.left, src_bbox.bottom, src_bbox.right, src_bbox.top),
        ):
            raise ValueError(
                "The domain of the mesh and the domain of the GeoTIFF parameters are disjoint."
                f"{dst_bbox} / {src_bbox}"
            )
        # Warning if domains are partially disjoints
        if (
            dst_bbox.left < src_bbox.left
            or dst_bbox.right > src_bbox.right
            or dst_bbox.bottom < src_bbox.bottom
            or dst_bbox.top > src_bbox.top
        ):
            warnings.warn(
                "The domain of the mesh and the domain of the GeoTIFF parameters are partially disjoint."
                f"{dst_bbox} / {src_bbox}",
                stacklevel=2,
            )
        # Resampling first to avoid spatial shifting of the parameters
        src_data = ds.read(
            out_shape=(
                ds.count,
                int(ds.height * y_scale_factor),
                int(ds.width * x_scale_factor),
            ),
            resampling=rasterio.enums.Resampling.nearest,
        )

    # Use a memory dataset
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=src_height,
            width=src_width,
            count=1,
            dtype=src_data.dtype,
            transform=src_transform,
            crs=src_crs,
        ) as ds:
            ds.write(src_data[0, :, :], 1)

            dst_width = int((dst_bbox.right - dst_bbox.left) / mesh.yres)
            dst_height = int((dst_bbox.top - dst_bbox.bottom) / mesh.xres)

            dst_transform = rasterio.transform.from_bounds(
                west=dst_bbox.left,
                south=dst_bbox.bottom,
                east=dst_bbox.right,
                north=dst_bbox.top,
                width=dst_width,
                height=dst_height,
            )

            # Destination data
            dst_data = np.empty((dst_height, dst_width), dtype=np.float32)

            # Reproject dataset
            rasterio.warp.reproject(
                source=rasterio.band(ds, 1),
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=default_value,
                resampling=rasterio.enums.Resampling.nearest,
            )
    return dst_data


def save_grid_parameters(
    model: Model, path: FilePath, parameters: str | ListLike | None = None, save_active_cell: bool = False
) -> None:
    """
    Save the Model grid parameters (rainfall-runoff parameters and initial states) to GeoTIFF.

    Grid parameters are saved to GeoTIFF format in a specified directory. Relevant metadata
    will be embedded within the GeoTIFF files. Additionally, the active cell mask can
    be saved as a separate GeoTIFF.

    Parameters
    ----------
    model : `Model <smash.Model>`
        Primary data structure of the hydrological model `smash`.

    path : `str`
        Directory path where the GeoTIFF files will be saved. If the directory
        does not exist, it will be created.

    parameters : `str`, `list[str, ...]` or None, default None
        Name of parameters grid to save. Should be one or a sequence of any key of:

        - `Model.rr_parameters <smash.Model.rr_parameters>`
        - `Model.rr_initial_states <smash.Model.rr_initial_states>`

        >>> parameters = "cp"
        >>> parameters = ["cp", "ct", "kexc", "llr"]

        .. note::
            If not given, all parameters in `Model.rr_parameters <smash.Model.rr_parameters>`
            will be saved.

    save_active_cell : `bool`, default False
        Whether or not to save the active cell mask.

    See Also
    --------
    read_grid_parameters: Read the Model grid parameters (rainfall-runoff parameters and initial states)
        from GeoTIFF.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_grid_parameters
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save all `rr_parameters <smash.Model.rr_parameters>`

    >>> save_grid_parameters(model, "rr_parameters_dir")

    Save only a subset of `rr_parameters <smash.Model.rr_parameters>`

    >>> save_grid_parameters(model, "rr_parameters_dir", ["cp", "ct"])

    Save all `rr_parameters <smash.Model.rr_parameters>` and
    `rr_initial_states <smash.Model.rr_initial_states>`

    >>> parameters = list(model.rr_parameters.keys) + list(model.rr_initial_states.keys)
    >>> parameters
    ['ci', 'cp', 'ct', 'kexc', 'llr', 'hi', 'hp', 'ht', 'hlr']
    >>> save_grid_parameters(model, "rr_parameters_initial_states_dir", parameters)

    Save all `rr_parameters <smash.Model.rr_parameters>` and the active cell mask.

    >>> save_grid_parameters(model, "rr_parameters_dir", save_active_cell=True)
    """
    if not os.path.exists(path):
        os.mkdir(path)

    available_rr_parameters = model.rr_parameters.keys
    available_rr_initial_states = model.rr_initial_states.keys
    available_parameters = np.append(available_rr_parameters, available_rr_initial_states)

    if parameters is None:
        parameters = available_rr_parameters
    else:
        if isinstance(parameters, (str, list, tuple, np.ndarray)):
            parameters = np.array(parameters, ndmin=1)
            for i, prmt in enumerate(parameters):
                if prmt in available_parameters:
                    parameters[i] = prmt.lower()
                else:
                    raise ValueError(
                        f"Unknown parameter '{prmt}' at index {i} in parameters. "
                        f"Choices: {available_parameters}"
                    )
        else:
            raise TypeError("parameters must be a str or ListLike type (List, Tuple, np.ndarray)")

    tags = {
        "start_time": model.setup.start_time,
        "end_time": model.setup.end_time,
        "dt": model.setup.dt,
        "structure": model.setup.structure,
        "snow_module": model.setup.snow_module,
        "hydrological_module": model.setup.hydrological_module,
        "routing_module": model.setup.routing_module,
        "epsg": model.mesh.epsg,
    }
    for prmt in parameters:
        filename = os.path.join(path, f"{prmt}.tif")
        data = (
            model.get_rr_parameters(prmt)
            if prmt in model.rr_parameters.keys
            else model.get_rr_initial_states(prmt)
        )
        _write_grid_to_geotiff(
            filename,
            data,
            model.mesh,
            tags,
        )

    if save_active_cell:
        _write_grid_to_geotiff(
            os.path.join(path, "active_cell.tif"),
            model.mesh.active_cell,
            model.mesh,
            tags,
        )


def read_grid_parameters(model: Model, path: FilePath, parameters: str | ListLike | None = None) -> None:
    """
    Read the Model grid parameters (rainfall-runoff parameters and initial states) from GeoTIFF.

    Grid parameters are read from GeoTIFF format in a specified directory.

    Parameters
    ----------
    model : `Model <smash.Model>`
        Primary data structure of the hydrological model `smash`.

    path : `str`
        Directory path where the GeoTIFF files will be read. If the directory
        does not exist, an error will be raised.

    parameters : `str`, `list[str, ...]` or None, default None
        Name of parameters grid to read. Should be one or a sequence of any key of:

        - `Model.rr_parameters <smash.Model.rr_parameters>`
        - `Model.rr_initial_states <smash.Model.rr_initial_states>`

        >>> parameters = "cp"
        >>> parameters = ["cp", "ct", "kexc", "llr"]

        .. note::
            If not given, all parameters in `Model.rr_parameters <smash.Model.rr_parameters>`
            will be read.

    See Also
    --------
    save_grid_parameters: Save the Model grid parameters (rainfall-runoff parameters and initial states)
        to GeoTIFF.

    Examples
    --------
    >>> from smash.factory import load_dataset
    >>> from smash.io import save_grid_parameters, read_grid_parameters
    >>> setup, mesh = load_dataset("cance")
    >>> model = smash.Model(setup, mesh)

    Save all `rr_parameters <smash.Model.rr_parameters>` and
    `rr_initial_states <smash.Model.rr_initial_states>`

    >>> parameters = list(model.rr_parameters.keys) + list(model.rr_initial_states.keys)
    >>> parameters
    ['ci', 'cp', 'ct', 'kexc', 'llr', 'hi', 'hp', 'ht', 'hlr']
    >>> save_grid_parameters(model, "rr_parameters_initial_states_dir", parameters)

    Read all `rr_parameters <smash.Model.rr_parameters>`

    >>> read_grid_parameters(model, "rr_parameters_initial_states_dir")

    Read only a subset of `rr_parameters <smash.Model.rr_parameters>`

    >>> read_grid_parameters(model, "rr_parameters_initial_states_dir", ["cp", "ct"])

    Read all `rr_parameters <smash.Model.rr_parameters>` and
    `rr_initial_states <smash.Model.rr_initial_states>`

    >>> parameters
    ['ci', 'cp', 'ct', 'kexc', 'llr', 'hi', 'hp', 'ht', 'hlr']
    >>> read_grid_parameters(model, "rr_parameters_initial_states_dir", parameters)
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Parameters directory '{path}' does not exist")

    available_rr_parameters = model.rr_parameters.keys
    available_rr_initial_states = model.rr_initial_states.keys
    available_parameters = np.append(available_rr_parameters, available_rr_initial_states)

    if parameters is None:
        parameters = available_rr_parameters
    else:
        if isinstance(parameters, (str, list, tuple, np.ndarray)):
            parameters = np.array(parameters, ndmin=1)
            for i, prmt in enumerate(parameters):
                if prmt in available_parameters:
                    parameters[i] = prmt.lower()
                else:
                    raise ValueError(
                        f"Unknown parameter '{prmt}' at index {i} in parameters. "
                        f"Choices: {available_parameters}"
                    )
        else:
            raise TypeError("parameters must be a str or ListLike type (List, Tuple, np.ndarray)")

    for prmt in parameters:
        prmt_is_rr_parameters = prmt in model.rr_parameters.keys
        filename = os.path.join(path, f"{prmt}.tif")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"'{prmt}' parameter file '{filename}' does not exist")
        default = DEFAULT_RR_PARAMETERS[prmt] if prmt_is_rr_parameters else DEFAULT_RR_INITIAL_STATES[prmt]
        data = _read_grid_from_geotiff(filename, model.mesh, default)
        lb, ub = FEASIBLE_RR_PARAMETERS[prmt] if prmt_is_rr_parameters else FEASIBLE_RR_INITIAL_STATES[prmt]
        # Depending on how parameters have been written and which no_data value has been
        # chosen, smash will raise an error if the parameters are not included in the
        # feasible domain. We set any unfeasible value to the default one.
        data = np.where(np.logical_or(data <= lb, data >= ub), default, data)
        model.set_rr_parameters(prmt, data) if prmt_is_rr_parameters else model.set_rr_initial_states(
            prmt, data
        )
