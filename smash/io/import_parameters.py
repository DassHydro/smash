from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.enums import Resampling

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
    output_bbox = _get_bbox_from_smash_mesh(mesh)

    xres = mesh.xres
    yres = mesh.yres

    output_crs=rasterio.CRS.from_epsg(mesh.epsg)

    # Open the larger raster
    with rasterio.open(path) as dataset:

        x_scale_factor = dataset.res[0] / xres
        y_scale_factor = dataset.res[1] / yres

        transform=dataset.transform
        height=dataset.height
        width=dataset.width
        crs=dataset.crs

        # resampling first to avoid spatial shifting of the parameters
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * y_scale_factor),
                int(dataset.width * x_scale_factor),
            ),
            resampling=Resampling.nearest,
        )

    #Use a memory dataset
    with rasterio.io.MemoryFile() as memfile:

        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            transform=transform,
            crs=crs
        ) as dataset:
            dataset.write(data[0,:,:], 1)

            new_width = int((output_bbox["right"] - output_bbox["left"]) / yres)
            new_height = int((output_bbox["top"] - output_bbox["bottom"]) / xres)

            new_transform = rasterio.transform.from_bounds(
                west=output_bbox["left"], 
                south=output_bbox["bottom"], 
                east=output_bbox["right"], 
                north=output_bbox["top"], 
                width=new_width, 
                height=new_height)

            # Target array
            new_array = np.empty((new_height, new_width), dtype=np.float32)
            
            #reproject dataset
            rasterio.warp.reproject(
                source=rasterio.band(dataset, 1),
                destination=new_array,
                src_transform=transform,
                src_crs=crs,
                dst_transform=new_transform,
                dst_crs=output_crs,
                resampling=Resampling.nearest
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

    return bbox



def intersection_bbox(bbox1, bbox2):
    """
    
    Description
    -----------
    
    Function which compute the bounding boxes intersection of 2 input bbox. It return the working bbox

    Parameters
    ----------
    
    bbox1: dict() 
        containing the first bbox informations
    bbox2 : dict() 
        containing the second bbox informations
    
    returns
    ----------
    
    dict()
        containing the bbox union

    Examples
    ----------
    
    dataset=gdal_raster_open(filename)  
    possible_bbox=intersection_bbox(bbox,bbox_dataset)  
    
    """
    left = max(bbox1['left'], bbox2['left'])
    bottom = max(bbox1['bottom'], bbox2['bottom'])
    right = min(bbox1['right'], bbox2['right'])
    top = min(bbox1['top'], bbox2['top'])
    if (left < right) and (bottom < top):
        bbox_intersection = {"left": left, "bottom": bottom,
                      "right": right, "top": top}
        return bbox_intersection
    else:
        print("Impossible bounding boxes intersection")
        return {"left": 0, "bottom": 0,
                      "right": 0, "top": 0}



def get_cropped_window_from_bbox_intersection(bbox_intersection,bbox_origin,dx,dy):
    """
    
    Description
    -----------
    
    Function to compute the domain to crop between a bbox intersection (included into bbox_origin) and the origin bbox. This function return a window such as the domain with bbox_intersection can be cropped using the retruned window according the bbox_origin
    
    Parameters
    ----------
    
    bbox_intersection: dict
        A bbox that intersect bbox_origin
    
    bbox_origin: dict
        a bbox from which we want to extract data
    
    dx: float
        size of the grid in the x direction
    
    dy: float
        size of the grid in the y direction
    
    Return
    ------
    
    dict()
        a window dictionnary containing information to crop a matrix: {row_off, col_off, nrows, ncols}
    

    """
    if (bbox_intersection['left']<bbox_origin['left']) or (bbox_intersection['bottom']<bbox_origin['bottom']) or (bbox_intersection['right']>bbox_origin['right']) or (bbox_intersection['top']>bbox_origin['top']):
        print("The domain of bbox_intersection is not included in the domain of bbox_out")
        window= {'row_off': 0, 'col_off': 0, 'nrows': 0, 'ncols': 0}
        return window
    
    col_off = (bbox_intersection["left"] - bbox_origin["left"]) / dx
    row_off = (bbox_origin["top"]- bbox_intersection['top']) / dy
    
    ncols = (bbox_intersection["right"]-bbox_intersection["left"])/dx
    nrows = (bbox_intersection["top"]-bbox_intersection["bottom"])/dy
    
    window = {'row_off': int(row_off), 'col_off': int(
        col_off), 'nrows': int(nrows), 'ncols': int(ncols)}
    
    return window
