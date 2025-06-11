from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import zoom
import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling


if TYPE_CHECKING:
    from smash.core.model.model import Model
    from smash.util._typing import FilePath
    from smash.fcore._mwd_mesh import MeshDT


def import_parameters(model: Model, path_to_parameters: FilePath, method: str = "rasterio"):

    list_param=model.rr_parameters.keys

    bbox_out=_get_bbox_from_smash_mesh(model.mesh)

    res_out={"dx":np.mean(model.mesh.dx),
            "dy":np.mean(model.mesh.dy)
            }

    for param in list_param:

        if os.path.exists(os.path.join(path_to_parameters,param+".tif")):

            if method=="rasterio":
                cropped_param=rasterio_read_param(path= os.path.join(path_to_parameters, param + ".tif"), 
                                                  mesh=model.mesh)
            elif method=="scipy-zoom":
                tif_data, bbox_in, res_in =_read_geotiff(os.path.join(path_to_parameters,param+".tif"))

                _check_bbox_consistency(bbox_out, bbox_in)

                cropped_param = _crop_array(tif_data,
                                           bbox_in,
                                           res_in,
                                           bbox_out,
                                           res_out,
                                           order=0,
                                           cval=-99.,
                                           grid_mode=True)

            pos=np.argwhere(list_param==param).item()
            model.rr_parameters.values[:,:,pos]=cropped_param

        else:
            raise ValueError(f"Error: in load_param_from_tiffformat, missing parameter {param} in {path_to_parameters}")



def rasterio_read_param(path: FilePath, mesh: MeshDT):

    bounds=_get_bbox_from_smash_mesh(mesh)
    xres=mesh.xres
    yres=mesh.yres

    # Open the larger raster
    with rasterio.open(path) as dataset:

        x_scale_factor=dataset.res[0]/xres
        y_scale_factor=dataset.res[1]/yres

        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * y_scale_factor),
                int(dataset.width * x_scale_factor)
            ),
            resampling=Resampling.nearest,
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        # Get a window that corresponds to the smaller raster's bounds
        window = from_bounds(**bounds, transform=transform)

        # Read the data from the large raster using the bbox of small raster
        output_resampled=data[0, int(window.row_off):int(window.row_off+window.height), int(window.col_off):int(window.col_off+window.width)]
    
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

    if hasattr(mesh,'xres') and hasattr(mesh,'yres'):
        dx = mesh.xres
        dy = mesh.yres
    else:
        dx = np.mean(mesh.dx)
        dy = np.mean(mesh.dy)

    if hasattr(mesh,'ncol') and hasattr(mesh,'nrow'):
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


def _check_bbox_consistency(bbox_model_active_cell, bbox_param):
    
    if bbox_model_active_cell["left"]<bbox_param["left"]:
        print(f"Warning: Model domain is larger than the domain of the parameter. {bbox_model_active_cell['left']}<{bbox_param['left']} (bbox model left < bbox param left). Expect lacuna (-99.) in model parameters" )
        
    if bbox_model_active_cell["right"]>bbox_param["right"]:
        print(f"Warning: Model domain is larger than the domain of the parameter. {bbox_model_active_cell['right']}>{bbox_param['right']} (bbox model left < bbox param left). Expect lacuna (-99.) in model parameters" )
        
    if bbox_model_active_cell["bottom"]<bbox_param["bottom"]:
        print(f"Warning: Model domain is larger than the domain of the parameter. {bbox_model_active_cell['bottom']}<{bbox_param['bottom']} (bbox model left < bbox param left). Expect lacuna (-99.) in model parameters" )
    
    if bbox_model_active_cell["top"]<bbox_param["top"]:
        print(f"Warning: Model domain is larger than the domain of the parameter. {bbox_model_active_cell['top']}>{bbox_param['top']} (bbox model left < bbox param left). Expect lacuna (-99.) in model parameters" )


def _intersection_bbox(bbox1, bbox2):
    """
    Description
    -----------
    Function which compute the bounding boxe intersection of two bounding boxes. It return the 'working bbox'.

    Parameters
    ----------
    bbox1: dict() 
        containing the first bbox informations
    bbox2 : dict() 
        containing the second bbox informations

    returns
    -------
    dict()
        containing the bbox union

    Examples
    --------
    dataset=gdal_raster_open(filename)  
    possible_bbox=_intersection_bbox(bbox,bbox_dataset)  
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


def _get_mesh_window_from_bbox(mesh, bbox):
    """
    Function to get the mesh window from a defined bbox

    Parameters
    ----------
    dataset: gdal object
    bbox : dict containing the bbox

    returns
    -------
    dic containing the computed windows

    Examples
    --------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    window=_get_mesh_window_from_bbox(dataset,bbox_dataset)

    """

    if 'xres' in mesh and 'yres' in mesh:
         dx = mesh['xres']
         dy = mesh['yres']
    else:
         dx = np.mean(mesh['dx'])
         dy = np.mean(mesh['dy'])

    col_off = (bbox["left"] - mesh['xmin']) / dx
    row_off = (mesh['ymax'] - bbox["top"]) / dy
    ncols = (bbox["right"]-bbox["left"])/dx
    nrows = (bbox["top"]-bbox["bottom"])/dy

    if (col_off < 0) or (row_off < 0):
        raise Exception(
            'The requested bounding box exceeds the limits of the raster domain.')

    window = {'row_off': int(row_off), 'col_off': int(
        col_off), 'nrows': int(nrows), 'ncols': int(ncols)}

    return window


def _get_cropped_window_from_bbox(bbox_new,
                                  bbox_origin,
                                  dx,
                                  dy
                                  ):
    """
    
    Description
    -----------
    
    Function to compute the domain to crop between two bbox . This function return a window such that the domain defined by bbox_new can be cropped from the domain defined by bbox_origin
    
    Parameters
    ----------
    
    bbox_new: dict
        A new bbox that intersect bbox_origin
    
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
    if ((bbox_new['left']<bbox_origin['left']) or
        (bbox_new['bottom']<bbox_origin['bottom']) or
        (bbox_new['right']>bbox_origin['right']) or
        (bbox_new['top']>bbox_origin['top'])):

        raise ValueError(f"The domain of bbox_new {bbox_new} is not included in the domain of bbox_origin {bbox_origin}")

    col_off = (bbox_new["left"] - bbox_origin["left"]) / dx
    row_off = (bbox_origin["top"]- bbox_new['top']) / dy

    ncols = (bbox_new["right"]-bbox_new["left"])/dx
    nrows = (bbox_new["top"]-bbox_new["bottom"])/dy

    window = {'row_off': int(row_off), 'col_off': int(
        col_off), 'nrows': int(nrows), 'ncols': int(ncols)}

    return window


def _read_geotiff(path=""):

    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")

    with rasterio.open(path) as ds:
        bbox={"left": ds.bounds.left,
              "bottom": ds.bounds.bottom,
              "right": ds.bounds.right,
              "top": ds.bounds.top
              }

        res={"dx": ds.res[0],
             "dy": ds.res[1],
            }

        data=ds.read(indexes=1)

    return data, bbox, res


def _crop_array(array,
                bbox_in,
                res_in,
                bbox_out,
                res_out,
                order=0,
                cval=-99.,
                grid_mode=True
                ):
    """
    
    Description
    --------------

    Crop a part of a numpy array with an input bbox and resolution to a new bbox with a new resolution

    Parameters
    ----------

    array : numpy.array()
        Input gridded numpy array shape=(n,m)

    bbox_in : dict()
        bounding box of the input array. Dictionnary {"left":,"top":,"right":,"bottom":}

    res_in : dict()
        resolution of the input array in x and y direction. Disctionnary {"dx":, "dy":}

    bbox_out : dict()
        bounding box of the output array. Dictionnary {"left":,"top":,"right":,"bottom":}

    res_out : dict()
        resolution of the output array in x and y direction. Disctionnary {"dx":, "dy":}

    order : int()
        order of the resampling cubic interpolation

    cval : float() | np.nan
        fill value for the extended boundaries

    grid_mode : bool()
        True | False. if True coordinate start from the edge of the cell. If False coordinate starts from the center of the cell.  

    Return
    ------

    numpy.array()
        Cropped and resampled array according bbox_out and res_out  

    """

    #intersection bbox
    bbox_intersection=_intersection_bbox(bbox_in, bbox_out)
    
    #---------------------- make a coarse clip of the domain  -------

    #1- crop array on bbox_intersection+-res_in at res in: shrink the domain, speed up futur resampling on large domain ?
    if res_in['dx']>=res_out['dx'] and res_in['dy']>=res_out['dy']:
        res_shrinked={"dx":res_in['dx'],"dy":res_in['dy']}
    elif res_in['dx']<res_out['dx'] and res_in['dy']<res_out['dy']:
        res_shrinked={"dx":res_out['dy'],"dy":res_out['dy']}
    elif res_in['dx']<res_out['dx'] and res_in['dy']>=res_out['dy']:
        res_shrinked={"dx":res_out['dx'],"dy":res_in['dy']}
    elif res_in['dx']>=res_out['dx'] and res_in['dy']<res_out['dy']:
        res_shrinked={"dx":res_in['dx'],"dy":res_out['dy']}

    bbox_in_shrinked={
        "left":bbox_in["left"] +
            int(max(bbox_intersection["left"]-bbox_in["left"],0)/res_shrinked["dx"])*res_shrinked["dx"],
        "right":bbox_in["right"] -
            int(max(bbox_in["right"]-bbox_intersection["right"],0)/res_shrinked["dx"])*res_shrinked["dx"],
        "bottom":bbox_in["bottom"] +
            int(max(bbox_intersection["bottom"]-bbox_in["bottom"],0)/res_shrinked["dy"])*res_shrinked["dy"],
        "top":bbox_in["top"] -
            int(max(bbox_in["top"]-bbox_intersection["top"],0)/res_shrinked["dy"])*res_shrinked["dy"],
    }
    bbox_intersection_shrinked=_intersection_bbox(bbox_in, bbox_in_shrinked)

    windows_wrap=_get_mesh_window_from_bbox(mesh={'xmin':bbox_in["left"],
                                                  'ymax':bbox_in["top"],
                                                  'xres':res_shrinked["dx"],
                                                  'yres':res_shrinked["dy"],
                                                  },
                                            bbox=bbox_intersection_shrinked)

    #Erase input array and bbox_in
    array=array[windows_wrap['row_off']:windows_wrap['row_off']+windows_wrap["nrows"],
                windows_wrap['col_off']:windows_wrap['col_off']+windows_wrap["ncols"]]

    bbox_in=bbox_intersection_shrinked

    #--------------------------------------------------------------------

    #2- resample the array to res_out 
    resampled_array=_resample_array(array,
                                    res_in=res_in,
                                    res_out=res_out,
                                    order=order,
                                    cval=cval,
                                    grid_mode=grid_mode
                                    )

    #3- crop the array on the intersection of bbox_in and bbox_out
    #window of bbox_intersection in the domain of bbox_in (bbox_prcp, matrix to read)
    window_intersection=_get_cropped_window_from_bbox(bbox_intersection,
                                                      bbox_in,
                                                      res_out["dx"],
                                                      res_out["dy"]
                                                      )

    #reading the part of the matrix (array_in)
    cropped_array = resampled_array[window_intersection['row_off']:window_intersection['row_off']+window_intersection["nrows"],
                                    window_intersection['col_off']:window_intersection['col_off']+window_intersection["ncols"]]

    #allocate out array: shape of bbox_out
    array_out=np.zeros(shape=(int((bbox_out['top']-bbox_out['bottom'])/res_out["dx"]),int((bbox_out['right']-bbox_out['left'])/res_out["dy"])))-99.

    #window of bbox_intersection in the domain of bbox_smash
    window_intersection_out=_get_cropped_window_from_bbox(bbox_intersection,
                                                          bbox_out,
                                                          res_out["dx"],
                                                          res_out["dy"]
                                                          )

    #copy _crop_array (input matrix cropped) in array_out
    array_out[window_intersection_out['row_off']:window_intersection_out['row_off']+window_intersection_out["nrows"],
              window_intersection_out['col_off']:window_intersection_out['col_off']+window_intersection_out["ncols"]]=cropped_array

    return array_out


def _resample_array(array,
                    res_in={"dx":1,"dy":1},
                    res_out={"dx":1,"dy":1},
                    order=0,
                    cval=-99.,
                    grid_mode=True
                    ):
    '''
    
    Parameters
    ----------
    
    array: numpy.array()
        Input gridded numpy array shape=(n,m)
    
    res_in: dict()
        resolution of the input array in x and y direction. Disctionnary {"dx":, "dy":}
    
    res_out: dict()
        resolution of the output array in x and y direction. Disctionnary {"dx":, "dy":}
    
    order: int()
        order of the resampling cubic interpolation
    
    cval: float() | np.nan
        fill value for the extended boundaries
    
    grid_mode: bool()
        True | False. if True coordinate start from the edge of the cell. If False coordinate starts from the center of the cell.
    
    Return
    ------
    
    numpy.array()
        Cropped and resampled array according bbox_out and res_out
    
    '''
    
    ratio_x=res_in["dx"]/res_out["dx"]
    ratio_y=res_in["dy"]/res_out["dy"]
    resampled_array = zoom(array, (ratio_y, ratio_x), order=order, mode="grid-constant", cval=-99., grid_mode=True)
    
    return resampled_array

