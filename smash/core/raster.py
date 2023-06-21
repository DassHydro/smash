from __future__ import annotations

import numpy as np
from osgeo import gdal

import os
import errno

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT


### GDAL RASTER FUNCTIONS


# just open the raster and return the dataset
def gdal_raster_open(filename):
    """
    Opening a raster with gdal. this is just a wrapper around gdal.Open(filename)

    Parameters
    ----------
    filename : string, path to a file

    Returns
    -------
    dataset : gdal object

    Examples
    --------
    dataset = gdal_raster_open("filename")
    """
    
    if os.path.isfile(filename):
        dataset = gdal.Open(filename)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    return dataset


def read_windowed_raster_gdal(
    filename: str, smash_mesh: MeshDT, band=None, lacuna=None
) -> np.ndarray:
    """
    Reading a raster file with gdal and return a np.ndarray storing the different data bands according the SMASH model boundingbox.

    Parameters
    ----------
    filename : string, path to a file
    smash_mesh : smash.mesh object representing the mesh
    band: band to be read
    lacuna: float64 replacing the Nodata value

    Returns
    -------
    array : np.array or np.ndarray storing one or all different data, stored in filename, sliced compare to the mesh boundingbox

    Examples
    --------
    array=read_windowed_raster_gdal("filename", model.mesh)
    """
    dataset = gdal_raster_open(filename)

    geotransform = gdal_get_geotransform(dataset)

    if (geotransform["xres"] != smash_mesh.dx) or (
        geotransform["yres"] != smash_mesh.dx
    ):
        # Attempt to generate a smaller dataset before doing the reprojection. However, it is slower..
        # ~ window=gdal_smash_window_from_geotransform(geotransform,smash_mesh)
        # ~ dataset=gdal.Translate('/vsimem/raster.tif', dataset, srcWin=[window['col_off'], window['row_off'], window["ncols"], window["nrows"]])

        dataset = gdal_reproject_raster(dataset, smash_mesh.dx, smash_mesh.dx)
        geotransform = gdal_get_geotransform(dataset)

    # Todo:
    # If smash mesh larger than window: window=1,1,all,all
    # compute window of smash-mesh and get x_offset and y_offsets => offsets
    # pass this window to gdal_crop_dataset_to_ndarray(dataset=dataset,window=window,offsets=offset)
    # position the rainfall inside the mesh grid according offset !

    window = gdal_smash_window_from_geotransform(geotransform, smash_mesh)

    if band == None:
        array = gdal_crop_dataset_to_ndarray(
            dataset=dataset, window=window, lacuna=lacuna
        )
    else:
        array = gdal_crop_dataset_to_array(
            dataset=dataset, window=window, band=band, lacuna=lacuna
        )

    return array


def gdal_reproject_raster(dataset, xres, yres):
    """
    Reproject the dataset raster accoding a new resolution in the x and y directions

    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    xres: resolution in the x direction (columns) in meters
    yres: resolution in the y direction (rows) in meters

    Returns
    -------
    virtual_destination : a virtual gdal raster object at the new resolution

    Examples
    --------
    new_dataset=gdal_reproject_raster(dataset,smash_mesh.cellsize,smash_mesh.cellsize)
    """

    geotransform = gdal_get_geotransform(dataset)

    dataset_projection = dataset.GetProjection()

    new_dataset_geotranform = (
        geotransform["xleft"],
        float(xres),
        0.0,
        geotransform["ytop"],
        0.0,
        -float(yres),
    )

    # Do we must distinguish cases smash_mesh.dx<=geotransform['xres','yres'] and smash_mesh.dx>geotransform['xres','yres'] ? i.e use ceiling or floor function instead of int ?
    # At least it work for case smash_mesh.dx<=geotransform['xres','yres'] which is the moste common case for modelling.
    New_X_Size = int(dataset.RasterXSize * geotransform["xres"] / xres)
    New_Y_Size = int(dataset.RasterYSize * geotransform["yres"] / yres)

    in_memory_dataset = gdal.GetDriverByName("MEM")

    virtual_destination = in_memory_dataset.Create(
        "",
        New_X_Size,
        New_Y_Size,
        dataset.RasterCount,
        dataset.GetRasterBand(1).DataType,
    )

    ###########################################################
    # Workaround for gdal bug which initialise array to 0 instead as the No_Data value
    # Here we initialise the band manually with the nodata_value
    band = virtual_destination.GetRasterBand(
        1
    )  # Notice that band is a pointer to virtual_destination
    band.SetNoDataValue(-9999)
    Nodataarray = np.ndarray(shape=(New_Y_Size, New_X_Size))
    Nodataarray.fill(-9999.0)
    band.WriteArray(Nodataarray)
    ###########################################################

    virtual_destination.SetGeoTransform(new_dataset_geotranform)
    virtual_destination.SetProjection(dataset_projection)
    gdal.ReprojectImage(
        dataset,
        virtual_destination,
        dataset_projection,
        dataset_projection,
        gdal.GRA_NearestNeighbour,
        WarpMemoryLimit=500.0,
    )
    # WarpMemoryLimit=500. would probably increase the speed... but ... #https://gdal.org/programs/gdalwarp.html
    # choice are : gdal.GRA_NearestNeighbour, gdal.GRA_Mode, gdal.GRA_Average ... Not tested https://gdal.org/api/gdalwarp_cpp.html#_CPPv4N15GDALResampleAlg11GRA_AverageE
    # Use osgeo.gdal.Warp instead of ReprojectImage offer much more option like multithreading ? https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Warp

    return virtual_destination


# simply slice an array according a window
def gdal_crop_dataset_to_array(dataset=object(), window={}, band=1, lacuna=None):
    """
    Read the raster bands from gdal object and crop the array according the window

    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    window: window to crop (in grid unit)
    band: the band number to be read. default is band number 1
    lacuna: None or float64

    Returns
    -------
    sliced_array : an array

    Examples
    --------
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    array=gdal_crop_dataset_to_array(dataset,window,band=1)
    """

    dataset_band = dataset.GetRasterBand(band)

    sliced_array = dataset_band.ReadAsArray(
        window["col_off"], window["row_off"], window["ncols"], window["nrows"]
    )

    array_float = sliced_array.astype("float64")

    # Lacuna treatment here
    if isinstance(lacuna, float):
        Nodata = dataset_band.GetNoDataValue()
        mask = np.where(sliced_array == Nodata)
        array_float[mask] = lacuna

    return array_float


# simply slice an array according a window
def gdal_crop_dataset_to_ndarray(dataset=object(), window={}, lacuna=None):
    """
    Read the raster bands from gdal object and crop the array according the window

    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    window: window to crop (in grid unit)
    lacuna: None or float64

    Returns
    -------
    dictionnary : a dictionary with ndarrays (depending the number of bands)

    Examples
    ----------
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    array=gdal_crop_dataset_to_array(dataset,window)
    """

    dictionnary = {}
    nb_dataset = dataset.RasterCount
    for index in range(1, nb_dataset + 1):
        dataset_band = dataset.GetRasterBand(index)

        sliced_array = dataset_band.ReadAsArray(
            window["col_off"], window["row_off"], window["ncols"], window["nrows"]
        )

        array_float = sliced_array.astype("float64")

        # Lacuna treatment here
        if isinstance(lacuna, float):
            Nodata = dataset_band.GetNoDataValue()
            mask = np.where(sliced_array == Nodata)
            array_float[mask] = lacuna

        dictionnary.update({index: array_float})

    return dictionnary


# write a new data set according a name, a meta description and bands as a list of array
def gdal_write_dataset(filename, dataset, format="Gtiff"):
    """
    write a gdal object to a new file

    Parameters
    ----------
    filename : path to the new target file
    dataset : gdal object from gdal.Open()
    format: optional, raster format, default is Gtiff

    Returns
    -------
    none

    Examples
    --------
    virtual_dataset=gdal_reproject_raster(dataset,500.,500.)
    gdal_write_dataset('outfile',virtual_dataset)
    """
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(
        filename,
        xsize=width,
        ysize=height,
        bands=dataset.RasterCount,
        eType=dataset.GetRasterBand(1).DataType,
    )

    dst_ds.SetGeoTransform(dataset.GetGeoTransform())
    dst_ds.SetProjection(dataset.GetProjection())

    data = dataset.ReadAsArray(0, 0, width, height)

    # ~ for index in range(1,dataset.RasterCount+1):
    # ~ dst_ds.GetRasterBand(index).WriteArray(data[index-1])

    dst_ds.WriteRaster(
        0,
        0,
        width,
        height,
        data.tobytes(),
        width,
        height,
        band_list=list(range(1, dataset.RasterCount + 1)),
    )

    # destination=dataset.CreateCopy(filename, dataset, strict=0,options=["TILED=YES", "COMPRESS=PACKBITS"])
    dst_ds = None


def gdal_get_geotransform(dataset):
    """
    Getting the GeoTransform coeficients from a gdal object

    Parameters
    ----------
    dataset : gdal object from gdal.Open()

    Returns
    -------
    geotransform : Python dictionnary

    # ~ A GeoTransform consists in a set of 6 coefficients:
    # ~ GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
    # ~ GT(1) w-e pixel resolution / pixel width.
    # ~ GT(2) row rotation (typically zero).
    # ~ GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
    # ~ GT(4) column rotation (typically zero).
    # ~ GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).

    Examples
    ----------
    dataset = gdal_raster_open(filename)
    geotransform=gdal_get_geotransform(dataset)
    """

    transform = dataset.GetGeoTransform()
    geotransform = {
        "xleft": transform[0],
        "xres": transform[1],
        "ytop": transform[3],
        "yres": -transform[5],
    }

    return geotransform


def gdal_smash_window_from_geotransform(geotransform, smash_mesh):
    """
    Compute the dataset array window (from the geotransform) according the Smash mesh

    Parameters
    ----------
    geotransform : geotransform computed from a gdal dataset
    smash_mesh : Smash mesh object model.mesh

    Returns
    -------
    window : Python dictionnary with componnents: row_off (offset), col_off (offset), nrows, ncols

    Examples
    ----------
    dataset = gdal_raster_open(filename)
    geotransform=gdal_get_geotransform(dataset)
    window=gdal_smash_window_from_geotransform(geotransform,smash_mesh)
    """

    col_off = (smash_mesh.xmin - geotransform["xleft"]) / geotransform["xres"]
    row_off = (geotransform["ytop"] - smash_mesh.ymax) / geotransform["yres"]

    # If smash_mesh.dx==geotransform['xres','yres'] no problem !
    # It works for case : smash_mesh.dx!=geotransform['xres','yres']
    # Do we must distinguish case smash_mesh.dx<=geotransform['xres','yres'] and smash_mesh.dx>geotransform['xres','yres'] ? i.e use ceiling or floor function instead of int ?

    window = {
        "row_off": row_off,
        "col_off": col_off,
        "nrows": int(smash_mesh.nrow * smash_mesh.dx / geotransform["yres"]),
        "ncols": int(smash_mesh.ncol * smash_mesh.dx / geotransform["xres"]),
    }

    return window


def union_bbox(bbox1, bbox2):
    """
    Function which compute the bounding boxes union of 2 input bbox. It return the working bbox

    Parameters
    ----------
    bbox1: dict containin the first bbox informations
    bbox2 : dict containin the second bbox informations
    
    returns
    -------
    dic containing the bbox union

    Examples
    --------
    dataset=gdal_raster_open(filename)
    possible_bbox=union_bbox(bbox,bbox_dataset)
    """
    left = max(bbox1["left"], bbox2["left"])
    bottom = max(bbox1["bottom"], bbox2["bottom"])
    right = min(bbox1["right"], bbox2["right"])
    top = min(bbox1["top"], bbox2["top"])
    if (left < right) and (bottom < top):
        bbox_union = {"left": left, "bottom": bottom, "right": right, "top": top}
        return bbox_union
    else:
        raise Exception("Impossible bounding boxes union")


def get_bbox(dataset):
    """
    Function to get the bbox from a raster dataset opened with Gdal

    Parameters
    ----------
    dataset: gdal object
    
    returns
    -------
    dic containing the bbox of the dataset

    Examples
    --------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    """
    geotransform = gdal_get_geotransform(dataset)
    left = geotransform["xleft"]
    right = geotransform["xleft"] + dataset.RasterXSize * geotransform["xres"]
    bottom = geotransform["ytop"] - dataset.RasterYSize * geotransform["yres"]
    top = geotransform["ytop"]
    bbox = {"left": left, "bottom": bottom, "right": right, "top": top}

    return bbox


def get_bbox_from_window(dataset, window):
    """
    Function to get the bbox of a defined window of a dataset

    Parameters
    ----------
    dataset: gdal object
    window : dict with ncols, nrows, col offset and row offset
    
    returns
    -------
    dic containing the computed bbox

    Examples
    --------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    window=get_window_from_bbox(dataset,bbox_dataset)
    possible_bbox=get_bbox_from_window(dataset,window)

    """
    geotransform = gdal_get_geotransform(dataset)
    left = geotransform["xleft"] + window["col_off"] * geotransform["xres"]
    right = left + window["ncols"] * geotransform["xres"]
    top = geotransform["ytop"] - window["row_off"] * geotransform["yres"]
    bottom = top - window["nrows"] * geotransform["yres"]
    bbox = {"left": left, "bottom": bottom, "right": right, "top": top}

    return bbox


def get_window_from_bbox(dataset, bbox):
    """
    Function to get the window of a defined bbox of a dataset

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
    window=get_window_from_bbox(dataset,bbox_dataset)

    """
    geotransform = gdal_get_geotransform(dataset)

    col_off = (bbox["left"] - geotransform["xleft"]) / geotransform["xres"]
    row_off = (geotransform["ytop"] - bbox["top"]) / geotransform["yres"]
    ncols = (bbox["right"] - bbox["left"]) / geotransform["xres"]
    nrows = (bbox["top"] - bbox["bottom"]) / geotransform["yres"]

    if (col_off < 0) or (row_off < 0):
        raise Exception(
            "The requested bounding box exceeds the limits of the raster domain."
        )

    window = {
        "row_off": int(row_off),
        "col_off": int(col_off),
        "nrows": int(nrows),
        "ncols": int(ncols),
    }

    return window


# simply crop an array according a window
def crop_array(array, window):
    """
    Function to crop an array according a window

    Parameters
    ----------
    array: numpy array
    window : dict containg the window to crop
    
    returns
    -------
    crop_array: the cropped numpy array, shape of the defined window

    """
    crop_array[
        window["col_off"] : window["col_off"] + window["ncols"],
        window["row_off"] : window["row_off"] + window["nrows"],
    ]
    return crop_array
