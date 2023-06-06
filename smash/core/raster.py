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

def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]


def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:,0]),
                       np.min(arr[:,1]),
                       np.max(arr[:,0]),
                       np.max(arr[:,1]))



def xy_to_colrow(x, y, xmin, ymax, xres, yres):
    
    col = int((x - xmin) / xres)
    row = int((ymax - y) / yres)
    
    return col, row


def colrow_to_xy(col, row, xmin, ymax, xres, yres):
    
    x = int(col * xres + xmin)
    y = int(ymax - row * yres)
    
    return x, y



def trim_zeros_2D(array, shift_value=False):
    
    for ax in [0, 1]:
        
        mask = ~(array == 0).all(axis=ax)
        
        inv_mask = mask[::-1]
        
        start_ind = np.argmax(mask)
        
        end_ind = len(inv_mask) - np.argmax(inv_mask)
        
        if ax == 0:
            scol, ecol = start_ind, end_ind
            array = array[:, start_ind:end_ind]
        else:
            srow, erow = start_ind, end_ind
            array = array[start_ind:end_ind, :]
        
    if shift_value:
        return array, scol, ecol, srow, erow
    else:
        return array




#just open the raster and return the dataset
def gdal_raster_open(filename):
    """
    Opening a raster with gdal. this is just a wrapper around gdal.Open(filename)
    
    Parameters
    ----------
    filename : string, path to a file
    
    Returns
    ----------
    dataset : gdal object
    
    Examples
    ----------
    dataset = gdal_raster_open("filename")
    """
    dataset=object()
    if os.path.isfile(filename):
        dataset = gdal.Open(filename)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    
    return dataset


def read_windowed_raster_gdal(filename: str, smash_mesh: MeshDT, band=None, lacuna=None) -> np.ndarray:
    """
    Reading a raster file with gdal and return a np.ndarray storing the different data bands according the SMASH model boundingbox.
    
    Parameters
    ----------
    filename : string, path to a file
    smash_mesh : smash.mesh object representing the mesh
    band: band to be read
    lacuna: float64 replacing the Nodata value
    
    Returns
    ----------
    array : np.array or np.ndarray storing one or all different data, stored in filename, sliced compare to the mesh boundingbox
    
    Examples
    ----------
    array=read_windowed_raster_gdal("filename", model.mesh)
    """
    dataset = gdal_raster_open(filename)
    
    geotransform=gdal_get_geotransform(dataset)
    
    if (geotransform['xres'] != smash_mesh.dx) or (geotransform['yres'] != smash_mesh.dx):
        new_dataset=gdal_reproject_raster(dataset,smash_mesh.dx,smash_mesh.dx)
        dataset=new_dataset
    
    #si mesh larger than window: window=1,1,all,all
    #compute window of smash-mesh and get xoffset and y offsets => offsets
    #pass this window to gdal_crop_dataset_to_ndarray(dataset=dataset,window=window,offsets=offset)
    #position the rainfall inside the mesh grid according offset !
    
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    
    if (band==None):
        array=gdal_crop_dataset_to_ndarray(dataset=dataset,window=window,lacuna=lacuna)
    else:
        array=gdal_crop_dataset_to_array(dataset=dataset,window=window,band=band,lacuna=lacuna)
                
    
    return array
    


def gdal_get_geotransform(dataset):
    """
    Getting the GeoTransform coeficients from a gdal object
    
    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    
    Returns
    ----------
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
    xmin = transform[0]
    ymax = transform[3]
    xres = transform[1]
    yres = -transform[5]
    geotransform={'xleft':xmin,'xres':xres, 'ytop':ymax, 'yres':yres}
    return geotransform



def gdal_smash_window_from_geotransform(dataset,smash_mesh):
    """
    Compute the dataset array window according the Smash mesh
    
    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    smash_mesh : Smash mesh object model.mesh
    
    Returns
    ----------
    window : Python dictionnary with componnents: row_off (offset), col_off (offset), nrows, ncols
    
    Examples
    ----------
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    """
    geotransform=gdal_get_geotransform(dataset)
    
    col_off = (smash_mesh.xmin - geotransform['xleft']) / geotransform['xres']
    row_off = (geotransform['ytop'] - smash_mesh.ymax) / geotransform['yres']
    
    window={"row_off":row_off,'col_off':col_off,'nrows':int(smash_mesh.nrow*smash_mesh.dx/geotransform['yres']),'ncols':int(smash_mesh.ncol*smash_mesh.dx/geotransform['xres'])}
    
    return window
    

def union_bbox(bbox1,bbox2):
    """
    Function which compute the bounding boxes union of 2 input bbox. It return the working bbox 
    
    Parameters
    ----------
    bbox1: dict containin the first bbox informations
    bbox2 : dict containin the second bbox informations
    ----------
    returns
    dic containing the bbox union
    
    Examples
    ----------
    dataset=gdal_raster_open(filename)
    possible_bbox=union_bbox(bbox,bbox_dataset)
    """
    left=max(bbox1['left'],bbox2['left'])
    bottom=max(bbox1['bottom'],bbox2['bottom'])
    right=min(bbox1['right'],bbox2['right'])
    top=min(bbox1['top'],bbox2['top'])
    if (left<right) and (bottom<top):
        bbox_union={"left":left,"bottom":bottom,"right":right,"top":top}
        return bbox_union
    else:
        raise Exception("Impossible bounding boxes union")


def get_bbox(dataset):
    """
    Function to get the bbox from a raster dataset opened with Gdal
    
    Parameters
    ----------
    dataset: gdal object
    ----------
    returns
    dic containing the bbox of the dataset
    
    Examples
    ----------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    """
    geotransform=gdal_get_geotransform(dataset)
    left = geotransform['xleft']
    right = geotransform['xleft'] + dataset.RasterXSize * geotransform['xres']
    bottom = geotransform['ytop'] - dataset.RasterYSize * geotransform['yres']
    top = geotransform['ytop']
    bbox={"left":left,"bottom":bottom,"right":right,"top":top}
    
    return bbox


def get_bbox_from_window(dataset, window):
    """
    Function to get the bbox of a defined window of a dataset
    
    Parameters
    ----------
    dataset: gdal object
    window : dict with ncols, nrows, col offset and row offset
    ----------
    returns
    dic containing the computed bbox
    
    Examples
    ----------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    window=get_window_from_bbox(dataset,bbox_dataset)
    possible_bbox=get_bbox_from_window(dataset,window)
    
    """
    geotransform=gdal_get_geotransform(dataset)
    left=geotransform['xleft'] + window['col_off']*geotransform['xres']
    right=left + window['ncols']*geotransform['xres']
    top = geotransform['ytop']-window['row_off']*geotransform['yres']
    bottom = top - window['nrows']*geotransform['yres']
    bbox={"left":left,"bottom":bottom,"right":right,"top":top}
    
    return bbox



def get_window_from_bbox(dataset,bbox):
    """
    Function to get the window of a defined bbox of a dataset
    
    Parameters
    ----------
    dataset: gdal object
    bbox : dict containing the bbox
    ----------
    returns
    dic containing the computed windows
    
    Examples
    ----------
    dataset=gdal_raster_open(filename)
    bbox_dataset=get_bbox(dataset)
    window=get_window_from_bbox(dataset,bbox_dataset)
    
    """
    geotransform=gdal_get_geotransform(dataset)
    
    col_off = (bbox["left"] - geotransform['xleft']) / geotransform['xres']
    row_off = (geotransform['ytop'] - bbox["top"]) / geotransform['yres']
    ncols=(bbox["right"]-bbox["left"])/geotransform['xres']
    nrows=(bbox["top"]-bbox["bottom"])/geotransform['yres']
    
    if (col_off<0) or (row_off<0):
        raise Exception('The requested bounding box exceeds the limits of the raster domain.')
    
    window={'row_off':int(row_off),'col_off':int(col_off),'nrows':int(nrows),'ncols':int(ncols)}
    
    return window



#simply crop an array according a window
def crop_array(array,window):
    """
    Function to crop an array according a window
    
    Parameters
    ----------
    array: numpy array
    window : dict containg the window to crop
    ----------
    returns
    crop_array: the cropped numpy array, shape of the defined window
    
    """
    crop_array[window['col_off']:window['col_off']+window["ncols"], window['row_off']:window['row_off']+window["nrows"]]
    return crop_array



def gdal_reproject_raster(dataset,xres,yres):
    """
    Reproject the dataset raster accoding a new resolution in the x and y directions
    
    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    xres: resolution in the x direction (columns) in meters
    yres: resolution in the y direction (rows) in meters
    
    Returns
    ----------
    virtual_destination : a virtual gdal raster object at the new resolution
    
    Examples
    ----------
    new_dataset=gdal_reproject_raster(dataset,smash_mesh.cellsize,smash_mesh.cellsize)
    """
    dataset_geotrans = dataset.GetGeoTransform()
    dataset_projection=dataset.GetProjection()
    
    new_dataset_geotranform=(dataset_geotrans[0],float( xres ),0.0,dataset_geotrans[3],0.0,- float( yres ))
    
    in_memory_dataset=gdal.GetDriverByName('MEM')
    
    geotransform=gdal_get_geotransform(dataset)
    New_X_Size=int(dataset.RasterXSize*geotransform['xres']/xres)
    New_Y_Size=int(dataset.RasterYSize*geotransform['yres']/yres)
    
    virtual_destination=in_memory_dataset.Create('',New_X_Size, New_Y_Size, dataset.RasterCount, dataset.GetRasterBand(1).DataType)
    ###########################################################
    #Workaround for gdal bug which initialise array to 0 instead as the No_Data value
    #Here we initialise the band manually with the nodata_value
    band=virtual_destination.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    Nodataarray = np.ndarray(shape=(New_Y_Size,New_X_Size))
    Nodataarray.fill(-9999.0)
    band.WriteArray(Nodataarray)
    ###########################################################
    virtual_destination.SetGeoTransform(new_dataset_geotranform)
    virtual_destination.SetProjection(dataset_projection)
    gdal.ReprojectImage( dataset, virtual_destination, dataset_projection, dataset_projection, gdal.GRA_NearestNeighbour)
    
    return virtual_destination



#simply slice an array according a window
def gdal_crop_dataset_to_array(dataset=object(),window={},band=1,lacuna=None):
    """
    Read the raster bands from gdal object and crop the array according the window
    
    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    window: window to crop (in grid unit)
    band: the band number to be read. default is band number 1
    lacuna: None or float64
        
    Returns
    ----------
    sliced_array : an array
    
    Examples
    ----------
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    array=gdal_crop_dataset_to_array(dataset,window,band=1)
    """
        
    sliced_array=dataset.GetRasterBand(band).ReadAsArray(window['col_off'], window['row_off'], window["ncols"], window["nrows"])
    
    array_float=sliced_array.astype('float64')
    #Lacuna treatment here
    Nodata=dataset.GetRasterBand(band).GetNoDataValue()
    if (lacuna!=None):
        mask=np.where(sliced_array==Nodata)
        array_float[mask]=lacuna 
    
    return array_float



#simply slice an array according a window
def gdal_crop_dataset_to_ndarray(dataset=object(),window={},lacuna=None):
    """
    Read the raster bands from gdal object and crop the array according the window
    
    Parameters
    ----------
    dataset : gdal object from gdal.Open()
    window: window to crop (in grid unit)
    lacuna: None or float64
    
    Returns
    ----------
    dictionnary : a dictionary with ndarrays (depending the number of bands)
    
    Examples
    ----------
    window=gdal_smash_window_from_geotransform(dataset,smash_mesh)
    array=gdal_crop_dataset_to_array(dataset,window)
    """
    dictionnary={}
    for index in range(1,dataset.RasterCount+1):
        sliced_array=dataset.GetRasterBand(index).ReadAsArray(window['col_off'], window['row_off'], window["ncols"], window["nrows"])
        
        array_float=sliced_array.astype('float64')
        #Lacuna treatment here
        Nodata=dataset.GetRasterBand(index).GetNoDataValue()
        if (lacuna!=None):
            mask=np.where(sliced_array==Nodata)
            array_float[mask]=lacuna 
        
        dictionnary.update({index:array_float})
    
    return dictionnary


#write a new data set according a name, a meta description and bands as a list of array
def gdal_write_dataset(filename,dataset,format='Gtiff'):
    """
    write a gdal object to a new file
    
    Parameters
    ----------
    filename : path to the new target file
    dataset : gdal object from gdal.Open()
    format: optional, raster format, default is Gtiff
            
    Returns
    ----------
    none
    
    Examples
    ----------
    virtual_dataset=gdal_reproject_raster(dataset,500.,500.)
    gdal_write_dataset('outfile',virtual_dataset)
    """
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    driver = gdal.GetDriverByName( format )
    dst_ds = driver.Create(filename, xsize=width, ysize=height,bands=dataset.RasterCount, eType=dataset.GetRasterBand(1).DataType)
    
    dst_ds.SetGeoTransform(dataset.GetGeoTransform())
    dst_ds.SetProjection(dataset.GetProjection())
    
    data = dataset.ReadAsArray(0,0,width,height)
    
    # ~ for index in range(1,dataset.RasterCount+1):
        # ~ dst_ds.GetRasterBand(index).WriteArray(data[index-1])
    
    dst_ds.WriteRaster(0,0,width,height,data.tobytes(),width,height,band_list=list(range(1,dataset.RasterCount+1)))
    
    #destination=dataset.CreateCopy(filename, dataset, strict=0,options=["TILED=YES", "COMPRESS=PACKBITS"])
    dst_ds=None




### ASCII GRID Functions


def init_asciigrid(source={}):
    """
    initialise a empty asciigrid dictionnary
    
    Parameters
    ----------
    none
            
    Returns
    ----------
    dict, with default properties of a asciigrid
    
    """
    if (source.__len__()==0):
        
        asciigrid = {}
        asciigrid["ncols"] = 0
        asciigrid["nrows"] = 0
        asciigrid["xllcorner"] = 0.0
        asciigrid["yllcorner"] = 0.0
        asciigrid["cellsize"] = 0.0
        asciigrid["NODATA_value"] = -99.0
        asciigrid["data"] = np.full(shape=(0,0),fill_value=-99.0)
        asciigrid["extend"] = [0.0, 0.0, 0.0, 0.0]
    
    else:
        asciigrid=source.copy()
        asciigrid["data"] = np.full(shape=(source['nrows'],source['ncols']),fill_value=source['NODATA_value'])
    
    return asciigrid



def read_asciigrid(filename):
    """
    Read an asciigrid file
    
    Parameters
    ----------
    filename: path to the file
            
    Returns
    ----------
    dict, containing the asciigrid, data and its properties
    
    """
    asciigrid = init_asciigrid()
    if os.path.exists(filename):
        with open(filename, 'r') as input_file:
            header = input_file.readlines()[:6]
            header = [item.strip().split()[1] for item in header]
        
        asciigrid["ncols"] = int(header[0])
        asciigrid["nrows"] = int(header[1])
        asciigrid["xllcorner"] = float(header[2])
        asciigrid["yllcorner"] = float(header[3])
        asciigrid["cellsize"] = float(header[4])
        asciigrid["NODATA_value"] = float(header[1])
        asciigrid["data"] = np.loadtxt(filename, dtype=float, skiprows=6)
        asciigrid["extend"] = [
            asciigrid["xllcorner"], asciigrid["xllcorner"] + asciigrid["ncols"] * asciigrid["cellsize"],
            asciigrid["yllcorner"], asciigrid["yllcorner"] + asciigrid["nrows"] * asciigrid["cellsize"]]
    else:
        print(filename + " does not exist")
    
    return asciigrid


def set_asciigrid(data, ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value):
    """
    Set an asciigrid with data
    
    Parameters
    ----------
    data: numpy array containing the data (a matrix)
    ncols : integer, number of column
    nrows : integer, number of rows
    xllcorner : latitue coordinate of the lower left corner
    yllcorner : longitude coordinate of the lower left corner
    cellsize : cell resolution
    NODATA_value : value of the non-data value
    
    Returns
    ----------
    dict, containing the asciigrid, data and its properties
    
    """
    asciigrid = init_asciigrid()
    extend = [xllcorner, xllcorner + ncols * cellsize, yllcorner, yllcorner + nrows * cellsize]
    asciigrid = {"data": data,
                 "ncols": ncols,
                 "nrows": nrows,
                 "xllcorner": xllcorner,
                 "yllcorner": yllcorner,
                 "cellsize": cellsize,
                 "NODATA_value": NODATA_value,
                 "extend": extend}
    return asciigrid


def diff_asciigrid(grid1, grid2):
    """
    Compute the difference of 2 asccii grid
    
    Parameters
    ----------
    grid1: asciigrid dict
    grid2 : asciigrid dict
    
    Returns
    ----------
    asciigrid dict 1 - asciigrid dict 2
    
    """
    asciigrid = init_asciigrid()
    if grid1["extend"] == grid2["extend"]:
        asciigrid = {"data": grid1["data"] - grid2["data"],
                     "ncols": grid1["ncols"],
                     "nrows": grid1["nrows"],
                     "xllcorner": grid1["xllcorner"],
                     "yllcorner": grid1["yllcorner"],
                     "cellsize": grid1["cellsize"],
                     "NODATA_value": grid1["NODATA_value"],
                     "extend": grid1["extend"]}
    else:
        print("error: grids extend are different.")
    
    return asciigrid


def write_asciigrid(filename,asciigrid):
    """
    Write an asciigrid dict to a file
    
    Parameters
    ----------
    filename: path to a output filename
    asciigrid : dict containing the asciigrid
    
    Returns
    ----------
    None
    
    """
    header = (
        f"NCOLS {asciigrid['ncols']} \nNROWS {asciigrid['nrows']}"
        f"\nXLLCENTER {asciigrid['xllcorner']} \nYLLCENTER {asciigrid['yllcorner']} \nCELLSIZE {asciigrid['cellsize']}\nNODATA_value {asciigrid['NODATA_value']}\n"
    )
    
    
    head,tail=os.path.split(filename)
    if (len(head)!=0):
        if not (os.path.exists(head)):
            os.makedirs(head)
    
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, asciigrid['data'], "%5.2f")


