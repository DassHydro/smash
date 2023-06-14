import smash
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from functions_smash_plot import *
from functions_smash_time import *

setup,mesh=smash.load_dataset('cance')
model=smash.Model(setup,mesh)
plot=plot_discharges(model)
plot[0].show()

#test the model with smaller dx
mesh["dx"]=500.
model_desag=smash.Model(setup,mesh)

index=np.where(model.input_data.mean_prcp[0,:]>10)[0]

plot_matrix(model.input_data.prcp[:,:,index[0]],mask=model.mesh.active_cell,figname='rainfall_dx1000.png',title="rainfall dx=1000m",vmin=0.0,vmax=25)
plot_matrix(model_desag.input_data.prcp[:,:,index[0]],mask=model_desag.mesh.active_cell,figname='rainfall_dx500.png',title="rainfall dx=500m",vmin=0.0,vmax=25) #compare the upper-left corner with the previous figure, it is like the catchment is half sized from the upper-left corner



#tiff filename seleciton
date='201410100600'
YY,MM,DD,hh,mm=decompose_date(date)
filename=f'/home/maxime/DassHydro-Dev/smash/smash/dataset/Cance/prcp/{YY}/{MM}/{DD}/rain_precipitation_{date}_{date}.tif'

#test the function globally
model.mesh.dx=500.
array=smash.core.raster.read_windowed_raster_gdal(filename=filename, smash_mesh=model.mesh, band=1, lacuna=-99.)

#manually test all action of the function, test to crop the array before with vsimem fsystem (I expected to be faster but it not)
dataset =smash.core.raster.gdal_raster_open(filename)
geotransform=smash.core.raster.gdal_get_geotransform(dataset)
window=smash.core.raster.gdal_smash_window_from_geotransform(geotransform,model.mesh)
dataset=gdal.Translate('/vsimem/raster.tif', dataset, srcWin=[window['col_off'], window['row_off'], window["ncols"], window["nrows"]])
dataset=smash.core.raster.gdal_reproject_raster(dataset,model.mesh.dx,model.mesh.dx)
geotransform=smash.core.raster.gdal_get_geotransform(dataset)
window=smash.core.raster.gdal_smash_window_from_geotransform(geotransform,model.mesh)
array=smash.core.raster.gdal_crop_dataset_to_array(dataset=dataset,window=window,band=1,lacuna=-99.)


#Test the reprojection
dataset = smash.core.raster.gdal_raster_open(filename)
dataset_500=smash.core.raster.gdal_reproject_raster(dataset,500,500)
dataset_2000=smash.core.raster.gdal_reproject_raster(dataset,2000,2000)
array=dataset.GetRasterBand(1).ReadAsArray()
array_500=dataset_500.GetRasterBand(1).ReadAsArray()
array_2000=dataset_2000.GetRasterBand(1).ReadAsArray()

maxval=np.max(array)
plot_matrix(array,figname='grid_rainfall_dx1000.png',title="rainfall dx=1000m",vmin=0.0,vmax=maxval)
plot_matrix(array_500,figname='grid_rainfall_dx500.png',title="rainfall dx=500m",vmin=0.0,vmax=maxval)
plot_matrix(array_2000,figname='grid_rainfall_dx2000.png',title="rainfall dx=2000m",vmin=0.0,vmax=maxval)
#Notice grid_rainfall_dx1000.png grid_rainfall_dx500.png are very similar but the size is different, grid_rainfall_dx2000.png has the same aspect but the rainfall is averaged for each cell 2x2 km
