.. _api_reference.raster_handler:

==============
Raster handler
==============

.. currentmodule:: smash.tools.raster_handler

Some functions to manipulate raster files
*****************************************
.. autosummary::
   :toctree: smash/
  
   gdal_raster_open
   gdal_read_windowed_raster
   gdal_reproject_raster
   gdal_crop_dataset_to_array
   gdal_crop_dataset_to_ndarray
   gdal_write_dataset
   gdal_get_geotransform
   gdal_smash_window_from_geotransform
   union_bbox
   get_bbox
   get_bbox_from_window
   get_window_from_bbox
   crop_array
