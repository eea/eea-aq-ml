# Databricks notebook source
"""
================================================================================
Download ERA5 meteorology Datasets to Azure Blob storages.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157025
Author   : mgonzalez@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))

# Mount the 'AQ CAMS' Azure Blob Storage Container as a File System.
aq_meteo_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-meteorology', 
  sas_key = 'sv=2021-10-04&st=2022-12-20T10%3A39%3A26Z&se=2030-11-21T10%3A39%3A00Z&sr=c&sp=racwdl&sig=DT1L2jy%2B8KcECwCiDaNO%2F1bIsQCytDmKhNz9i2r4ZHg%3D'
)

# COMMAND ----------


# Climate Data Store (CDS) API settings.
# Docs:
# https://cds.climate.copernicus.eu/cdsapp
# https://cds.climate.copernicus.eu/api-how-to
#
CDSAPI_URL = 'https://cds.climate.copernicus.eu/api/v2'#'https://ads.atmosphere.copernicus.eu/api/v2'
CDSAPI_KEY = '149766:f99170e3-044f-4173-8be7-176ed9b3733f'

# Default delay in Days of "Today" Date (It seems the CDS Service does not provide Data for "Today", as much for "Yesterday").
DEFAULT_DAYS_DELAY = 1

valid_ranges_of_aggregations = {
  'AVG': [0, 10000], 
  'AOT40': [0, 1000000],
  'SOMO35': [0, 1000000]
}

variables = [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'surface_net_solar_radiation', 'surface_pressure', 'total_precipitation',
        ]

times = [
  '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
  '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', 
  '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', 
  '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
]


# COMMAND ----------

# DBTITLE 1,Prepare DataFactory environment

import traceback
import json
import logging
import os
import datetime

# Create Widgets for leveraging parameters:
# https://docs.microsoft.com/es-es/azure/databricks/notebooks/widgets
dbutils.widgets.removeAll()
dbutils.widgets.text(name='delay', defaultValue=str(DEFAULT_DAYS_DELAY), label='Delay of Date')
delay_of_date = dbutils.widgets.get('delay')
delay_of_date = int(delay_of_date) if delay_of_date is not None and len(str(delay_of_date)) > 0 else DEFAULT_DAYS_DELAY
dbutils.widgets.text(name='date', defaultValue='', label='Downloading Date')
downloading_date = dbutils.widgets.get('date')
if not downloading_date: downloading_date = (datetime.datetime.now() - datetime.timedelta(days=delay_of_date)).strftime('%Y-%m-%d')

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_meteo_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing download:')
print_message(logging.INFO, ' + Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Delay of Date: {}'.format(delay_of_date))




# COMMAND ----------

# DBTITLE 1,Prepare GDAL environment

import numpy as np

from osgeo import gdal
from osgeo import osr

import uuid

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

print_message(logging.INFO, 'GDAL_VERSION={}'.format(gdal.VersionInfo()))

def transform_netcdf_to_raster_file(source_file, temp_file_dir, target_file_dir, variable, srid=4326, driver_name='Gtiff', options=[], valid_range=None):
    """
    Creates a new Dataset with a new Band as the Mean of the whole existing Bands.
    """
    dataset = gdal.Open(source_file, gdal.GA_ReadOnly)

    band_r = dataset.GetRasterBand(1)
    no_data = band_r.GetNoDataValue()
    band_r = None
    
    # Main properties of current Dataset.
    crs = osr.SpatialReference()
    if hasattr(crs, 'SetAxisMappingStrategy'): crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs.ImportFromEPSG(srid)
    
    # Append the Mean of all Hours (24 Hours per Day) as 25th Band.
    raster = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
    #
    # Apply valid range of Values.
    if valid_range:
        min_valid_value = valid_range[0]
        max_valid_value = valid_range[1]
        mask_r = (raster >= min_valid_value) & (raster <= max_valid_value)
        raster[~mask_r] = no_data
    #
        
    raster_count, rows, cols = raster.shape

    # Driver for output rasters
    driver = gdal.GetDriverByName(driver_name)    
    
    dataset_metadata = dataset.GetMetadata()
    
    era5_orig_time = datetime.datetime.strptime('1900-01-01 00:00:00.0','%Y-%m-%d %H:%M:%S.%f')
    
    for band in range(0, raster_count):
      # Load current band
      r = raster[band]
      n_band =  band
      i_band = dataset.GetRasterBand(n_band+1)
      metadata = i_band.GetMetadata()
      
      # prepare output file
      raster_date = era5_orig_time + datetime.timedelta(hours=int(metadata.get('NETCDF_DIM_time')))
      target_temp_file = os.path.join(temp_file_dir
                                 ,str(uuid.uuid1()).replace('-', '')+'.tiff')
      target_file = os.path.join(target_file_dir
                                 ,'{}/{}/{}/ERA5_{}_{}_{}.tiff'.format(raster_date.year, raster_date.month, raster_date.day, variable, raster_date.strftime('%Y%m%d'), raster_date.strftime('%H%M')))
      
      
      
      output = driver.Create(target_temp_file, cols, rows, 1, gdal.GDT_Float32, options=options)
      output.SetGeoTransform(dataset.GetGeoTransform())
      output.SetProjection(crs.ExportToWkt())
      
      if dataset_metadata: output.SetMetadata(metadata)

      r_band = output.GetRasterBand(1)

      unit_type = i_band.GetUnitType()
      if unit_type is not None: r_band.SetUnitType(unit_type)
      no_data = i_band.GetNoDataValue()
      if no_data is not None: r_band.SetNoDataValue(no_data)

      
      #if metadata and band == 24 and metadata.get('NETCDF_DIM_time'): metadata['NETCDF_DIM_time'] = 'MEAN'
      if metadata: r_band.SetMetadata(metadata)
      # print(r_band.GetMetadata())

      r_band.WriteArray(r)
      r_band.FlushCache()
      i_band = None
      r_band = None
      print_message(logging.INFO, 'Ok! Result={}'.format(target_file))
      output.FlushCache()
      del output
      output = None
      
      directory = os.path.dirname(target_file)
      os.makedirs(directory, exist_ok=True)
      dbutils.fs.cp('file:' + target_temp_file, 'dbfs:' + target_file)
    del dataset
    dataset = None
    
    return None


# COMMAND ----------

# DBTITLE 1,Download Meteorology rasters

import os
import cdsapi
import yaml
import tempfile
import uuid

try:
    client = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)
    
    # Dowload CAMS for each Pollutant.
    for key in variables:
        y, m = downloading_date[0:4], downloading_date[5:7]
        
        #pollutant = pollutants[key]
        subfolder = os.path.join(aq_meteo_path, 'Reanalysis/{}/'.format(key))
        #file_name = '/dbfs' + os.path.join(aq_cams_path, '{}/ERA5_{}_{}.tiff'.format(subfolder, key))
        #file_name = '/dbfs' + os.path.join(aq_cams_path, '{}/ERA5_{}_{}.tiff')
        
        print_message(logging.INFO, 'Downloading CAMS of "{}"...'.format(key))
        
        #directory = os.path.dirname(file_name)
        #os.makedirs(directory, exist_ok=True)
        
        #if os.path.exists(file_name):
        #    print_message(logging.INFO, 'Skipping downloading, the CAMS Dataset ({}) already exists.'.format(file_name), indent=1)
        #    continue
            
        # Define local temporary filenames, we will add the Mean of all Bands (24 Hours per Day) as a new Band.
        temp_name = str(uuid.uuid1()).replace('-', '')
        temp_era5_file = os.path.join(tempfile.gettempdir(), temp_name+'.nc')
        #temp_tif_file = os.path.join(tempfile.gettempdir())
        # print(temp_name)
        # print(temp_cams_file)
        # print(temp_mean_file)
        
        raster = client.retrieve(
            'reanalysis-era5-single-levels', 
            {
              'product_type': 'reanalysis',
              'year': '2022',
              'month': '12',
              'day': [
                  '01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
                  '13',
              ],
              'format': 'netcdf',             
              'time': times,
               'area': [
                  73, -40, 31,
                  50,
              ],
              'variable': [key]
            },
            'download.nc'
        )
        result = raster.download(temp_era5_file)
        print_message(logging.INFO, 'Ok! Result={}'.format(result))
        
        # Append the Mean of all Hours (24 Hours per Day) as 25th Band.
        #print_message(logging.INFO, 'Appending the Mean of all Bands...')
        transform_netcdf_to_raster_file(
            source_file=temp_era5_file, temp_file_dir=tempfile.gettempdir(), target_file_dir=subfolder, variable=key, srid=4326, driver_name='Gtiff', options=[]#, valid_range=valid_ranges_of_aggregations['AVG']
        )
                
            
        # Copy the temporary raster file to output path.
        #final_file = file_name[5:]
        #dbutils.fs.cp('file:' + temp_mean_file, 'dbfs:' + final_file)
        #os.remove(temp_cams_file)
        #os.remove(temp_mean_file)
        
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date, 'delay': delay_of_date})

print_message(logging.INFO, 'Downloadings successfully processed!')

# COMMAND ----------


