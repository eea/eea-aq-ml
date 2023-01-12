# Databricks notebook source
"""
================================================================================
Download CLIMATE Datasets to Azure Blob storages.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/131081
Author   : ahuarte@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))

# Mount the 'AQ CLIMATE' Azure Blob Storage Container as a File System.
aq_climate_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-climate', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# COMMAND ----------

# DBTITLE 1,Main variables & constants

# Default delay in Days of "Today" Date (https://surfobs.climate.copernicus.eu/dataaccess/access_eobs_months.php).
DEFAULT_DAYS_DELAY = 16


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
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_climate_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing download:')
print_message(logging.INFO, ' + Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Delay of Date: {}'.format(delay_of_date))


# COMMAND ----------

# DBTITLE 1,Prepare GDAL environment

import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

gdal.SetConfigOption('VSI_CACHE', 'TRUE')
gdal.SetConfigOption('VSI_CACHE_SIZE', '100000000')
gdal.SetConfigOption('GDAL_CACHE', '100000000')

def get_envelope(dataset):
    """
    Get the spatial envelope of the GDAL dataset.
    """
    geo_transform = dataset.GetGeoTransform()
    c = geo_transform[0]
    a = geo_transform[1]
    b = geo_transform[2]
    f = geo_transform[3]
    d = geo_transform[4]
    e = geo_transform[5]
    t = 0 # Texel offset, by default the texel is centered to CENTER-CENTER pixel.
    col = 0
    row = 0
    env_a = [a * (col + t) + b * (row + t) + c, d * (col + t) + e * (row + t) + f]
    col = dataset.RasterXSize
    row = dataset.RasterYSize
    env_b = [a * (col + t) + b * (row + t) + c, d * (col + t) + e * (row + t) + f]
    min_x = min(env_a[0], env_b[0])
    min_y = min(env_a[1], env_b[1])
    max_x = max(env_a[0], env_b[0])
    max_y = max(env_a[1], env_b[1])
    return min_x, min_y, max_x, max_y

def climate_to_raster_file(band, raster_size, data_type, target_file, crs, transform, metadata, driver_name='Gtiff', options=[]):
    """
    Creates a new Dataset with a new Band from the specified CLIMATE input.
    """
    mt_data = band.GetMetadata()
    no_data = band.GetNoDataValue()
    n_cols, n_rows = raster_size
    raster  = band.ReadAsArray(0, 0, n_cols, n_rows)
    # print('n_cols={} n_rows={}'.format(n_cols, n_rows))
    # print(raster.shape)
    
    # Rescale values of input Raster?
    scale_factor = 1.0
    if no_data is None: no_data = -9999
    if mt_data and mt_data.get('scale_factor'): scale_factor = float(mt_data.get('scale_factor'))
    if scale_factor != 1.0:
        mask_r = (raster != no_data)
        raster = raster.astype(np.double)
        raster[mask_r] *= scale_factor
        raster = raster.astype(np.float)
        data_type = gdal.GDT_Float32
    
    # print('scale_factor={}'.format(scale_factor))

    # Main properties of current Dataset.    
    driver = gdal.GetDriverByName(driver_name)
    output = driver.Create(target_file, n_cols, n_rows, 1, data_type, options=options)
    output.SetGeoTransform(transform)
    output.SetProjection(crs.ExportToWkt())
    if metadata: output.SetMetadata(metadata)
    
    i_band = band
    r_band = output.GetRasterBand(1)
    
    # Metadata of Band.
    no_data = i_band.GetNoDataValue()
    if no_data is not None: r_band.SetNoDataValue(no_data)    
    unit_type = i_band.GetUnitType()
    if unit_type is not None: r_band.SetUnitType(unit_type)
    metadata_b = i_band.GetMetadata()
    if metadata_b: r_band.SetMetadata(metadata_b)
    
    r_band.WriteArray(raster)
    r_band.FlushCache()
    i_band = None
    r_band = None
    
    output.FlushCache()
    del output
    output = None
    
    return target_file


# COMMAND ----------

# DBTITLE 1,Download CLIMATE rasters

import os
import glob
import datetime
import tempfile
import uuid
import requests

# Source root folder from original CLIMATE Datasets are downloaded (https://surfobs.climate.copernicus.eu/dataaccess/access_eobs_months.php):
CLIMATE_SOURCE_FOLDER = '/dbfs' + aq_climate_path + '/Ensemble'

start_date = datetime.datetime.strptime('1950-01-01', '%Y-%m-%d')
srid = 4326

# Main GDAL settings.
crs = osr.SpatialReference()
if hasattr(crs, 'SetAxisMappingStrategy'): crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
crs.ImportFromEPSG(srid)

try:
    for source_url in [      
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/tg_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/tn_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/tx_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/rr_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/pp_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/hu_0.1deg_day_{}_grid_ensmean.nc',
        'https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/months/ens/qq_0.1deg_day_{}_grid_ensmean.nc',
        ]:
        source_url = source_url.format(downloading_date[0:4])
        print_message(logging.INFO, 'Importing "{}"...'.format(source_url))
        
        minimum_date_days = None
        maximum_date_days = None
        date_count = 0
        
        measure = os.path.basename(source_url)[0:2]
        measure = measure.upper()
        
        # Define local temporary filename, now working with a local copy of remote files.
        temp_name = str(uuid.uuid1()).replace('-', '')
        temp_clim_file = os.path.join(tempfile.gettempdir(), temp_name+'.nc')
        
        try:
            print_message(logging.INFO, '+ Temporary file: {}'.format(temp_clim_file), indent=1)
            
            # Download input file from AWS.
            r = requests.get(source_url, stream=True)
            total_length = int(r.headers.get('content-length'))
            print_message(logging.INFO, '+ File size: {:.2f} Mb'.format(total_length/(1024.0*1024.0)), indent=1)
            
            with open(temp_clim_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size = 8192):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            
            # Get main Metadata.
            dataset   = gdal.Open(temp_clim_file, gdal.GA_ReadOnly)
            envelope  = get_envelope(dataset)
            transform = dataset.GetGeoTransform()
            metadata  = dataset.GetMetadata()
            n_bands   = dataset.RasterCount
            cols,rows = dataset.RasterXSize, dataset.RasterYSize
            data_type = dataset.GetRasterBand(1).DataType
            print_message(logging.INFO, '+ Bands: {}, Envelope: {}, Size: ({},{}), DataType: {}.'.format(n_bands, envelope, cols, rows, gdal.GetDataTypeName(data_type)), indent=1)
            
            if metadata:
                for key in ['NETCDF_DIM_ensemble_DEF','NETCDF_DIM_ensemble_VALUES','NETCDF_DIM_EXTRA','NETCDF_DIM_time_DEF','NETCDF_DIM_time_VALUES']:
                    if metadata.get(key): del metadata[key]
            
            # Process each valid Date.
            for date_index in range(n_bands):
                band_r  = dataset.GetRasterBand(date_index + 1)
                no_data = band_r.GetNoDataValue()
                mt_data = band_r.GetMetadata()
                time_r  = int(mt_data['NETCDF_DIM_time'])
                date_r  = start_date + datetime.timedelta(days=time_r)
                
                # Has current Band valid data???
                raster  = band_r.ReadAsArray(0, 0, cols, rows)
                mask_r  = (raster != no_data)
                count_r = np.count_nonzero(mask_r)
                if count_r == 0: continue
                
                if minimum_date_days is None or time_r < minimum_date_days: minimum_date_days = time_r
                if maximum_date_days is None or time_r > maximum_date_days: maximum_date_days = time_r
                y, m, d = date_r.year, date_r.month, date_r.day
                
                # File already saved?
                target_file = '{}/{}/{}/{:02d}/CLIMATE_{}_{}.tiff'.format(CLIMATE_SOURCE_FOLDER, measure, y, m, measure, date_r.strftime('%Y-%m-%d'))
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                if os.path.exists(target_file): continue
                
                # Create temporary file.
                climate_to_raster_file(band=band_r, raster_size=(cols,rows), data_type=data_type,
                                       target_file=temp_clim_file, 
                                       crs=crs, 
                                       transform=transform,
                                       metadata=metadata, 
                                       driver_name='Gtiff', 
                                       options=['TILED=YES','COMPRESS=DEFLATE'])
                
                # Copy the temporary raster file to output path.
                final_file = target_file[5:]
                dbutils.fs.cp('file:' + temp_clim_file, 'dbfs:' + final_file)
                os.remove(temp_clim_file)
                print_message(logging.INFO, '+ Ok! Result={}'.format(target_file), indent=1)
                
                band_r = None
                date_count += 1
                #break
                
            dataset = None
        finally:
            if os.path.exists(temp_clim_file): os.remove(temp_clim_file)
            
        print_message(logging.INFO, '+ MinimumDate: {}'.format(start_date + datetime.timedelta(days=minimum_date_days)), indent=1)
        print_message(logging.INFO, '+ MaximumDate: {}'.format(start_date + datetime.timedelta(days=maximum_date_days)), indent=1)
        print_message(logging.INFO, '+ DateCount: {}'.format(date_count), indent=1)            
        print_message(logging.INFO, '+ Ok!', indent=1)
        #break
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date, 'delay': delay_of_date})
    
print_message(logging.INFO, 'Downloadings successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date, 'delay': delay_of_date})
notebook_mgr = None

