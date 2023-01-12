# Databricks notebook source
"""
================================================================================
Download CAMS Datasets to Azure Blob storages.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/129254
Author   : niriarte@tracasa & ahuarte@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))

# Mount the 'AQ CAMS' Azure Blob Storage Container as a File System.
aq_cams_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-cams', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# COMMAND ----------

# DBTITLE 1,Main variables & constants

# Climate Data Store (CDS) API settings.
# Docs:
# https://cds.climate.copernicus.eu/cdsapp
# https://cds.climate.copernicus.eu/api-how-to
#
CDSAPI_URL = 'https://ads.atmosphere.copernicus.eu/api/v2'
CDSAPI_KEY = '1703:bf419efb-2f2b-490b-86da-e0b7120857c6'

# Default delay in Days of "Today" Date (It seems the CDS Service does not provide Data for "Today", as much for "Yesterday").
DEFAULT_DAYS_DELAY = 1

# Pollutant settings.

pollutants = {
  'NO2': 'nitrogen_dioxide', 
  'O3': 'ozone',
  'PM10': 'particulate_matter_10um',
  'PM25': 'particulate_matter_2.5um',
  'SO2': 'sulphur_dioxide'
}

valid_ranges_of_aggregations = {
  'AVG': [0, 10000], 
  'AOT40': [0, 1000000],
  'SOMO35': [0, 1000000]
}

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
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_cams_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing download:')
print_message(logging.INFO, ' + Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Delay of Date: {}'.format(delay_of_date))
print_message(logging.INFO, ' + Pollutants:')
for key in pollutants: print_message(logging.INFO, '\t{}: {}'.format(key, pollutants[key]))


# COMMAND ----------

# DBTITLE 1,Prepare GDAL environment

import numpy as np

from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

print_message(logging.INFO, 'GDAL_VERSION={}'.format(gdal.VersionInfo()))

def append_mean_to_raster_file(source_file, target_file, srid=4326, driver_name='Gtiff', options=[], valid_range=None):
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
    rcount = np.zeros(raster.shape, dtype=np.int32)
    mask_r = raster != no_data
    rcount[mask_r] = 1
    rcount = np.sum(rcount, axis=0)
    base_r = np.ones(raster.shape[1:], dtype=np.float32) * no_data
    mean_r = np.sum(raster, axis=0, where=mask_r, out=base_r)
    mean_r = np.divide(mean_r, rcount, where=(rcount!=0), out=mean_r)
    mean_r = mean_r[np.newaxis, ...]
    raster = np.append(raster, mean_r, axis=0)
    raster_count, rows, cols = raster.shape
    """    
    mean_r = np.sum(raster, axis=0) / float(raster_count)
    mean_r = mean_r[np.newaxis, ...]
    "
    print(raster.shape)
    print(raster_count, rows, cols)
    print(np.array_equal(raster[24], mean_r[0]))
    """
    
    driver = gdal.GetDriverByName(driver_name)
    output = driver.Create(target_file, cols, rows, raster_count, gdal.GDT_Float32, options=options)
    output.SetGeoTransform(dataset.GetGeoTransform())
    output.SetProjection(crs.ExportToWkt())
    
    metadata = dataset.GetMetadata()
    if metadata: output.SetMetadata(metadata)
    
    for band in range(0, raster_count):
        r = raster[band]
        n_band = 0 if band == 24 else band
        i_band = dataset.GetRasterBand(n_band + 1)
        r_band = output.GetRasterBand(band + 1)
        
        unit_type = i_band.GetUnitType()
        if unit_type is not None: r_band.SetUnitType(unit_type)
        no_data = i_band.GetNoDataValue()
        if no_data is not None: r_band.SetNoDataValue(no_data)
        
        metadata = i_band.GetMetadata()
        if metadata and band == 24 and metadata.get('NETCDF_DIM_time'): metadata['NETCDF_DIM_time'] = 'MEAN'
        if metadata: r_band.SetMetadata(metadata)
        # print(r_band.GetMetadata())
        
        r_band.WriteArray(r)
        r_band.FlushCache()
        i_band = None
        r_band = None

    output.FlushCache()
    del output
    output = None
    del dataset
    dataset = None
    
    return target_file


# COMMAND ----------

# DBTITLE 1,Define SOMO35 utility functions

import os
import tempfile
import uuid
from shutil import copyfile

# Linear regression constants (a*x+b) calculated by Artur Gsella to transform O3 to O3P8Hdmax (P8H-dmax_proxy = 13.662793 + 1.096963*[P1D]).
# See: https://taskman.eionet.europa.eu/issues/139681
A = 1.096963
B = 13.662793

# Import Geospatial utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/geoutils.py').read(), 'geoutils.py', 'exec'))

def convert_O3_to_O3P8Hdmax_file(source_file, a=A, b=B, only_manage_daily_average=True, target_file=None):
    """
    Convert the GDAL Dataset with O3 Pollutant to P8Hdmax with linear regression.
    """
    file_path = os.path.dirname(source_file)
    file_name = os.path.basename(source_file)
    file_name, file_ext = os.path.splitext(file_name)
    if not target_file: target_file = os.path.join(file_path.replace('/O3/', '/SOMO35/'), 'CAMS_O3_P8Hdmax_' + file_name[8:]+file_ext)
    print_message(logging.INFO, 'Running "O3_to_O3P8Hdmax": {} -> {}'.format(source_file, target_file))
    
    directory = os.path.dirname(target_file)
    os.makedirs(directory, exist_ok=True)
    
    dataset = gdal.Open(source_file, gdal.GA_ReadOnly)
    convert_O3_to_O3P8Hdmax_dataset(dataset, target_file, a=a, b=b, only_manage_daily_average=only_manage_daily_average)
    dataset = None
    
    return target_file

def convert_O3_to_O3P8Hdmax_dataset(dataset, target_file, a, b, only_manage_daily_average):
    """
    Convert the GDAL Dataset with O3 Pollutant to P8Hdmax with linear regression.
    """
    envelope = GdalDataset(dataset).get_extent()
    srs_wkt = dataset.GetProjection()
    
    bands, cols, rows = dataset.RasterCount, dataset.RasterXSize, dataset.RasterYSize
    if only_manage_daily_average: bands = 1
    
    data_format = dataset.GetRasterBand(1).DataType
    data_type = gdal.GetDataTypeName(data_format)
    pixel_size_x = (envelope[2] - envelope[0]) / float(cols)
    pixel_size_y = (envelope[3] - envelope[1]) / float(rows)
    # print('bands={} rows={} cols={} type={} format={} psize_x={} psize_y={}'.format(bands, rows, cols, data_type, data_format, pixel_size_x, pixel_size_y))
    
    # Create temporary target file and fill the new GDAL Dataset.
    temp_name = str(uuid.uuid1()).replace('-', '')
    temp_file = os.path.join(tempfile.gettempdir(), temp_name) + os.path.splitext(target_file)[1]
    
    driver = gdal.GetDriverByName('Gtiff')
    output = driver.Create(temp_file, cols, rows, bands, data_format, options=[])
    output.SetGeoTransform([envelope[0], pixel_size_x, 0.0, envelope[3], 0.0, -pixel_size_y])
    output.SetProjection(dataset.GetProjection())
    
    for band_index in range(0, bands):
        source_index = 1 + band_index if not only_manage_daily_average else dataset.RasterCount
        target_index = 1 + band_index
        source_band = dataset.GetRasterBand(source_index)
        target_band = output .GetRasterBand(target_index)
        
        metadata = source_band.GetMetadata()
        if metadata.get('units'): metadata['units'] = ''
        if metadata.get('NETCDF_VARNAME'): metadata['NETCDF_VARNAME'] = 'O3P8Hdmax_proxy'
        if metadata.get('standard_name'): metadata['standard_name'] = 'O3P8Hdmax'
        # print(metadata)
        target_band.SetMetadata(metadata)
        
        no_data = source_band.GetNoDataValue()
        target_band.SetNoDataValue(no_data)
        target_band.FlushCache()
        
        # Transform O3 pollutant to P8H-dmax with linear regression.
        raster = source_band.ReadAsArray(0, 0, cols, rows)
        mask_r = (raster != no_data)
        raster[mask_r] = raster[mask_r] * a + b
        target_band.WriteArray(raster)
        
        target_band.FlushCache()
        source_band = None
        target_band = None
        
    # Finish GDAL dataset & resources.
    output.FlushCache()
    output = None
    copyfile(temp_file, target_file)
    os.remove(temp_file)
    
    return target_file


# COMMAND ----------

# DBTITLE 1,Download CAMS rasters

import os
import cdsapi
import yaml
import tempfile
import uuid

try:
    client = cdsapi.Client(url=CDSAPI_URL, key=CDSAPI_KEY)
    
    # Dowload CAMS for each Pollutant.
    for key in pollutants:
        y, m = downloading_date[0:4], downloading_date[5:7]
        
        pollutant = pollutants[key]
        subfolder = 'Ensemble/{}/{}/{}'.format(key, y, m)
        file_name = '/dbfs' + os.path.join(aq_cams_path, '{}/CAMS_{}_{}.tiff'.format(subfolder, key, downloading_date))
        
        print_message(logging.INFO, 'Downloading CAMS of "{}"...'.format(key))
        
        directory = os.path.dirname(file_name)
        os.makedirs(directory, exist_ok=True)
        
        if os.path.exists(file_name):
            print_message(logging.INFO, 'Skipping downloading, the CAMS Dataset ({}) already exists.'.format(file_name), indent=1)
            continue
            
        # Define local temporary filenames, we will add the Mean of all Bands (24 Hours per Day) as a new Band.
        temp_name = str(uuid.uuid1()).replace('-', '')
        temp_cams_file = os.path.join(tempfile.gettempdir(), temp_name+'.nc')
        temp_mean_file = os.path.join(tempfile.gettempdir(), temp_name+'.tiff')
        # print(temp_name)
        # print(temp_cams_file)
        # print(temp_mean_file)
        
        raster = client.retrieve(
            'cams-europe-air-quality-forecasts', 
            {
              'model': 'ensemble',
              'date': downloading_date+'/'+downloading_date,
              'format': 'netcdf',
              'level': '0',
              'type': 'analysis',
              'leadtime_hour': '0',
              'time': times,
              'variable': [pollutant]
            },
            'download.nc'
        )
        result = raster.download(temp_cams_file)
        print_message(logging.INFO, 'Ok! Result={}'.format(result))
        
        # Append the Mean of all Hours (24 Hours per Day) as 25th Band.
        print_message(logging.INFO, 'Appending the Mean of all Bands...')
        append_mean_to_raster_file(
            source_file=temp_cams_file, target_file=temp_mean_file, srid=4326, driver_name='Gtiff', options=[], valid_range=valid_ranges_of_aggregations['AVG']
        )
        
        # Convert the GDAL Dataset with O3 Pollutant to P8Hdmax with linear regression.
        if key == 'O3':
            o3_file_path = os.path.dirname(file_name)
            o3_file_name = os.path.basename(file_name)
            o3_file_name, file_ext = os.path.splitext(o3_file_name)
            o3_target_file = os.path.join(o3_file_path.replace('/O3/', '/SOMO35/'), 'CAMS_O3_P8Hdmax_' + o3_file_name[8:]+file_ext)
            #print(o3_target_file)
            o3_target_file = convert_O3_to_O3P8Hdmax_file(temp_mean_file, a=A, b=B, only_manage_daily_average=True, target_file=o3_target_file)
            
        # Copy the temporary raster file to output path.
        final_file = file_name[5:]
        dbutils.fs.cp('file:' + temp_mean_file, 'dbfs:' + final_file)
        os.remove(temp_cams_file)
        os.remove(temp_mean_file)
        print_message(logging.INFO, 'Ok! Result={}'.format(file_name))
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date, 'delay': delay_of_date})

print_message(logging.INFO, 'Downloadings successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date, 'delay': delay_of_date})
notebook_mgr = None

