# Databricks notebook source
"""
================================================================================
Notebook for generating SOMO35 aggregation from CAMS data (O3).
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/139681
Author   : ahuarte@tracasa.es

================================================================================
"""

# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))

# Mount the 'AQ CAMS' Azure Blob Storage Container as a File System.
aq_cams_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-cams', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# COMMAND ----------


# Source root folder where original CAMS Datasets where downloaded with CDS API.
CAMS_SOURCE_FOLDER = '/dbfs' + aq_cams_path + '/Ensemble'

# Linear regression constants (a*x+b) calculated by Artur Gsella to transform O3 to O3P8Hdmax (P8H-dmax_proxy = 13.662793 + 1.096963*[P1D]).
# See: https://taskman.eionet.europa.eu/issues/139681
A = 1.096963
B = 13.662793

import glob
import os
import numpy as np
import traceback
from shutil import copyfile
import tempfile
import uuid

from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

print('GDAL_VERSION={}'.format(gdal.VersionInfo()))

def convert_O3_to_O3P8Hdmax_file(source_file, a=A, b=B, only_manage_daily_average=True, target_file=None):
    """
    Convert the GDAL Dataset with O3 Pollutant to P8Hdmax with linear regression.
    """
    file_path = os.path.dirname(source_file)
    file_name = os.path.basename(source_file)
    file_name, file_ext = os.path.splitext(file_name)
    if not target_file: target_file = os.path.join(file_path.replace('/O3/', '/SOMO35/'), 'CAMS_O3_P8Hdmax_' + file_name[8:]+file_ext)
    print('Running O3_to_O3P8Hdmax: {} -> {}'.format(source_file, target_file))
    
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

# Process all timeseries of GeoTiffs to transform O3 to P8H-dmax...
try:
    source_pollutant = 'O3'
    source_folder = os.path.join(CAMS_SOURCE_FOLDER, source_pollutant+'/**/CAMS_{}*.tiff'.format(source_pollutant))
    file_count = 0
    
    for file_name in glob.glob(source_folder, recursive=True):
        source_file = file_name
        
        # Ignore input files that are not CAMS O3 GDAL Datasets (e.g. CAMS_O3_2019-01-11.tiff).
        if len(os.path.basename(file_name)) != 23: continue
        
        # Transform O3 pollutant to P8H-dmax with linear regression.
        target_file = convert_O3_to_O3P8Hdmax_file(source_file, a=A, b=B, only_manage_daily_average=True)
        file_count = file_count + 1
        
        # Early stop for DEBUGGING?
        # break
        
    print('OK! FileCount={}'.format(file_count))
    
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    print('ERROR: ' + message)

