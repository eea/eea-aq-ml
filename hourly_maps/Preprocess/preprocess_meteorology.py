# Databricks notebook source
"""
================================================================================
Preprocess Meteorology data already in azure to transform to EPSG3035, assign a Gridnum and store it as parquet.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157037
Author   : mgonzalez@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
aq_meteo = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-meteorology', 
  sas_key = 'sv=2021-10-04&st=2022-12-20T10%3A39%3A26Z&se=2030-11-21T10%3A39%3A00Z&sr=c&sp=racwdl&sig=DT1L2jy%2B8KcECwCiDaNO%2F1bIsQCytDmKhNz9i2r4ZHg%3D'
)
# Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
aq_predictions_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-predictions', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)

context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

from osgeo import gdal
from osgeo import osr
from datetime import datetime

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

print('INFO: GDAL_VERSION={}'.format(gdal.VersionInfo()))

import numpy as np
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

def raster_to_dataframe(dataset, column_name='x', filter_nodata=True, valid_range=None):
    """
    Convert the specified GDAL Dataset to a Spark Dataframe.
    """
    transform = dataset.GetGeoTransform()
    cols,rows = dataset.RasterXSize, dataset.RasterYSize
    band_r    = dataset.GetRasterBand(1)
    no_data   = band_r.GetNoDataValue()
    band_r    = None
    raster    = dataset.ReadAsArray(0, 0, cols, rows)
    
    # Apply valid range of Values.
    if valid_range: 
        min_valid_value = valid_range[0]
        max_valid_value = valid_range[1]
        mask_r = (raster >= min_valid_value) & (raster <= max_valid_value)
        raster[~mask_r] = no_data
    
    print('transform={}'.format(transform))
    print('cols={}, rows={}'.format(cols, rows))
    print('raster.size={}, raster.shape={}, data_type={}, no_data={} XY=({},{})'.format(raster.size, raster.shape, raster.dtype, no_data, transform[0], transform[3]))
    
    column_names = [column_name] if dataset.RasterCount == 1 else [column_name+'_'+str(n) for n in range(0,dataset.RasterCount)]
    column_names = ['id'] + column_names
    
    a = np.arange(raster.size, dtype=np.int32)
    b = raster.flatten()
    t = np.array([a, b]).transpose()
    
    print('t.shape={}'.format(t.shape))
    
    temp_pd = pd.DataFrame(t, columns=column_names)
    if filter_nodata: temp_pd = temp_pd[temp_pd[column_name] != no_data]
    
    # Append XY location.
    temp_pd['r'] = (temp_pd['id'] / cols).astype(np.int32)
    temp_pd['c'] = (temp_pd['id'] - (temp_pd['r'] * cols)).astype(np.int32)
    temp_pd['x'] = (temp_pd['c'] * transform[1]) + (transform[0] + (0.5 * transform[1]))
    temp_pd['y'] = (temp_pd['r'] * transform[5]) + (transform[3] + (0.5 * transform[5]))
    """
    debg_pd = temp_pd[['r', 'c']].min(axis=0)
    r_min, c_min = debg_pd['r'], debg_pd['c']
    debg_pd = temp_pd[['r', 'c']].max(axis=0)
    r_max, c_max = debg_pd['r'], debg_pd['c']
    bbox = [r_min, c_min, r_max, c_max]
    print('RC_BBOX={}'.format(bbox))
    """
    temp_pd.drop(['r', 'c'], axis='columns', inplace=True)
    return temp_pd


# COMMAND ----------


# Import 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))

import glob
import os
import numpy as np
import pyarrow as pa
from pyarrow import fs
import pyarrow.parquet as pq

meteo_variables = {
  '10m_u_component_of_wind',
  '10m_v_component_of_wind',
  '2m_temperature',
  'surface_net_solar_radiation',
  'surface_pressure',
  'total_precipitation'
}

for var in meteo_variables:
    source_folder = '/dbfs{}/Reanalysis/{}/**/ERA5_{}**.tiff'.format(aq_meteo, var, var)
    target_folder = '/dbfs{}/Reanalysis/{}/**/ERA5_{}-month.parquet'.format(aq_meteo, var, var)
    file_count = 0
    print(source_folder)
    dfs = pd.DataFrame()
    for file_name in glob.glob(source_folder, recursive=True):
        parquet_file = os.path.join(os.path.dirname(file_name), os.path.splitext(os.path.basename(file_name))[0] + '.parquet')
        print(file_name + ' -> ' + parquet_file)
        print(os.path.dirname(file_name))
        print(os.path.splitext(os.path.basename(file_name))[0])
        
        # Load Geotiff and convert to Pandas object.
        dataset = gdal.Open(file_name, gdal.GA_ReadOnly)
        meteo_pd = raster_to_dataframe(dataset, column_name=var, filter_nodata=True)
        dataset = None
        
        # Append GridNum1km attribute to the Pandas object.
        meteo_pd['GridNum1km'] = np.int64(CalcGridFunctions.calcgridnum_np(x=meteo_pd['x'] - 500, y=meteo_pd['y'] + 500))        
        file = os.path.splitext(os.path.basename(file_name))[0]
        splited_text = file.split('_')
        string_date = splited_text[-2] + ' ' + splited_text[-1]
        print(string_date)
        dt = datetime.strptime(string_date , '%Y%m%d %H%M')         
        meteo_pd['dt'] = dt.strftime("%Y-%m-%d %H:%M:%S")
        # Save to Parquet file.
        meteo_pd.to_parquet(parquet_file, compression='gzip')
        month_target = '/dbfs{}/Reanalysis/{}/{}/{}/ERA5_{}_{}-{}.parquet'.format(aq_meteo, var, dt.year, dt.month,var, dt.year, dt.month)
        meteo_pd.to_parquet(month_target, compression='gzip')
        #break
        
print('ok!')

