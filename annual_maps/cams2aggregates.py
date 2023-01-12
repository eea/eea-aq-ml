# Databricks notebook source
"""
================================================================================
Process CAMS Datasets in Azure Blob storages aggregating some Statistics.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/129254
Author   : ahuarte@tracasa.es

### WARNING:
  Processing of SOMO35 is always using SUM operation and substracting 70 to 
  input data (See "SOMO35_CHANGES_BY_HAND" tag changing behavior by hand).
  
  From 2019 it is calculated from a proxy of "O3_P8Hdmax" using linear 
  regression of O3.

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

# Source root folder where original CAMS Datasets where downloaded with CDS API.
CAMS_SOURCE_FOLDER = '/dbfs' + aq_cams_path + '/Ensemble'
# Target root folder where saving the new CAMS Datasets with new Aggregates and transfomed to EPSG:3035.
AGGR_TARGET_FOLDER = '/dbfs' + aq_cams_path + '/Ensemble'

# Default delay in Days of "Today" Date (It seems the CDS Service does not provide Data for "Today", as much for "Yesterday").
DEFAULT_DAYS_DELAY = 1

# Pollutant settings.
pollutants = {
  'NO2': 'nitrogen_dioxide', 
  'O3': 'ozone',
  'PM10': 'particulate_matter_10um',
  'PM25': 'particulate_matter_2.5um',
  'SO2': 'sulphur_dioxide',
  'SOMO35': 'sum of means of ozone over 35 ppb (daily maximum 8-hour)'
}

valid_ranges_of_aggregations = {
  'AVG': [0, 10000], 
  'AOT40': [0, 1000000],
  'SOMO35': [70.0001, 1000000]
}

# Statistical functions to apply (Mean or Average or Avg, Min, Max, Add or Sum).
operations = [
  'Avg'
]

# Date criterias to apply (year, month, week, day).
criterias = [
  'year'
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
print_message(logging.INFO, 'Processing aggregates:')
print_message(logging.INFO, ' + Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Delay of Date: {}'.format(delay_of_date))
print_message(logging.INFO, ' + Pollutants:')
for key in pollutants: print_message(logging.INFO, '\t{}: {}'.format(key, pollutants[key]))


# COMMAND ----------

# DBTITLE 1,Prepare GDAL environment

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

GDAL_RESAMPLE_ALGORITHM = gdal.GRA_Bilinear

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

gdal.SetConfigOption('VSI_CACHE', 'TRUE')
gdal.SetConfigOption('VSI_CACHE_SIZE', '100000000')
gdal.SetConfigOption('GDAL_CACHE', '100000000')

print_message(logging.INFO, 'GDAL_VERSION={}'.format(gdal.VersionInfo()))

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

def create_geometry_from_bbox(x_min, y_min, x_max, y_max):
    """
    Returns the Geometry from the specified BBOX.
    INFO:
        This function adds four mid-point vertices to the Geometry because the transform 
        between EPSG:4326 and EPSG:3035 generates curvilinear borders.
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_min + 0.5*(x_max-x_min), y_min)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_max, y_min + 0.5*(y_max-y_min))
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_max - 0.5*(x_max-x_min), y_max)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_min, y_max - 0.5*(y_max-y_min))
    ring.AddPoint(x_min, y_min)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly
  
def warp_dataset_to_eeagrid(raster_file, dataset, resample_alg=gdal.GRA_Bilinear, driver_name='Gtiff'):
    """
    Warp Dataset to the EEA Reference GRID (EPSG:3035):
    https://www.eea.europa.eu/data-and-maps/data/eea-reference-grids-2
    """
    envelope = get_envelope(dataset)
    band = dataset.GetRasterBand(1)
    no_data = band.GetNoDataValue()
    band = None
    
    # Transform Bounds from source SRS (EPSG:4326) to target SRS (EPSG:3035).
    source_crs = osr.SpatialReference()
    if hasattr(source_crs, 'SetAxisMappingStrategy'): source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    source_crs.ImportFromEPSG(4326)
    target_crs = osr.SpatialReference()
    if hasattr(target_crs, 'SetAxisMappingStrategy'): target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_crs.ImportFromEPSG(3035)
    
    transform = osr.CoordinateTransformation(source_crs, target_crs)
    envelope = create_geometry_from_bbox(envelope[0], envelope[1], envelope[2], envelope[3])
    envelope.Transform(transform)
    envelope = envelope.GetEnvelope()
    
    # Clamp Bounds to 1x1km and conforming the EEA Reference GRID
    x_min = int(envelope[0] / 1000.0) * 1000.0
    y_min = int(envelope[2] / 1000.0) * 1000.0
    x_max = int(envelope[1] / 1000.0) * 1000.0 + 1000.0
    y_max = int(envelope[3] / 1000.0) * 1000.0 + 1000.0
    envelope = [x_min, y_min, x_max, y_max]
    
    # Define the Warping options for resampling and reprojection.
    options = gdal.WarpOptions(format=driver_name,
                               outputBounds=envelope,
                               resampleAlg=resample_alg,
                               xRes=1000,
                               yRes=1000,
                               srcSRS='EPSG:4326',
                               srcNodata=no_data,
                               dstSRS='EPSG:3035',
                               dstNodata=no_data,
                               copyMetadata=True,
                               creationOptions=['TILED=YES','COMPRESS=DEFLATE','BLOCKXSIZE=256','BLOCKYSIZE=256'])

    return gdal.Warp(raster_file, dataset, options=options)

def get_dataset_props(dataset, default_srid=4326):
    """
    Returns the Metadata of the specified GDAL Dataset.
    """
    projection_wkt = dataset.GetProjectionRef()
    
    crs = osr.SpatialReference()
    if projection_wkt: crs.ImportFromWkt(projection_wkt)
    else: crs.ImportFromEPSG(default_srid)
    
    bands_metadata = []
    for i in range(0, dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)
        bands_metadata.append(
            {'metadata': band.GetMetadata(), 'no_data': band.GetNoDataValue(), 'unit_type': band.GetUnitType()}
        )
        band = None
        
    dataset_props = {
        'rasterxsize': dataset.RasterXSize,
        'rasterysize': dataset.RasterYSize,
        'rastercount': dataset.RasterCount,
        'envelope': get_envelope(dataset),
        'crs': crs,
        'metadata': dataset.GetMetadata(),
        'bands': bands_metadata
    }
    return dataset_props

def save_raster_to_file(raster_file, raster, dataset_props, driver_name='Gtiff', options=[], warp_to_eeagrid=True, resample_alg=gdal.GRA_Bilinear):
    """
    Write a new GDAL Dataset from the specified parameters.
    """
    import tempfile
    import uuid
    import os
    
    temp_name = str(uuid.uuid1()).replace('-', '')
    
    # DataType codes (For Numpy & Spark) of available data-types of a GDAL Dataset.
    GDT_DataTypeCodes = dict([
        ('unknown', gdal.GDT_Unknown),
        ('byte', gdal.GDT_Byte),
        ('uint8', gdal.GDT_Byte),
        ('uint16', gdal.GDT_UInt16), ('int16', gdal.GDT_Int16),
        ('uint32', gdal.GDT_UInt32), ('int32', gdal.GDT_Int32), ('int64', gdal.GDT_Float64),
        ('float32', gdal.GDT_Float32), ('float64', gdal.GDT_Float64),
        ('cint16', gdal.GDT_CInt16), ('cint32', gdal.GDT_CInt32), ('cfloat32', gdal.GDT_CFloat32),
        ('cfloat64', gdal.GDT_CFloat64)
    ])
        
    envelope = dataset_props['envelope']
    metadata = dataset_props['metadata']
    crs = dataset_props['crs']
    
    bands, rows, cols = raster.shape
    data_type = str(raster.dtype)
    data_format = GDT_DataTypeCodes[data_type]
    pixel_size_x = (envelope[2] - envelope[0]) / float(cols)
    pixel_size_y = (envelope[3] - envelope[1]) / float(rows)
    # print('bands={} rows={} cols={} type={} format={} psize_x={} psize_y={} no_data={} file={}'.format(bands, rows, cols, data_type, data_format, pixel_size_x, pixel_size_y, dataset_props['bands'][0], raster_file))
    
    # Manage current CRS.
    spatial_ref = crs
    if hasattr(spatial_ref, 'SetAxisMappingStrategy'): spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # Write raster file using a temporary folder, we'll copy it to output path later.
    with tempfile.TemporaryDirectory() as temp_dir:
        # print(temp_dir)
        temp_file = os.path.join(temp_dir, temp_name) + os.path.splitext(raster_file)[1]
        # print(temp_file)
        
        # When warping to the EEA 3035 GRID, we work with a temporary MEM Dataset before writing the final file.
        if warp_to_eeagrid:
            driver = gdal.GetDriverByName('MEM')
            dataset = driver.Create(temp_name, cols, rows, bands, data_format)
        else:
            driver = gdal.GetDriverByName(driver_name)
            dataset = driver.Create(temp_file, cols, rows, bands, data_format, options=options)
            
        dataset.SetGeoTransform([envelope[0], pixel_size_x, 0.0, envelope[3], 0.0, -pixel_size_y])
        dataset.SetProjection(crs.ExportToWkt())
        if metadata is not None: dataset.SetMetadata(metadata)
        
        for band in range(0, bands):
            r = raster[band]
            r_band = dataset.GetRasterBand(band + 1)            
            m_band = dataset_props['bands']
            m_band = m_band[band]
            if m_band['metadata'] is not None: r_band.SetMetadata(m_band['metadata'])            
            if m_band['unit_type'] is not None: r_band.SetUnitType(m_band['unit_type'])
            if m_band['no_data'] is not None: r_band.SetNoDataValue(m_band['no_data'])
            r_band.WriteArray(r)
            r_band.FlushCache()
            r_band = None
            
        dataset.FlushCache()
        
        # Warp Dataset to the EEA 3035 GRID.
        if warp_to_eeagrid:
            warp_dataset_to_eeagrid(temp_file, dataset, resample_alg=resample_alg, driver_name=driver_name)
            
        dataset = None
        raster = None
        # print('file_size={}'.format(os.stat(temp_file).st_size))
        
        # Copy the temporary raster file to output path.
        if raster_file.startswith('/dbfs'):
            final_file = raster_file[5:]
            dbutils.fs.cp('file:' + temp_file, 'dbfs:' + final_file)
        else:
            import shutil
            shutil.copy2(temp_file, raster_file)
            
        # print('Copy ok! {} -> {}'.format(temp_file, raster_file))
    
    return raster_file


# COMMAND ----------


import numpy as np
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

def raster_to_dataframe(dataset, column_name='cams', filter_nodata=True, valid_range=None):
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
    
    # print('transform={}'.format(transform))
    # print('cols={}, rows={}'.format(cols, rows))
    # print('raster.size={}, raster.shape={}, data_type={}, no_data={} XY=({},{})'.format(raster.size, raster.shape, raster.dtype, no_data, transform[0], transform[3]))
    
    column_names = [column_name] if dataset.RasterCount == 1 else [column_name+'_'+str(n) for n in range(0,dataset.RasterCount)]
    column_names = ['id'] + column_names
    
    a = np.arange(raster.size, dtype=np.int32)
    b = raster.flatten()
    t = np.array([a, b]).transpose()
    # print('t.shape={}'.format(t.shape))
    
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


#
# Define extra GDAL functions to enable merging between CAMS Datasets with different spatial schemas.
# Check change from 2019/06/11 (24 x 400 x 700) to 2019/06/12 (24 x 420 x 700).
# https://taskman.eionet.europa.eu/issues/129656#note-13
#

def get_spatial_schema(dataset):
    """
    Returns the Spatial Schema of the specified GDAL Dataset.
    """
    envelope = get_envelope(dataset)
    
    spatial_schema = {
        'rasterxsize': dataset.RasterXSize,
        'rasterysize': dataset.RasterYSize,
        'envelope': envelope,
        'envelope_r': [round(v, 5) for v in envelope],
    }
    return spatial_schema

def same_spatial_schema(schema_1, schema_2):
    """
    Returns if the specified Datasets Schema are spatially equal.
    """
    raster_x_size_1, raster_x_size_2 = schema_1['rasterxsize'], schema_2['rasterxsize']
    if raster_x_size_1 != raster_x_size_2: return False
    raster_y_size_1, raster_y_size_2 = schema_1['rasterysize'], schema_2['rasterysize']
    if raster_y_size_1 != raster_y_size_2: return False
    
    envelope_r1, envelope_r2 = schema_1['envelope_r'], schema_2['envelope_r']
    return envelope_r1 == envelope_r2

def get_reading_window(envelope, width, height, bbox, clamp_bounds=True):
    """
    Calculates a window in pixel coordinates for which data will be read from a raster.
    """
    pixel_size_x = (envelope[2] - envelope[0]) / width
    pixel_size_y = (envelope[3] - envelope[1]) / height
    
    # If these coordinates wouldn't be rounded here, rasterio.io.DatasetReader.read would round them in the same way
    left   = round((bbox[0] - envelope[0]) / pixel_size_x)
    top    = round((envelope[3] - bbox[3]) / pixel_size_y)
    right  = width  - round((envelope[2] - bbox[2]) / pixel_size_x)
    bottom = height - round((envelope[1] - bbox[1]) / pixel_size_y)
    
    if clamp_bounds:
        left   = max(left, 0)
        top    = max(top, 0)
        right  = min(right, width)
        bottom = min(bottom, height)
    
    return left, top, right, bottom

def get_envelope_of_window(envelope, width, height, left, top, right, bottom):
    """
    Returns the spatial Envelope of the specified GRID window.
    """
    pixel_size_x = (envelope[2] - envelope[0]) / width
    pixel_size_y = (envelope[3] - envelope[1]) / height

    x_min = envelope[0] + (left   * pixel_size_x)
    y_max = envelope[3] - (top    * pixel_size_y)
    x_max = envelope[0] + (right  * pixel_size_x)
    y_min = envelope[3] - (bottom * pixel_size_y)
    
    return x_min, y_min, x_max, y_max


# COMMAND ----------

# DBTITLE 1,Process CAMS rasters (for each Pollutant & Operation & Criteria)

import glob
import os
import numpy as np
import re

# Import 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))

def parse_date_of_date(date: str):
    """
    Returns the (year, month, day, week) of the specified Date (yyyy-MM-dd).
    """
    y, m, d = date[0:4], date[5:7], date[8:10]
    date_ob = datetime.date(int(y), int(m), int(d))
    w = date_ob.strftime('%W')
    return date, y,m,d, w    

def parse_date_of_file(file_name: str):
    """
    Returns the (year, month, day, week) of the specified file name.
    It assumes that the filename ends with the pattern 'YYYY-MM-dd.xxx'.
    """
    base_name = os.path.basename(file_name)
    date = os.path.splitext(base_name)[0]
    date = date[-10:]
    return parse_date_of_date(date)

def pattern_of_date(date: str, criteria: str):
    """
    Returns the Regex pattern of the Date according to the specified Criteria.
    """
    date, y,m,d, w = parse_date_of_date(date)
    criteria = criteria.upper()
    if criteria == 'YEAR': return '({})'.format(y) + '-([0-9]{2})-([0-9]{2})'
    if criteria == 'MONTH': return '({}-{})'.format(y, m) + '-([0-9]{2})'
    if criteria == 'WEEK': return 'week-' + w
    return date
    
def get_operation_function(operation: str):
    """
    Provides some cumulative-like statistical functions (sum, average, min, max) for numpy streams.
    They are the simplest functions because of we can accumulate all chuncks of input data.
    """
    operation = operation.upper()
    if operation in ['ADD', 'SUM', 'MEAN', 'AVERAGE', 'AVG']: return np.add
    if operation in ['MIN']: return np.minimum
    if operation in ['MAX']: return np.maximum
    return None

def get_output_file_name(root_folder: str, pollutant_key: str, operation: str, criteria: str, date: str):
    """
    Returns the Output file name of the new Aggregate CAMS Dataset.
    """
    date, y,m,d, w = parse_date_of_date(date)
    criteria = criteria.upper()
    
    subfolder = '{}/{}/'.format(pollutant_key, y)
    if criteria in ['MONTH','DAY']: subfolder += (m + '/')
    if criteria in ['WEEK']: subfolder += 'Weeks/'
    if criteria == 'YEAR' : file_name = 'CAMS_{}_{}_{}-XX-XX'.format(pollutant_key, operation.lower(), y)
    if criteria == 'MONTH': file_name = 'CAMS_{}_{}_{}-{}-XX'.format(pollutant_key, operation.lower(), y, m)
    if criteria == 'WEEK' : file_name = 'CAMS_{}_{}_{}-wk-{}'.format(pollutant_key, operation.lower(), y, w)
    if criteria == 'DAY'  : file_name = 'CAMS_{}_{}_{}-{}-{}'.format(pollutant_key, operation.lower(), y, m, d)    
    return '{}/{}{}.tiff'.format(root_folder, subfolder, file_name)

# print(get_output_file_name(AGGR_TARGET_FOLDER, 'NO2', 'Avg', 'Year', downloading_date))
# print(get_output_file_name(AGGR_TARGET_FOLDER, 'NO2', 'Avg', 'Month', downloading_date))
# print(get_output_file_name(AGGR_TARGET_FOLDER, 'NO2', 'Avg', 'Week', downloading_date))
# print(get_output_file_name(AGGR_TARGET_FOLDER, 'NO2', 'Avg', 'Day', downloading_date))

# Process for each Pollutant...
try:
    for key in pollutants:
        pollutant = pollutants[key]
        
        # Processing of SOMO35 is always using SUM operation and substracting 70 to input data (See "SOMO35_CHANGES_BY_HAND" tag changing behavior by hand).
        # From 2019 it is calculated from a proxy of "O3_P8Hdmax" using linear regression of O3.
        key_file_pattern = key if key != 'SOMO35' else 'O3_P8Hdmax'
        
        print_message(logging.INFO, 'Processing Pollutant "{}" ({})...'.format(key, downloading_date))
        # if key != 'SOMO35': continue
        
        # For each Operation...
        for operation in operations:
            dataset_props = None
            dataset_schema = None
            file_count = 0
            
            print_message(logging.INFO, ' + Operation "{}":'.format(operation.upper() if key != 'SOMO35' else 'SUM of (value-70)'))
            
            # Numpy function to apply.
            operation_func = get_operation_function(operation)
            # Initialize partial results for each Date criteria.
            criterias_args = dict([(c.upper(), {'criteria': c, 'pattern': re.compile(pattern_of_date(downloading_date, c)), 'raster': None, 'rcount': None, 'count': 0}) for c in criterias])
            # print(operation_func)
            # print(criterias_args)
            
            source_folder = os.path.join(CAMS_SOURCE_FOLDER, key+'/**/CAMS_{}*.tiff'.format(key_file_pattern))
            source_re = re.compile('CAMS_' + key_file_pattern.upper() + '_([0-9]{4})-([0-9]{2})-([0-9]{2})')
            # print(source_folder)
            
            # For each source file...
            for file_name in glob.glob(source_folder, recursive=True):
                f_name = os.path.splitext(os.path.basename(file_name))[0]
                f_name = f_name.upper()
                
                # We have to check if this File represents a valid individual CAMS Dataset.
                # print('{} {} {} {}'.format(f_name, f_name.startswith(pollutant), len(f_name), source_re.match(f_name)))
                if not source_re.match(f_name): continue
                
                # Ok, continue...
                date, y,m,d, w = parse_date_of_file(file_name)
                current_schema = None
                update_schema = False
                no_data = -999.0
                raster = None
                
                # print('{} -> {}: {},{},{},{}'.format(file_name, date, y,m,d,w))
                # if file_count==2: break
                
                # For each Date criteria...
                for criteria_key in criterias_args:
                    criteria_arg = criterias_args[criteria_key]
                    criteria = criteria_arg['criteria']
                    pattern = criteria_arg['pattern']
                    match_p = 'week-'+w if criteria_key == 'WEEK' else date
                    
                    # Do we have to process current NetCDF file?
                    # print_message(logging.INFO, criteria_key+' => '+str(pattern)+' match with? => '+match_p, indent=2)
                    if not pattern.match(match_p): continue
                    # print_message(logging.INFO, 'YEAH!', indent=2)
                    
                    # Read all raster streams in one unique step (24 bands X cols=700 X rows=420 / Or 25th Mean Band), they easily fit in RAM.
                    if raster is None:
                        print_message(logging.INFO, '{}, Date=({})'.format(file_name, (date,y,m,d,w)), indent=1)
                        
                        dataset = gdal.Open(file_name, gdal.GA_ReadOnly)
                        current_schema = get_spatial_schema(dataset)
                        raster_count = dataset.RasterCount
                        file_count += 1
                        
                        band_r = dataset.GetRasterBand(1)
                        no_data = band_r.GetNoDataValue()
                        band_r = None
                        
                        # Sum all bands (24 hours) as one unique? or using the 25th with the Mean of all 24 Hours.
                        if raster_count == 25:
                            band_r = dataset.GetRasterBand(25)
                            raster = band_r.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
                            band_r = None
                        else:
                            raster = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
                            band_count = 1 if raster.ndim < 3 else raster.shape[0]
                            if raster.ndim < 3: raster = raster[np.newaxis,...]
                            #
                            # Apply valid range of Values.
                            valid_range = valid_ranges_of_aggregations.get(key, valid_ranges_of_aggregations['AVG'])
                            min_valid_value = valid_range[0]
                            max_valid_value = valid_range[1]
                            mask_r = (raster >= min_valid_value) & (raster <= max_valid_value)
                            # print('mask_valid={}'.format(mask_r.sum()))
                            raster[~mask_r] = no_data
                            #
                            mask_r = (raster != no_data)
                            # print('mask_ok={}'.format(mask_r.sum()))
                            # temp_r = raster
                            raster = np.divide(np.sum(raster, axis=0, where=mask_r), float(band_count), out=raster, where=mask_r)
                            raster = np.squeeze(raster, axis=0)
                            # mask_t = (raster != no_data)
                            # print('mask_test={}'.format(mask_t.sum()))
                            # print('mask_equal={}'.format(np.equal(temp_r, raster).sum()))
                            
                        # band_count = 1 if raster.ndim < 3 else raster.shape[0]
                        # print(' > source_raster_A: shape={}, mean={} band_count={}'.format(raster.shape, raster[(raster != no_data)].mean(), band_count))
                        raster = raster[np.newaxis,...]
                        # print(' > source_raster_B: shape={}, mean={} band_count={}'.format(raster.shape, raster[(raster != no_data)].mean(), raster.shape[0]))
                        
                        # Get Dataset properties to use later when saving.
                        if dataset_props is None:
                            # print_message(logging.INFO, 'Reading Dataset properties', indent=2)
                            dataset_props  = get_dataset_props(dataset)
                            dataset_schema = current_schema
                            # print_message(logging.INFO, str(dataset_props), indent=2)
                            #
                        elif not same_spatial_schema(dataset_schema, current_schema):
                            update_schema  = True
                            dataset_props  = get_dataset_props(dataset)
                            print_message(logging.WARNING, 'Warning: Incoherent new Spatial schema, we have to redefine the Window of pixels!', indent=1)
                            
                        del dataset
                        dataset = None
                        
                    # Substracting 70 to "O3_P8Hdmax" to convert to SOMO35 pollutant (###Tag: SOMO35_CHANGES_BY_HAND).
                    if key == 'SOMO35':
                        mask_r = (raster != no_data)
                        raster[mask_r] -= float(70.0)
                    
                    # Verify that new Spatial Schema is coherent with current managed, oterwise redefine the Window of Pixels.
                    if update_schema:
                        envelope_1 = dataset_schema['envelope']
                        envelope_2 = current_schema['envelope']
                        left, top, right, bottom = get_reading_window(envelope_2, current_schema['rasterxsize'], current_schema['rasterysize'], envelope_1, clamp_bounds=True)
                        # print('left={}, top={}, right={}, bottom={}'.format(left, top, right, bottom))
                        
                        r = criteria_arg['raster']
                        c = criteria_arg['rcount']
                        # print(r.shape, r.mean())
                        # print(c.shape, c.mean())

                        mask_r = raster != no_data
                        temp_c = np.zeros(raster.shape, dtype=np.int32)
                        temp_c[mask_r] = 1
                        temp_r = raster.copy()
                        # print(temp_r.shape, temp_r.mean())
                        # print(temp_c.shape, temp_c.mean())
                        
                        temp_r[...,top:bottom,left:right] = r
                        temp_c[...,top:bottom,left:right] = c
                        # print(temp_r.shape, temp_r.mean())
                        # print(temp_c.shape, temp_c.mean())
                        # print(temp_r[...,top:bottom,left:right].shape, temp_r[...,top:bottom,left:right].mean())
                        # print(temp_c[...,top:bottom,left:right].shape, temp_c[...,top:bottom,left:right].mean())
                        
                        criteria_arg['raster'] = temp_r
                        criteria_arg['rcount'] = temp_c                        
                    
                    # Process raster streams with Numpy!
                    r = criteria_arg['raster']
                    if r is None:
                        mask_r = raster != no_data
                        rcount = np.zeros(raster.shape, dtype=np.int32)
                        rcount[mask_r] = 1
                        criteria_arg['raster'] = raster
                        criteria_arg['rcount'] = rcount
                    else:
                        mask_r = raster != no_data
                        rcount = criteria_arg['rcount']
                        # print(r     .shape, r     [(r != no_data)].mean())
                        # print(rcount.shape, rcount[(r != no_data)].mean())
                        # print(raster.shape, raster[mask_r].mean())
                        # print(mask_r.shape, mask_r[mask_r].mean())
                        
                        rcount[mask_r] += 1
                        # print(rcount.shape, rcount.mean())
                        
                        temp_m = (raster != no_data) & (r != no_data)
                        temp_r = operation_func(r, raster, out=r, where=temp_m)
                        # print(temp_r.shape, temp_r[(temp_r != no_data)].mean())
                        # print(temp_m.shape, temp_m.mean())
                        
                        temp_m = (raster != no_data) & (r == no_data)
                        np.copyto(temp_r, raster, where=temp_m)
                        # print(temp_r.shape, temp_r[(temp_r != no_data)].mean())
                        # print(temp_m.shape, temp_m.mean())
                        
                        criteria_arg['raster'] = temp_r
                        criteria_arg['rcount'] = rcount
                                            
                    criteria_arg['count'] = criteria_arg['count'] + 1
                    
                dataset_schema = current_schema
                raster = None
            
            # Finish Operation, save new Raster files...
            if file_count > 0:
                print_message(logging.INFO, 'Finishing operation...', indent=1)
                
                for criteria_key in criterias_args:
                    criteria_arg = criterias_args[criteria_key]
                    r = criteria_arg['raster']
                    if r is None: continue
                    
                    # Divide by number of passes accumulated according to current operation (###Tag: SOMO35_CHANGES_BY_HAND)?
                    if operation.upper() in ['MEAN', 'AVERAGE', 'AVG'] and key != 'SOMO35':
                        c = criteria_arg['rcount']
                        r = np.divide(r, c, out=r, where=(c!=0))
                        # print(r.shape, r[(r != no_data)].mean())
                        # print(c.shape, c.mean())
                        
                    # print(' > criteria={}, np.shape={}, np.mean={}'.format(criteria_key, r.shape, r[(r != no_data)].mean()))
                    
                    # Fix metadata of SOMO35 Datasets (###Tag: SOMO35_CHANGES_BY_HAND).
                    if key == 'SOMO35' and dataset_props.get('bands'):
                        m_bands = dataset_props['bands']
                        
                        for index in range(0, len(m_bands)):
                            m_band = m_bands[index]
                            metadata = m_band.get('metadata')
                            if metadata and metadata.get('NETCDF_VARNAME'): metadata['NETCDF_VARNAME'] = 'SOMO35'
                            if metadata and metadata.get('standard_name'): metadata['standard_name'] = 'SOMO35'
                            
                    # Output new raster file...
                    raster_file = get_output_file_name(AGGR_TARGET_FOLDER, key, operation, criteria_arg['criteria'], downloading_date)
                    print_message(logging.INFO, ' + Writing "{}"...'.format(raster_file), indent=1)
                    #
                    save_raster_to_file(raster_file=raster_file, raster=r, dataset_props=dataset_props, warp_to_eeagrid=True, resample_alg=GDAL_RESAMPLE_ALGORITHM)
                    print_message(logging.INFO, 'Ok!', indent=2)
                    #
                    # Output as Parquet file...
                    parquet_file = os.path.join(os.path.dirname(raster_file), os.path.splitext(os.path.basename(raster_file))[0] + '.parquet')
                    print_message(logging.INFO, ' + Writing "{}"...'.format(parquet_file), indent=1)
                    #
                    dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
                    temp_pd = raster_to_dataframe(dataset, column_name='cams', filter_nodata=True, valid_range=None)
                    dataset = None
                    temp_pd['GridNum1km'] = np.int64(CalcGridFunctions.calcgridnum_np(x=temp_pd['x'] - 500, y=temp_pd['y'] + 500))
                    temp_pd['Year'] = np.int32(downloading_date[0:4])
                    temp_pd.to_parquet(parquet_file, compression='snappy', index=False)
                    temp_pd = None
                    print_message(logging.INFO, 'Ok!', indent=2)
                    
            print_message(logging.INFO, 'Operation done! Files={}'.format(file_count), indent=1)
            
        print_message(logging.INFO, 'Pollutant done!', indent=1)
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date, 'delay': delay_of_date})

print_message(logging.INFO, 'Aggregates successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date, 'delay': delay_of_date})    
notebook_mgr = None

