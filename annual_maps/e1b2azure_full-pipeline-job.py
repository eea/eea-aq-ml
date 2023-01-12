# Databricks notebook source
"""
================================================================================
Process ROD674/E1b Datasets in Azure Blob storages aggregating some Statistics.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/140367
Author   : ahuarte@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))

# Mount the 'AQ ROD674/E1b' Azure Blob Storage Container as a File System.
aq_e1b_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-e1b', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# COMMAND ----------

# DBTITLE 1,Main variables & constants

# Source root folder where original ZIP ROD674/E1b Datasets are localted.
E1B_SOURCE_FOLDER = '/dbfs' + aq_e1b_path + '/Input'
# Target root folder where output Tables.
E1B_TARGET_FOLDER = '/dbfs' + aq_e1b_path + '/Ensemble'

# List of Years processed, to finally update Yearly-average Datasets with ALL attributes.
list_of_years_processed = dict()

# Pollutant settings.
pollutants = {
  'NO2': 'nitrogen_dioxide', 
  'O3': 'ozone',
  'PM10': 'particulate_matter_10um',
  'PM25': 'particulate_matter_2.5um',
  'BaP': 'benzo[a]_pyrene'
}

# Delete input files to save space disc?
DELETE_INPUT_FILES = True


# COMMAND ----------

# DBTITLE 1,Prepare DataFactory environment

import traceback
import json
import logging
import os
import datetime

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_e1b_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, ' + Input:  ' + E1B_SOURCE_FOLDER)
print_message(logging.INFO, ' + Output: ' + E1B_TARGET_FOLDER)


# COMMAND ----------

# DBTITLE 1,Prepare GDAL environment

from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')

print('INFO: GDAL_VERSION={}'.format(gdal.VersionInfo()))


# COMMAND ----------

# DBTITLE 1,Function to transform input E1B datasets (Geotiff raster files) to Dataframes

# Import and register 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
from pyspark.sql.types import LongType

# Import EEA Dataflow engine.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/datapipeline.py').read(), 'datapipeline.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/geodatapipeline.py').read(), 'geodatapipeline.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/geoutils.py').read(), 'geoutils.py', 'exec'))


# COMMAND ----------


import os
import fnmatch
import re
from zipfile import ZipFile

def e1b_to_dataframe(file_name, gridnum_attrib='GridNum1km', output_pixel_size_x=1000.0, output_pixel_size_y=1000.0, output_no_data=-999.0, append_aggr=True, ignore_pattern_file_name=''):
    """
    Transform the specified input E1B datasets (Geotiff raster files) to Dataframes
    """
    import pyspark.sql.functions as F
    
    match_pattern = fnmatch.translate('*.tif')
    re_ob = re.compile(match_pattern)
    basic_columns = [gridnum_attrib, 'x','y']
    raster_count = 0
    result_pd = None
    
    with ZipFile(file_name, 'r') as zf:
        for zc in zf.infolist():
            if re_ob.match(zc.filename):
                raster_file_name = '/vsizip/' + file_name + '/' + zc.filename
                
                field_name = os.path.splitext(os.path.basename(zc.filename))[0]
                field_name = field_name.replace('.', '_')
                field_name = field_name.replace('-', '_')
                field_name = field_name.replace(' ', '_')
                
                if ignore_pattern_file_name and field_name.find(ignore_pattern_file_name) != -1:
                    print_message(logging.INFO, '> Ignoring Dataset, its name does not match the specified Pattern ("{}").'.format(field_name), indent=1)
                    continue
                
                """
                if field_name not in ['2019_100m_8_P1Y_4764_GB_Model_70', '2019_100m_8_P1Y_5111_MDL_ATMOSTREET_00008']:
                    continue
                ""
                if field_name not in ['2019_100m_8_P1Y_5111_MDL_ATMOSTREET_00008']:
                    continue
                ""
                if field_name.find('_MDL_')==-1 or field_name.find('_NO2_')==-1:
                    continue
                ""
                if field_name.find('_MOD_OBE_')==-1:
                    continue
                """
                print_message(logging.INFO, '> Processing "{}"...'.format(zc.filename), indent=1)
                
                # print('Filename: {f}, origsize: {uc}, compressed_size: {s}, Rasterfile: {rf}'.format(f=zc.filename, s=zc.compress_size, uc=zc.file_size, rf=raster_file_name))
                dataset = gdal.Open(raster_file_name, gdal.GA_ReadOnly)
                temp_ob = GdalDataset(dataset)
                
                srid = temp_ob.get_spatial_srid()
                envelope = temp_ob.get_extent()
                pixel_size_x = (envelope[2] - envelope[0]) / float(dataset.RasterXSize)
                pixel_size_y = (envelope[3] - envelope[1]) / float(dataset.RasterYSize)
                # print('A: srid={}, envelope={}, PixelSize=({}, {}), NumPixel=({}, {})'.format(srid, envelope, pixel_size_x, pixel_size_y, dataset.RasterXSize, dataset.RasterYSize))
                
                # Clamp Bounds to 1x1km and conforming the EEA Reference GRID.
                x_min = int(envelope[0] / output_pixel_size_x) * output_pixel_size_x
                y_min = int(envelope[1] / output_pixel_size_y) * output_pixel_size_y
                x_max = int(envelope[2] / output_pixel_size_x) * output_pixel_size_x + output_pixel_size_x
                y_max = int(envelope[3] / output_pixel_size_y) * output_pixel_size_y + output_pixel_size_y
                envelope = [x_min, y_min, x_max, y_max]
                # print('B: srid={}, envelope={}, PixelSize=({}, {}), NumPixel=({}, {})'.format(srid, envelope, pixel_size_x, pixel_size_y, dataset.RasterXSize, dataset.RasterYSize))
                clamp_env = True
                
                # Do we need resample?
                if pixel_size_x != output_pixel_size_x or pixel_size_y != output_pixel_size_y or clamp_env:
                    dataset = GdalDataset.transform(dataset, srid, 3035, output_envelope=envelope, output_res_x=output_pixel_size_x, output_res_y=output_pixel_size_y, resample_alg=gdal.GRA_Bilinear, raster_scale=1)
                    temp_ob = GdalDataset(dataset)
                    """
                    srid = temp_ob.get_spatial_srid()
                    envelope = temp_ob.get_extent()
                    pixel_size_x = (envelope[2] - envelope[0]) / float(dataset.RasterXSize)
                    pixel_size_y = (envelope[3] - envelope[1]) / float(dataset.RasterYSize)
                    print('C: srid={}, envelope={}, PixelSize=({}, {}), NumPixel=({}, {})'.format(srid, envelope, pixel_size_x, pixel_size_y, dataset.RasterXSize, dataset.RasterYSize))
                    """
                
                # Create Dataframe_(i) to join to final result.
                temp_pd = raster_to_dataframe(dataset, column_name=field_name, filter_nodata=True, valid_range=[0,10000])
                if len(temp_pd) == 0:
                    print_message(logging.INFO, '> Ignoring Dataset because of it is empty or GDAL could not read its data.', indent=1)
                    continue
                #
                temp_pd[gridnum_attrib] = np.int64(CalcGridFunctions.calcgridnum_np(x=temp_pd['x'] - 0.5*output_pixel_size_x, y=temp_pd['y'] + 0.5*output_pixel_size_y))
                temp_pd.drop(['id','x','y'], axis='columns', inplace=True)
                temp_ob = None
                dataset = None
                
                # Join Datasets using the GridNum attribute.
                if result_pd is None:
                    result_pd = temp_pd
                else:
                    temp_ky = gridnum_attrib + '_'
                    temp_pd = temp_pd.rename({gridnum_attrib: temp_ky}, axis='columns')
                    #display(temp_pd)
                    result_pd = pd.merge(result_pd, temp_pd, left_on=gridnum_attrib, right_on=temp_ky, how='outer')
                    result_pd[gridnum_attrib].fillna(result_pd[temp_ky], inplace=True)                    
                    result_pd[gridnum_attrib] = np.int64(result_pd[gridnum_attrib])
                    #display(result_pd)
                    result_pd.drop([temp_ky], axis='columns', inplace=True)
                    temp_pd = None
                    
                raster_count += 1
                
    # Append some statistics (MEAN, MIX, MAX, STD) accross Columns.
    data_columns = [c for c in result_pd.columns if c not in basic_columns]
    if append_aggr:
        temp_pd = result_pd[data_columns]
        result_pd['mean'] = temp_pd.mean(axis='columns')
        result_pd['min' ] = temp_pd.min (axis='columns')
        result_pd['max' ] = temp_pd.max (axis='columns')
        result_pd['std' ] = temp_pd.std (axis='columns')
        
    # print('Raster Count: {}'.format(raster_count))
    if output_no_data is not None: result_pd = result_pd.fillna(output_no_data)
    result_pd['x'] = CalcGridFunctions.gridid2laea_x_np(result_pd[gridnum_attrib]) + 0.5 * output_pixel_size_x
    result_pd['y'] = CalcGridFunctions.gridid2laea_y_np(result_pd[gridnum_attrib]) - 0.5 * output_pixel_size_y
    result_pd = result_pd[basic_columns + [c for c in result_pd.columns if c not in basic_columns]]
    return result_pd


# COMMAND ----------

# DBTITLE 1,Process ROD674/E1b rasters

import glob
import os
import numpy as np
import re
import tempfile
import uuid

def get_output_file_name(source_file, output_folder):
    """
    Returns the Output file name of the new Aggregate E1b Dataset.
    """
    file_name = os.path.basename(source_file)
    file_name, file_fext = os.path.splitext(file_name)
    
    def get_output_file_name_(output_folder_, pollutant_, year_):
        """
        Returns Output file name.
        """
        return os.path.join(output_folder_, '{}/{}/E1b_{}_avg_{}-XX-XX.parquet'.format(pollutant_, year_, pollutant_, year_))
    
    if file_fext.lower() != '.zip': return None
    year = file_name[0:4]
    if file_name.endswith('_8_P1Y'): return get_output_file_name_(output_folder, 'NO2', year)
    if file_name.endswith('_5_P1Y'): return get_output_file_name_(output_folder, 'PM10', year)
    if file_name.endswith('_6001_P1Y'): return get_output_file_name_(output_folder, 'PM25', year)
    if file_name.endswith('_7_AOT40c'): return get_output_file_name_(output_folder, 'O3', year)
    if file_name.endswith('_5029_P1Y'): return get_output_file_name_(output_folder, 'BaP', year)
    return None

# Process for each input ZIP file...
try:
    source_folder = os.path.join(E1B_SOURCE_FOLDER, '*.zip')
    list_of_years_processed.clear()
    
    gridnum_attrib = 'GridNum1km'
    pixel_size_x = 1000.0
    pixel_size_y = 1000.0
    no_data = None
    raster_no_data = -9999.0
    ignore_pattern_file_name = '_10km_'
    
    # Write raster files using a temporary folder, we'll copy it to output path later.
    with tempfile.TemporaryDirectory() as temp_dir:
        #
        # Converting each input to Dataframe.
        for source_file in glob.glob(source_folder, recursive=False):
            parquet_file = get_output_file_name(source_file, E1B_TARGET_FOLDER)
            if not parquet_file: continue
            
            # year = os.path.basename(source_file)[0:4]
            # if year != '2020': continue
            
            print_message(logging.INFO, '+ Processing "{}" to "{}"...'.format(os.path.basename(source_file), parquet_file), indent=0)
            os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
            
            # Convert to Dataframe (Pandas).
            temp_pd = e1b_to_dataframe(
              file_name=source_file, gridnum_attrib=gridnum_attrib, output_pixel_size_x=pixel_size_x, output_pixel_size_y=pixel_size_y, output_no_data=no_data, ignore_pattern_file_name=ignore_pattern_file_name
            )
            
            # Append current Year as attribute.
            year = os.path.basename(source_file)[0:4]
            temp_pd['Year'] = np.int32(year)
            
            # Output as Parquet file...
            print_message(logging.INFO, ' + Writing "{}"...'.format(parquet_file), indent=1)
            if os.path.exists(parquet_file): os.remove(parquet_file)
            temp_pd.to_parquet(parquet_file, compression='snappy', index=False)
            print_message(logging.INFO, 'Ok!', indent=2)
            
            # Output as Raster file...
            raster_file = os.path.splitext(parquet_file)[0] + '.tiff'
            print_message(logging.INFO, ' + Writing "{}"...'.format(raster_file), indent=1)
            #
            temp_name = str(uuid.uuid1()).replace('-', '')
            temp_file = os.path.join(temp_dir, temp_name) + os.path.splitext(raster_file)[1]
            temp_pd['mean'].fillna(raster_no_data)
            dataframe_to_raster(raster_file=temp_file, dataset=temp_pd, attributes=['mean'], x_attrib='x', y_attrib='y', pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y, no_data=raster_no_data)
            final_file = raster_file[5:]
            dbutils.fs.cp('file:' + temp_file, 'dbfs:' + final_file)
            os.remove(temp_file)
            print_message(logging.INFO, 'Ok!', indent=2)
            
            """
            # Show results.
            temp_df = spark.createDataFrame(temp_pd)
            print('Table Count: {}'.format(temp_df.count()))
            display(temp_df)
            break
            """
            if DELETE_INPUT_FILES: os.remove(source_file)
            year = os.path.basename(source_file)[0:4]
            list_of_years_processed[year] = int(year)
            temp_pd = None
            print_message(logging.INFO, 'Pollutant done!', indent=1)
        #
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'Input': E1B_SOURCE_FOLDER, 'Output': E1B_TARGET_FOLDER})
    
print_message(logging.INFO, 'Aggregates successfully processed!')


# COMMAND ----------


"""
import folium

# Import Folium map utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/maputils.py').read(), 'maputils.py', 'exec'))

# Create Folium Map with my own Layer from a Dataframe.
my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': temp_df, 'attributes': ['mean']})
my_map
"""


# COMMAND ----------

# DBTITLE 1,Joining all ROD674/E1b Datasets in an unique one

import pyspark.sql.functions as F

def parse_date_of_date(date: str):
    """
    Returns the (year, month, day, week) of the specified Date (yyyy-MM-dd).
    """
    y, m, d = date[0:4], date[5:7], date[8:10]
    date_ob = datetime.date(int(y), int(m), int(d))
    w = date_ob.strftime('%W')
    return date, y,m,d, w

def get_output_file_name_2(root_folder: str, pollutant_key: str, operation: str, criteria: str, date: str):
    """
    Returns the Output file name of the new Aggregate ROD674/E1b Dataset.
    """
    date, y,m,d, w = parse_date_of_date(date)
    criteria = criteria.upper()
    
    subfolder = '{}/{}/'.format(pollutant_key, y)
    if criteria in ['MONTH','DAY']: subfolder += (m + '/')
    if criteria in ['WEEK']: subfolder += 'Weeks/'
    if criteria == 'YEAR' : file_name = 'E1b_{}_{}_{}-XX-XX'.format(pollutant_key, operation.lower(), y)
    if criteria == 'MONTH': file_name = 'E1b_{}_{}_{}-{}-XX'.format(pollutant_key, operation.lower(), y, m)
    if criteria == 'WEEK' : file_name = 'E1b_{}_{}_{}-wk-{}'.format(pollutant_key, operation.lower(), y, w)
    if criteria == 'DAY'  : file_name = 'E1b_{}_{}_{}-{}-{}'.format(pollutant_key, operation.lower(), y, m, d)    
    return '{}/{}{}.parquet'.format(root_folder, subfolder, file_name)

# Process for each Pollutant and Year...
try:
    print_message(logging.INFO, 'Starting JOIN of all Pollutants...', indent=0)
    AGGR_TARGET_FOLDER = E1B_TARGET_FOLDER
    
    #list_of_years_processed.clear()
    #list_of_years_processed['2020'] = 2020
    
    for year in list_of_years_processed.keys():
        null_columns = []
        dataset = None        
        
        downloading_date = str(year) + '-01-01'
        print_message(logging.INFO, ' + Processing {}:'.format(year))
        
        # ... join current Dataset with previous one.
        for key in pollutants:
            pollutant = pollutants[key]
            print_message(logging.INFO, ' > Processing Table of Pollutant "{}" ({})...'.format(key, downloading_date[0:4]))
            
            file_name = get_output_file_name_2(AGGR_TARGET_FOLDER, key, 'avg', 'year', downloading_date)
            file_name = file_name[5:]
            
            if not os.path.exists('/dbfs' + file_name):
                null_columns.append('e1b_' + key)
                print_message(logging.INFO, ' > The file "{}" not exist!'.format(file_name))
                continue
                
            temp_ob = spark.read.parquet(file_name)
            temp_ob = temp_ob.select(['GridNum1km', 'x', 'y', 'mean', 'Year'])
            temo_ob = temp_ob.alias(key)
            
            if dataset is None:
                dataset = temp_ob
                dataset = dataset.withColumnRenamed('mean', 'e1b_' + key)
            else:
                temp_ob = temp_ob.withColumnRenamed('mean', 'e1b_' + key)
                temp_ob = temp_ob.withColumnRenamed('GridNum1km', 'GridNum1km_')
                temp_ob = temp_ob.withColumnRenamed('x', 'x_')
                temp_ob = temp_ob.withColumnRenamed('y', 'y_')
                temp_ob = temp_ob.withColumnRenamed('Year', 'Year_')
                dataset = dataset.join(temp_ob, dataset['GridNum1km']==temp_ob['GridNum1km_'], how='outer')
                dataset = dataset.withColumn('GridNum1km', F.coalesce('GridNum1km', 'GridNum1km_'))
                dataset = dataset.withColumn('x', F.coalesce('x', 'x_'))
                dataset = dataset.withColumn('y', F.coalesce('y', 'y_'))
                dataset = dataset.withColumn('Year', F.coalesce('Year', 'Year_'))
                dataset = dataset.drop('GridNum1km_')
                dataset = dataset.drop('x_')
                dataset = dataset.drop('y_')
                dataset = dataset.drop('Year_')
                temp_ob = None
                
        for c in null_columns:
            dataset = dataset.withColumn(c, F.lit(None).cast('double'))

        # Sort Columns.
        columns = [c for c in dataset.columns if c.startswith('e1b_')]
        dataset = dataset.select(['x', 'y', 'GridNum1km', 'Year'] + columns)
        
        # Write Join Dataset (Using Pandas, we prefer do not partitionate output files).
        file_name = '/dbfs' + get_output_file_name_2(AGGR_TARGET_FOLDER, 'ALL', 'avg', 'year', downloading_date)
        file_name = file_name[5:]
        file_dir = os.path.dirname(file_name)
        os.makedirs(file_dir, exist_ok=True)
        
        print_message(logging.INFO, ' > Writing JOIN Dataset "{}"...'.format(file_name))
        if os.path.exists(file_name): os.remove(file_name)
        temp_pd = dataset.toPandas()
        temp_pd.to_parquet(file_name, compression='snappy', index=False)
        
        # display(dataset)
        # dataset2 = spark.read.parquet(file_name[5:])
        # display(dataset2)
        temp_pd = None
        print_message(logging.INFO, ' > Year done!')
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date})
    
print_message(logging.INFO, 'Join successfully processed!')


# COMMAND ----------


"""
temp_df = spark.read.parquet('/mnt/airquality-e1b/Ensemble/ALL/2019/E1b_ALL_avg_2019-XX-XX.parquet')
print('Table Count: {}'.format(temp_df.count()))
display(temp_df)

import folium

# Import Folium map utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/maputils.py').read(), 'maputils.py', 'exec'))

# Create Folium Map with my own Layer from a Dataframe.
my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': temp_df, 'attributes': ['e1b_NO2']})
my_map
"""


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'Input': E1B_SOURCE_FOLDER, 'Output': E1B_TARGET_FOLDER})    
notebook_mgr = None

