# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets
# MAGIC

# COMMAND ----------

# dbutils.widgets.removeAll()


# COMMAND ----------


# Set default parameters for input widgets
DEFAULT_START_DATE = '01/01/2019'
DEFAULT_END_DATE = '23/03/2023'
DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'SOMO', 'SOMO35', 'NO2', 'SO2']
DEFAULT_STORE_PREDICTIONS_LIST = ['YES', 'NO']

# Set widgets for notebook
dbutils.widgets.text(name='StartDate', defaultValue=str(DEFAULT_START_DATE), label='Start Date')
dbutils.widgets.text(name='EndDate', defaultValue=str(DEFAULT_END_DATE), label='End Date')
dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.dropdown('StoreParquet', 'NO', DEFAULT_STORE_PREDICTIONS_LIST, label='Store Parquet')


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables
# MAGIC

# COMMAND ----------

import logging
import datetime

import numpy as np
import pandas as pd

from osgeo import gdal, osr, ogr
from pyspark.sql.functions import abs, col, floor, when, lit

# Enable Arrow-based columnar data transfers + limiting size of partitions so we can prevent memory issues
spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
# spark.conf.set("spark.sql.files.maxPartitionBytes", "128m")
# spark.conf.set("spark.sql.files.openCostInBytes", "128m")

# Import 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+'/mnt/dis2datalake_airquality-cams', logging_mode='w')

# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)


# COMMAND ----------

# Adding input variables from widgets
start_date:str = datetime.datetime.strptime(dbutils.widgets.get('StartDate'), '%d/%m/%Y')
end_date:str = datetime.datetime.strptime(dbutils.widgets.get('EndDate'), '%d/%m/%Y') 
pollutants:list = dbutils.widgets.get('Pollutants').split(',')
store_parquet:bool = True if dbutils.widgets.get('StoreParquet') == 'YES' else False

logging.info(f'Your chosen parameters: start_date: "{start_date}", end_date: "{end_date}", pollutants: {pollutants}, store_parquet: "{store_parquet}"')

if (end_date < start_date): raise Exception('¡¡¡ WARNING !!!! End dates cannot be earlier than starting dates. Double check!') 


# COMMAND ----------

# MAGIC %run "./Lib"
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Functions
# MAGIC
# MAGIC <br/>

# COMMAND ----------

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

    output_dataset = gdal.Warp(raster_file, dataset, options=options)

    return output_dataset


# def add_xy_location(df:pd.DataFrame, raster_x_size:int, transform:tuple):
#     """Adds the X and Y locations to the input dataframe `df` based on the given `raster_x_size` and `transform`.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         The input dataframe to which X and Y locations need to be added.
#     raster_x_size : int
#         The size of the raster in the X direction.
#     transform : tuple
#         A tuple containing the affine transformation matrix. The tuple contains 6 elements representing the
#         coefficients a, b, c, d, e, and f of the 2D affine transformation matrix.
    
#     Returns:
#     --------
#     df_spark : pyspark.sql.DataFrame
#         A new dataframe with columns for X and Y locations added to the input dataframe `df`.
#     """

#     df_spark = spark.createDataFrame(df)
#     raster_df = None

#     # Append XY location.
#     df_spark = df_spark.withColumn('r', (col('id') / raster_x_size).cast('integer'))
#     df_spark = df_spark.withColumn('c', ((col('id') - (col('r') * raster_x_size)).cast('integer')))
#     df_spark = df_spark.withColumn('x', ((col('c') * transform[1]) + (transform[0] + (0.5 * transform[1]))))
#     df_spark = df_spark.withColumn('y', ((col('r') * transform[5]) + (transform[3] + (0.5 * transform[5]))))
#     df_spark = df_spark.drop('r', 'c')

#     return df_spark

from pyspark.sql.types import DoubleType
# from pyspark.sql.functions import col, floor

def add_xy_location(df: pd.DataFrame, raster_x_size: int, transform: tuple):
    """Adds the X and Y locations to the input dataframe `df` based on the given `raster_x_size` and `transform`.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe to which X and Y locations need to be added.
    raster_x_size : int
        The size of the raster in the X direction.
    transform : tuple
        A tuple containing the affine transformation matrix. The tuple contains 6 elements representing the
        coefficients a, b, c, d, e, and f of the 2D affine transformation matrix.
    
    Returns:
    --------
    pyspark.sql.DataFrame
        A new dataframe with columns for X and Y locations added to the input dataframe `df`.
    """
    # Convert pandas dataframe to pyspark dataframe
    df_spark = spark.createDataFrame(df)

    # Change data type of 'id' column to DoubleType
    df_spark = df_spark.withColumn("id", col("id").cast(DoubleType()))

    # Append XY location.
    df_spark = df_spark.withColumn('r', (col('id') / raster_x_size).cast('integer'))
    df_spark = df_spark.withColumn('c', ((col('id') - (col('r') * raster_x_size)).cast('integer')))
    df_spark = df_spark.withColumn('x', ((col('c') * transform[1]) + (transform[0] + (0.5 * transform[1]))))
    df_spark = df_spark.withColumn('y', ((col('r') * transform[5]) + (transform[3] + (0.5 * transform[5]))))
    df_spark = df_spark.drop('r', 'c')

    return df_spark


# COMMAND ----------

def raster_to_dataframe(dataset, column_name:str='cams', valid_range=None):
    """Convert a raster dataset to a Pandas DataFrame.

    Parameters
    ----------
    dataset : osgeo.gdal.Dataset
        A GDAL dataset object representing the raster.
    column_name : str, default = 'cams'
        The prefix for the column names in the output DataFrame.
    valid_range : tuple, default = None
        A tuple representing the minimum and maximum valid values for the raster

    Returns
    -------
    pd.DataFrame
        raster_df: pd.DataFrame = DataFrame containing the raster values.
        raster_x_size_cols: int = The 'x' shape of the raster (number of cols)
        transform: tuple = The GeoTransform parameters of the raster.
    """

    transform  = dataset.GetGeoTransform()
    raster_x_size_cols, raster_y_size_rows = dataset.RasterXSize, dataset.RasterYSize
    no_data    = dataset.GetRasterBand(1).GetNoDataValue()
    raster     = dataset.ReadAsArray(0, 0, raster_x_size_cols, raster_y_size_rows)

    # Apply valid range of Values.
    if valid_range:
        min_valid_value, max_valid_value = valid_range
        raster = np.where((raster >= min_valid_value) & (raster <= max_valid_value), raster, no_data)

    # Building dataframe from arrays
    rows = []
    for num, arr in enumerate(raster):
        col_name = f'{column_name}_{num}' if dataset.RasterCount >= 1 else column_name
        valid_indices = np.where(arr != no_data)
        values = arr[valid_indices]
        ids = np.ravel_multi_index(valid_indices, arr.shape)
        rows.append(pd.DataFrame({col_name: values}, index=ids))

    raster_df = pd.concat(rows, axis=1)
    raster_df.index.name = 'id'
    raster_df = raster_df.reset_index()
    
    return raster_df, raster_x_size_cols, transform

# COMMAND ----------

def process_raster_dataset(storage_path = '', raster_dataset=None, resample_alg=gdal.GRA_Bilinear, driver_name='MEM'):

    logging.info('Warping dataset to eea grid...')
    output_array = warp_dataset_to_eeagrid(storage_path, raster_dataset, resample_alg=resample_alg, driver_name=driver_name)

    logging.info('Converting raster to dataframe...')
    raster_df, raster_x_size_cols, transform = raster_to_dataframe(output_array)

    logging.info('Adding x,y locations...')
    raster_df_ps  = add_xy_location(raster_df, raster_x_size_cols, transform)

    return raster_df_ps


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Execute hourly data preprocessing for CAMS pollutants
# MAGIC
# MAGIC **Note column "cams_24" represents the daily average**

# COMMAND ----------

from pyspark import StorageLevel


# dates_list = pd.date_range(start_date, end_date).tolist()
missing_dates = []
try:
  for pollutant in pollutants:
    preprocess_data_handler = PreProcessDataHandler(pollutant)
    logging.info('Processing Pollutant: {}'.format(pollutant))

    dates = (date for date in pd.date_range(start_date, end_date))
    for date in dates:
      raster_dataset, input_path, output_path = preprocess_data_handler.data_collector(date)
      logging.info('Processing file: {}'.format(input_path))

      if raster_dataset is not None:
        raster_df_ps = process_raster_dataset(storage_path = '', raster_dataset=raster_dataset, resample_alg=gdal.GRA_Bilinear, driver_name='MEM')

        logging.info('Calculating gridnums...')
        raster_df_ps = raster_df_ps.withColumn("GridNum1km", CalcGridFunctions.calcgridnum_ps(col("x"), col("y"))).drop('id', 'x', 'y')#.persist()#.persist(storageLevel=StorageLevel.MEMORY_ONLY)
        
        logging.info('Hourly pollutant processed for {}!'.format(date))
        
        if store_parquet:
          logging.info('Storing data into parquet at: {}'.format(preprocess_data_handler.data_handler.file_system_path + output_path))
          raster_df_ps = raster_df_ps.coalesce(3).write.parquet(preprocess_data_handler.data_handler.file_system_path + output_path)
          
      else:
        missing_dates.append(date)
        logging.info('Tiff file missing for: ', str(date))
        continue
      # raster_df_ps = raster_df_ps.unpersist()
      # raster_df_ps = None

  logging.info('Pollutant Done!\n\n')
  logging.info('Missing dates: %s', missing_dates) if len(missing_dates)>0 else logging.info('No missing dates!')

except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'start_date': str(start_date), 'end_date': str(end_date), 'pollutants': str(pollutants), 'store_parquet': str(store_parquet), 'current_date': str(date), 'missing_dates': str(missing_dates)})


# COMMAND ----------

# MISSING  2022-07-12, 

# COMMAND ----------

# 1128-120227-d36bde4    # cluster ID dapi704d7dfde3b81ea411e7f4ba71d53b1b

# COMMAND ----------



# COMMAND ----------

# 1128-120227-d36bde4 clusterID

# COMMAND ----------

# dates_list = pd.date_range(start_date, end_date).tolist()

# try:
#   for pollutant in pollutants:
#     preprocess_data_handler = PreProcessDataHandler(pollutant)
#     logging.info('Processing Pollutant: {}'.format(pollutant))

#     for date in dates_list:
#       raster_dataset, input_path, output_path = preprocess_data_handler.data_collector(date)
#       logging.info('Processing file: {}'.format(input_path))

#       logging.info('Warping dataset to eea grid...')
#       output_array = warp_dataset_to_eeagrid('', raster_dataset, resample_alg=gdal.GRA_Bilinear, driver_name='MEM')
#       raster_dataset = None

#       logging.info('Converting raster to dataframe...')
#       raster_df, raster_x_size_cols, transform = raster_to_dataframe(output_array)
#       output_array = None
  
#       logging.info('Adding x,y locations...')
#       raster_df_ps  = add_xy_location(raster_df, raster_x_size_cols, transform)
#       raster_df = None

#       logging.info('Calculating gridnums...')
#       raster_df_ps = raster_df_ps.withColumn("GridNum1km", CalcGridFunctions.calcgridnum_ps(col("x"), col("y"))).drop('id', 'x', 'y').cache()
#       # raster_df_ps = raster_df_ps.drop('id', 'x', 'y')
#       # raster_df_ps = raster_df_ps.cache()
#       logging.info('Hourly pollutant processed for {}!'.format(date))
      
#       if store_parquet:
#         logging.info('Storing data into parquet at: {}'.format(preprocess_data_handler.data_handler.file_system_path + output_path))
#         raster_df_ps = raster_df_ps.coalesce(3)
#         raster_df_ps.write.parquet(preprocess_data_handler.data_handler.file_system_path + output_path)

#       raster_df_ps = raster_df_ps.unpersist()
#       raster_df_ps = None

#   logging.info('Pollutant Done!\n\n')

# except Exception as e:
#     message = '{}\n{}'.format(str(e), traceback.format_exc())
#     notebook_mgr.exit(status='ERROR', message=message, options={'start_date': str(start_date), 'end_date': str(end_date), 'pollutants': str(pollutants), 'store_parquet': str(store_parquet)})


# COMMAND ----------

# # Configure the Parquet file format options
# parquet_options = {
#   "parquet.row.group.size": str(500000)
# }

# # Write the DataFrame to disk using the configured options
# raster_df_ps.write.format("parquet").mode("overwrite").options(**parquet_options).save(preprocess_data_handler.data_handler.file_system_path + output_path)


# COMMAND ----------

# batch_size = 10000
# total_rows = raster_df_ps.count()
# num_batches = (total_rows // batch_size) + 1

# for i in range(num_batches):
#     start_index = i * batch_size
#     end_index = (i + 1) * batch_size
    
#     current_batch = raster_df_ps[start_index:end_index]
#     current_batch.write.parquet(preprocess_data_handler.data_handler.file_system_path + output_path + f"/batch_{i}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Finishing Job

# COMMAND ----------

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'start_date': str(start_date), 'end_date': str(end_date), 'pollutants': str(pollutants), 'store_parquet': str(store_parquet)})
notebook_mgr = None


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Check values match after our x, y calculations

# COMMAND ----------

# On QGIS selected value to find on band_2 (our cams_1 since they use band_24 for 00:00:00 and while we use cams_0) = 8.791517

for i in range(24):
  try:
    display(raster_df[round(raster_df[f'cams_{i}'], 6)==8.791517])
    print(f'cams_{i}')

  except:
    pass

# COMMAND ----------

# On QGIS selected value to find on band_3 (our cams_2) = 9.038125

for i in range(24):
  try:
    display(raster_df[round(raster_df[f'cams_{i}'], 6)==9.038125])
    print(f'cams_{i}')

  except:
    pass
  
  
# We see id 219975 match on both cells. Then we check our values for x and y and compare them with QGIS coordinates (-7.48º, 38.54º) vs our dataset x, y (-7.450000, 38.549999)

# COMMAND ----------

  
# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# COMMAND ----------

my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': raster_df[[col for col in raster_df.columns if col != 'id']], 'attributes': [f'cams_1']})
display(my_map)


# COMMAND ----------



