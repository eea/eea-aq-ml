# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# dbutils.widgets.removeAll()


# COMMAND ----------


# Set default parameters for input widgets
DEFAULT_TRAIN_START = '2016'
DEFAULT_TRAIN_END = '2019'
DEFAULT_PREDVAL_START = '2020'
DEFAULT_PREDVAL_END = '2020'
DEFAULT_VERSION = 'v0'
DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b']
DEFAULT_STORE_PREDICTIONS_LIST = ['YES', 'NO']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                       # We need this to load the pretrained model
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')                             # We need this to load the pretrained model
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                          
dbutils.widgets.dropdown('StorePredictions', 'NO', DEFAULT_STORE_PREDICTIONS_LIST, label='Store Predictions')  


# https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html
# https://docs.databricks.com/_extras/notebooks/source/xgboost-pyspark.html


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

# MAGIC %run "../utils/Lib1"

# COMMAND ----------

# MAGIC %run "../config/ConfigFile"

# COMMAND ----------

import logging
from pyspark.sql.types import LongType
import pyspark.sql.functions as F


from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
# gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')                            # Es posible que esto sea lo que no nos permitia guardar en memoria en el hourly  ?????????????
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')


# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

gridid2laea_x_udf = spark.udf.register('gridid2laea_x', CalcGridFunctions.gridid2laea_x, LongType())
gridid2laea_y_udf = spark.udf.register('gridid2laea_y', CalcGridFunctions.gridid2laea_y, LongType())


# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)



# Adding input variables from widgets
train_start_year:str = dbutils.widgets.get('TrainStartDate')
train_end_year:str = dbutils.widgets.get('TrainEndDate')
predval_start_year:str = dbutils.widgets.get('PredValStartDate')
predval_end_year:str = dbutils.widgets.get('PredValEndDate')
pollutants:list = dbutils.widgets.get('Pollutants').split(',')
trainset:list = dbutils.widgets.get('Trainset').split(',')
date_of_input:str = dbutils.widgets.get('DateOfInput')
version:str = dbutils.widgets.get('Version')
features:list = ['selected'] 
store_predictions:bool = True if dbutils.widgets.get('StorePredictions') == 'YES' else False


logging.info(f'Your chosen parameters to PREDICT: predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", store_predictions:"{store_predictions}"')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if predval_end_year < predval_start_year: raise Exception('End dates cannot be earlier than starting dates. Double check!') 


# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Functions
# MAGIC 
# MAGIC <br/>

# COMMAND ----------

  
def write_dataset_to_raster(output_raster_path, dataset, attribute, x_attrib='x', y_attrib='y',
                            driver_name='Gtiff', srid=3035, bbox=None, pixel_size_x=1000.0, pixel_size_y=1000.0, no_data=-9999,
                            options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']):
    """
    Write to raster file the specified Dataset object and parameters.
    """
    is_a_dataframe = hasattr(dataset, 'select')
    
    import tempfile
    import uuid
    import os
    import numpy as np
    
    temp_name = str(uuid.uuid1()).replace('-', '')
    columns = dataset.columns
    for c in [x_attrib, y_attrib, attribute]:
        if c not in columns: raise Exception('The Dataset does not contain the "{}" attribute.'.format(c))
              
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
    SPK_DataTypeCodes = dict([
        ('unknown', gdal.GDT_Unknown),
        ('byte', gdal.GDT_Byte),
        ('sort', gdal.GDT_Int16),
        ('int', gdal.GDT_Int32),
        ('bigint', gdal.GDT_Float64),
        ('long', gdal.GDT_Float64),
        ('float', gdal.GDT_Float32),
        ('double', gdal.GDT_Float64),
    ])
    if is_a_dataframe:
        data_type = str(dataset.select(attribute).dtypes[0][1])
        data_format = SPK_DataTypeCodes[data_type]
    else:
        data_type = str(dataset[attribute].dtypes)
        data_format = GDT_DataTypeCodes[data_type]
            
    # Calculate current CRS.
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(srid)
    if hasattr(spatial_ref, 'SetAxisMappingStrategy'): spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    # Calculate current BBOX when not specified, taking care of input Dataset object type.
    if bbox is None and is_a_dataframe:
        dataset.createOrReplaceTempView(temp_name)
        envelope_df = spark.sql('SELECT MIN({}) as x_min, MIN({}) as y_min, MAX({}) as x_max, MAX({}) as y_max FROM {}'.format(x_attrib, y_attrib, x_attrib, y_attrib, temp_name))
        temp_df = envelope_df.collect()[0]
        x_min, y_min, x_max, y_max = temp_df[0], temp_df[1], temp_df[2], temp_df[3]
        spark.catalog.dropTempView(temp_name)
        bbox = [x_min, y_min, x_max, y_max]
    if bbox is None and not is_a_dataframe:
        temp_df = dataset[[x_attrib, y_attrib]].min(axis=0)
        x_min, y_min = temp_df['x'], temp_df['y']
        temp_df = dataset[[x_attrib, y_attrib]].max(axis=0)
        x_max, y_max = temp_df['x'], temp_df['y']
        bbox = [x_min, y_min, x_max, y_max]
        
    n_cols = 1 + ((bbox[2] - bbox[0]) / pixel_size_x)
    n_rows = 1 + ((bbox[3] - bbox[1]) / pixel_size_y)
    n_cols = int(n_cols)
    n_rows = int(n_rows)
    
    # Append INDEX for each cell, for matching the INDEX/VALUE pairs when filling the target np.array.
    if is_a_dataframe:
        import pyspark.sql.functions as F
        from pyspark.sql.types import LongType
        
        dataset = dataset \
            .withColumn('idx_', (((F.lit(bbox[3]) - F.col(y_attrib)) / F.lit(pixel_size_y)) * F.lit(n_cols)) + ((F.col(x_attrib) - F.lit(bbox[0])) / F.lit(pixel_size_x))) \
            .withColumn('idx_', F.col('idx_').cast(LongType()))
    else:
        dataset['idx_'] = \
            (((bbox[3] - dataset[y_attrib]) / pixel_size_y) * n_cols) + ((dataset[x_attrib] - bbox[0]) / pixel_size_x)
        
    # Write raster file using a temporary folder, we'll copy it to output path later.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, temp_name) + os.path.splitext(output_raster_path)[1]
        
        driver = gdal.GetDriverByName(driver_name)
        raster = driver.Create(temp_file, n_cols, n_rows, 1, data_format, options=options)
        raster.SetGeoTransform([bbox[0] - 0.5*pixel_size_x, pixel_size_x, 0.0, bbox[3] + 0.5*pixel_size_y, 0.0, -pixel_size_y])
        raster.SetProjection(spatial_ref.ExportToWkt())
        raster_band = raster.GetRasterBand(1)
        raster_band.SetNoDataValue(no_data)
        
        # Write values (Using the 'INDEX' attribute as row/col locator of Cells).
        if is_a_dataframe:
            temp_np = dataset.select(['idx_', attribute]).toPandas()
            indx_np = temp_np['idx_']
            data_np = temp_np[attribute]
            
            r_array = np.full((n_rows * n_cols), no_data, dtype=data_type)
            np.put(r_array, indx_np, data_np)
            raster_band.WriteArray(r_array.reshape(n_rows, n_cols))
            del r_array
        else:
            indx_np = dataset['idx_']
            data_np = dataset[attribute]
            
            r_array = np.full((n_rows * n_cols), no_data, dtype=data_type)
            np.put(r_array, indx_np, data_np)
            raster_band.WriteArray(r_array.reshape(n_rows, n_cols))
            del r_array
            
        raster_band = None
        raster.FlushCache()
        raster = None
        
        # Copy the temporary raster file to output path.
        if output_raster_path.startswith('/dbfs'):
            final_file = output_raster_path[5:]
            dbutils.fs.cp('file:' + temp_file, 'dbfs:' + ml_data_handler.data_handler.file_system_path + final_file)
        else:
            import shutil
            shutil.copy2(temp_file, output_raster_path)
            
    return output_raster_path

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Execute predict

# COMMAND ----------

for pollutant in pollutants:   
  
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]

    ml_worker = MLWorker(pollutant)
    ml_data_handler = MLDataHandler(pollutant)

    # Prediction inputs data                                                                   ????? shall we also concatenate preds dataset into the final model training????
    pollutant_prediction_data, output_parquet_path, raster_output_path = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, None, None, features)
    logging.info('Data pollutant collected!')
    pollutant_prediction_data_pd = pollutant_prediction_data.toPandas()
    
    # Loading pretrained model and executing predictions
    model_to_train_details = {'model_name': f"{pollutant}_{ml_worker.ml_models_config.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}"}
    trained_model = ml_worker.train_load_ml_model(model_name=model_to_train_details['model_name'], X_train_data=None, Y_train_data=None)
    logging.info(f'Performing predictions with features:\n {pollutant_prediction_data_pd.count()}')
    predictions = trained_model.predict(pollutant_prediction_data_pd)
    
    # Joining predictions with their gridnum and year
    predictions_df = pd.DataFrame(predictions, columns=[pollutant.upper()])
    ml_outputs = pd.concat([pollutant_prediction_data_pd[['GridNum1km', 'Year']], predictions_df], axis=1)

    # Dealing with memory issues
    predictions_df = None
    pollutant_prediction_data_pd = None
    predictions = None

    if store_predictions:
      logging.info('Writing parquet file into {} '.format(output_parquet_path))
      ml_data_handler.data_handler.parquet_storer(ml_outputs, output_parquet_path)
      df_spark = spark.createDataFrame(ml_outputs)
      
      # Adding XY location using 'GridNum1km' attribute (For didactical purpose).
      ml_outputs_df_xy = df_spark \
                                    .withColumnRenamed('x', 'x_old') \
                                    .withColumnRenamed('y', 'y_old') \
                                    .withColumn('x', gridid2laea_x_udf('GridNum1km') + F.lit(500)) \
                                    .withColumn('y', gridid2laea_y_udf('GridNum1km') - F.lit(500))
      ml_outputs = None
      df_spark = None

      ml_outputs_df_xy = ml_outputs_df_xy.cache()

      # Write to geotiff       
      logging.info('Writing geotiff file into {} '.format(raster_output_path))
      write_dataset_to_raster(output_raster_path='/dbfs'+ raster_output_path, dataset=ml_outputs_df_xy, attribute=pollutant, pixel_size_x=1000.0, pixel_size_y=1000.0)

      ml_outputs_df_xy.unpersist()
      
logging.info(f'Finished predictions!')


# Note if we add some more features/rows to the df, we might need to use SPARK xgboost regressor since pandas cannot support it. If we add it now, we might be using spark for few data (unneficient)


# COMMAND ----------

# Plot predictions
my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': ml_outputs_df_xy, 'attributes': [pollutant]})
display(my_map)

# COMMAND ----------


