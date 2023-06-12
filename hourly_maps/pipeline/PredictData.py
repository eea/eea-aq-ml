# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets
# MAGIC

# COMMAND ----------

# dbutils.widgets.removeAll()


# COMMAND ----------


"""
================================================================================
Notebook to execute predictions of the pollutants. We should only need to modify the widgets for normal executions.
We will obtain a parquet + tiff files containing the forecasted values inside the selected paths at our config.py file.


Arguments:
  + date_of_input: date used to build the path where we are storing our input data
  + pollutants: list of pollutants we are willing to forecast 
  + predval_end_year: last date for the interval we are willing to forecast
  + predval_start_year: starting date for the period we are willing to forecast
  + store_predictions: bool to determine if we want to store our predictions or not 
  + train_end_year: last date we used to train the model (used for the naming of the ML model stored at Azure Experiments)
  + train_start_year: first date we used to train the model (used for the naming of the ML model stored at Azure Experiments)
  + trainset: list of the targets we are willing to predict 
  + version: version of the model (used for naming) 

================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157021
Author   : aiborra-ext@tracasa.es

================================================================================
"""

# Set default parameters for input widgets
# DEFAULT_TRAIN_START = '2023/05/22'
# DEFAULT_TRAIN_END = '2023/05/28'
DEFAULT_PREDVAL_START = '2023-05-22'
DEFAULT_PREDVAL_END =  '2023-05-28'
# DEFAULT_VERSION = 'v0'
# DEFAULT_DATE_OF_INPUT = '20230201'
DEFAULT_FEATURES_LIST = ['*', 'selected']

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b']
DEFAULT_STORE_PREDICTIONS_LIST = ['YES', 'NO']

# Set widgets for notebook
# dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                       # We need this to load the pretrained model
# dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')                             # We need this to load the pretrained model
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
# dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
# dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                          
dbutils.widgets.dropdown('StorePredictions', 'NO', DEFAULT_STORE_PREDICTIONS_LIST, label='Store Predictions')  


# https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html
# https://docs.databricks.com/_extras/notebooks/source/xgboost-pyspark.html


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables
# MAGIC

# COMMAND ----------

# MAGIC %run "../utils/Lib"
# MAGIC

# COMMAND ----------

import logging
import pyspark.sql.functions as F

from pyspark.sql.types import LongType
from osgeo import gdal
from osgeo import osr
from pyspark.sql.functions import dayofweek

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
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


# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+'/mnt/dis2datalake_airquality-predictions', logging_mode='w')

# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)


# Adding input variables from widgets
# train_start_year:str = dbutils.widgets.get('TrainStartDate')
# train_end_year:str = dbutils.widgets.get('TrainEndDate')
predval_start_date:str = dbutils.widgets.get('PredValStartDate')
predval_end_date:str = dbutils.widgets.get('PredValEndDate')
pollutants:list = dbutils.widgets.get('Pollutants').split(',')
trainset:list = dbutils.widgets.get('Trainset').split(',')
# date_of_input:str = dbutils.widgets.get('DateOfInput')
# version:str = dbutils.widgets.get('Version')
store_predictions:bool = True if dbutils.widgets.get('StorePredictions') == 'YES' else False

features:list = dbutils.widgets.get('Features') if isinstance(dbutils.widgets.get('Features'), list) else [dbutils.widgets.get('Features')]


logging.info(f'Your chosen parameters to PREDICT: train_start_year: predval_start_year: "{predval_start_date}", predval_end_year: "{predval_end_date}", pollutants: {pollutants}, trainset: {trainset}, store_predictions:"{store_predictions}"')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if predval_end_date < predval_start_date: raise Exception('End dates cannot be earlier than starting dates. Double check!') 


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

try:
  for pollutant in pollutants:     
    # In case we have different target variables i.e.: eRep and e1b.
    for target in trainset:
      logging.info(f'Processing pollutant: {pollutant} target {target}.')
      label = [pollutant.upper()][0]

      ml_worker = MLWorker(pollutant)
      ml_data_handler = MLDataHandler(pollutant)
      data_handler = DataHandler(pollutant)

      # Prediction inputs data                                                                   ????? shall we also concatenate preds dataset into the final model training????
      # pollutant_prediction_data, output_parquet_path, raster_output_path = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, None, None, features=['selected'])
      selected_cols_pollutants = data_handler.config.select_cols(data_handler.pollutant) if features[0]=='selected' else ['*'] 
      # prediction_path_struct:str = '/ML_Input/HOURLY_DATA/data-{}_{}-{}/{}_{}/prediction_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year



      # pollutant_prediction_data = data_handler.parquet_reader(f'/ML_Input/HOURLY_DATA/data-{pollutant}_{predval_start_date}_{predval_end_date}.parquet', features=selected_cols_pollutants)#.dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])    # MODIFICAR PATH PARA INPUT DATA DEL PREDICTION
      pollutant_prediction_data = data_handler.parquet_reader(f'/ML_Input/HOURLY_DATA/data-{pollutant}_2023-05-22_{predval_end_date}.parquet', features=selected_cols_pollutants)#.dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])    # MODIFICAR PATH PARA INPUT DATA DEL PREDICTION

      pollutant_prediction_data = pollutant_prediction_data.select([col for col in pollutant_prediction_data.columns if col not in [pol for pol in DEFAULT_POLLUTANTS_LIST]])
      pollutant_prediction_data = pollutant_prediction_data.withColumn('weekday', dayofweek('date'))

      dates = (date for date in pd.date_range(predval_start_date, predval_end_date))
      for dat in dates:
        print(dat)
        print( str(dat).split(' ')[0])
        if str(str(dat).split(' ')[0]) == '2023-05-24':
          start_hour = 15
        else:
          start_hour = 0
        for hour in range(start_hour,24):
          pollutant_pred_dated = pollutant_prediction_data.filter((pollutant_prediction_data.date==dat) & (pollutant_prediction_data.hour == hour))

          logging.info('Data pollutant collected for {} date {} hour {}!'.format(pollutant, dat, hour))
          pollutant_prediction_data_pd = pollutant_pred_dated.toPandas()
          pollutant_prediction_input_data_pd = pollutant_pred_dated.drop('GridNum1km', 'Year','AreaHa', 'level3_code', 'adm_country', 'datetime_end', 'datetime_begin', 'date').toPandas().rename(columns={f'cams_{pollutant}': f'CAMS_{pollutant}'})

          # Loading pretrained model and executing predictions
          model_to_train_details = {'model_name': f"{pollutant}_{ml_worker.ml_models_config.model_str.replace('()', '')}_hourly"}
          trained_model = ml_worker.train_load_ml_model(model_name=model_to_train_details['model_name'], X_train_data=None, Y_train_data=None)


          logging.info(f'Performing predictions with features:\n {pollutant_prediction_input_data_pd.count()}')
          predictions = trained_model.predict(pollutant_prediction_input_data_pd)
    
          # Joining predictions with their gridnum and year
          predictions_df = pd.DataFrame(predictions, columns=[pollutant.upper()])
          ml_outputs = pd.concat([pollutant_prediction_data_pd[['GridNum1km', 'date', 'hour']], predictions_df], axis=1)

          # Dealing with memory issues
          predictions_df = None
          pollutant_prediction_data_pd = None
          pollutant_prediction_input_data_pd = None
          predictions = None

          if store_predictions:
            # output_parquet_path_struct:str = f'/ML_Output/HOURLY_OUTPUTS/{pollutant}/{predval_start_date}_{predval_end_date}/{str(dat).split(" ")[0]}/hour_{hour}_maps_HOURLY_TEST'                                   # pollutant, predval_start_year, predval_end_year, date_of_input
            output_parquet_path_struct:str = f'/ML_Output/HOURLY_OUTPUTS/{pollutant}/2023-05-22_{predval_end_date}/{str(dat).split(" ")[0]}/hour_{hour}_maps_HOURLY_TEST'                                   # pollutant, predval_start_year, predval_end_year, date_of_input
            # raster_outputs_path_struct:str = f'/ML_Output/HOURLY_OUTPUTS/GeoTiffs/{pollutant}/{predval_start_date}_{predval_end_date}/{str(dat).split(" ")[0]}/hour_{hour}_1km_Europe_EEA_ML_XGB_HOURLY_TEST.tiff'  # predyear, code, agg, predyear, code, agg, ml_models_config.model_str[:2]
            raster_outputs_path_struct:str = f'/ML_Output/HOURLY_OUTPUTS/GeoTiffs/{pollutant}/2023-05-22_{predval_end_date}/{str(dat).split(" ")[0]}/hour_{hour}_1km_Europe_EEA_ML_XGB_HOURLY_TEST.tiff'  # predyear, code, agg, predyear, code, agg, ml_models_config.model_str[:2]


            logging.info('Writing parquet file into {} '.format(output_parquet_path_struct))
            ml_data_handler.data_handler.parquet_storer(ml_outputs, output_parquet_path_struct)
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
            logging.info('Writing geotiff file into {} '.format(raster_outputs_path_struct))
            write_dataset_to_raster(output_raster_path='/dbfs'+ raster_outputs_path_struct, dataset=ml_outputs_df_xy, attribute=pollutant, pixel_size_x=1000.0, pixel_size_y=1000.0)

          ml_outputs_df_xy.unpersist()

    logging.info(f'Finished predictions!')

except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={ 'PredValStartDate': str(predval_start_date),'PredValEndDate': str(predval_end_date),'Pollutants': str(pollutants),'Trainset': str(trainset), 'StorePredictions': str(store_predictions)})



# Note if we add some more features/rows to the df, we might need to use SPARK xgboost regressor since pandas cannot support it. If we add it now, we might be using spark for few data (unneficient)


# COMMAND ----------

# # Plot predictions
# # df_spark = spark.createDataFrame(ml_outputs_df_xy)
# ml_outputs_df_xy = ml_outputs_df_xy \
#                               .withColumnRenamed('x', 'x_old') \
#                               .withColumnRenamed('y', 'y_old') \
#                               .withColumn('x', gridid2laea_x_udf('GridNum1km') + F.lit(500)) \
#                               .withColumn('y', gridid2laea_y_udf('GridNum1km') - F.lit(500))
# ml_outputs = None
# df_spark = None
# my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': ml_outputs_df_xy, 'attributes': [pollutant]})
# display(my_map)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Finishing Job

# COMMAND ----------


# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'TrainStartDate': str(train_start_year), 'TrainEndDate': str(train_end_year), 'PredValStartDate': str(predval_start_year),'PredValEndDate': str(predval_end_year),'Pollutants': str(pollutants),'Trainset': str(trainset),'DateOfInput': str(date_of_input),'Version': str(version),'StorePredictions': str(store_predictions)})

notebook_mgr = None


# COMMAND ----------

  for pollutant in pollutants:     
  # In case we have different target variables i.e.: eRep and e1b.
    for target in trainset:
      logging.info(f'Processing pollutant: {pollutant} target {target}.')
      label = [pollutant.upper()][0]

      ml_worker = MLWorker(pollutant)
      ml_data_handler = MLDataHandler(pollutant)
      data_handler = DataHandler(pollutant)
      # Prediction inputs data                                                                   ????? shall we also concatenate preds dataset into the final model training????
      # pollutant_prediction_data, output_parquet_path, raster_output_path = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, None, None, features=['selected'])
      selected_cols_pollutants = data_handler.config.select_cols(data_handler.pollutant) if features[0]=='selected' else ['*'] 
pollutant_prediction_data_all = data_handler.parquet_reader(f'/ML_Input/HOURLY_DATA/data-{pollutant}_2023-05-22_2023-05-28.parquet', features=selected_cols_pollutants).dropna(subset=[pollutant]).drop_duplicates([pollutant, 'GridNum1km', 'date', 'hour'])    # MODIFICAR PATH PARA INPUT DATA DEL PREDICTION


pollutant_predictions = data_handler.parquet_reader(f'/ML_Output/HOURLY_OUTPUTS/{pollutant}_2023-05-22_2023-05-28_maps_HOURLY_TEST', features=selected_cols_pollutants)#.drop_duplicates([pollutant, 'GridNum1km', 'date'])
pollutant_prediction_data_all = pollutant_prediction_data_all.toPandas()
pollutant_predictions = pollutant_predictions.toPandas()
Y_mean = pollutant_prediction_data_all['O3'].mean()
rmse = np.sqrt(mean_squared_error(pollutant_prediction_data_all['O3'], ml_outputs['O3']))
mape = mean_absolute_percentage_error(pollutant_prediction_data_all['O3'], ml_outputs['O3'])
corr = np.corrcoef(pollutant_prediction_data_all['O3'].to_numpy(), ml_outputs['O3'], rowvar=False)
# mqi = self.mqi_calculator(y_test_data, predictions)

# COMMAND ----------

Y_mean = pollutant_prediction_data_all['O3'].mean()
rmse = np.sqrt(mean_squared_error(pollutant_prediction_data_all['O3'], ml_outputs['O3']))
mape = mean_absolute_percentage_error(pollutant_prediction_data_all['O3'], ml_outputs['O3'])
corr = np.corrcoef(pollutant_prediction_data_all['O3'].to_numpy(), ml_outputs['O3'], rowvar=False)
# mqi = self.mqi_calculator(y_test_data, predictions)

# COMMAND ----------

print('Ymean:', Y_mean)
print('rmse:', rmse)
print('mape:', mape)
print('corr:', corr[0][1])

# COMMAND ----------

Ymean: 70.82145477777861
rmse: 28.53018874480328
mape: 150551956001040.28
corr: 0.6157424596206004

# COMMAND ----------



# COMMAND ----------

ml_worker = MLWorker(pollutant)

ml_worker.evaluate_model(ml_model=, predictions=, y_test_data=, bins=100)
