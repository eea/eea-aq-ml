# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# dbutils.widgets.removeAll()
# 2023-03-27 10:39:34,607 INFO     Your chosen parameters: train_start_year: "2013", train_end_year: "2020", predval_start_year: "2013", predval_end_year: "2021", pollutants: ['PM10'], trainset: ['eRep'], date_of_input: "20220729", version: "v0", features: ['selected'], type_of_params: "optimized", store_model: False, train_model: "False", store_predictions:"False"

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
DEFAULT_FEATURES_LIST = ['*', 'selected']
DEFAULT_PARAMS_LIST = ['optimized', 'test']
DEFAULT_STORE_MODEL_LIST = ['YES', 'NO']
DEFAULT_TRAIN_MODEL_LIST = ['YES', 'NO']
DEFAULT_STORE_PREDICTIONS_LIST = ['YES', 'NO']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.dropdown('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('TypeOfParams', 'optimized', DEFAULT_PARAMS_LIST, label='Type of params')  
dbutils.widgets.dropdown('StoreModel', 'NO', DEFAULT_STORE_MODEL_LIST, label='Store Trained Model')  
dbutils.widgets.dropdown('TrainModel', 'NO', DEFAULT_TRAIN_MODEL_LIST, label='Train Model')  
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

from pyspark.sql import SparkSession
from pyspark.sql.types import LongType
import pyspark.sql.functions as F


# Import and register 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
gridid2laea_x_udf = spark.udf.register('gridid2laea_x', CalcGridFunctions.gridid2laea_x, LongType())
gridid2laea_y_udf = spark.udf.register('gridid2laea_y', CalcGridFunctions.gridid2laea_y, LongType())

# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('CPL_CURL_VERBOSE', 'NO')
gdal.SetConfigOption('CPL_DEBUG', 'NO')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')


# COMMAND ----------

import sys
import mlflow
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from xgboost import XGBRegressor, plot_importance
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

# from importlib import reload
# reload(logging)

sys.path.append('/dbfs/FileStore/scripts/eea/databricks')
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))


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
features:list = dbutils.widgets.get('Features') if isinstance(dbutils.widgets.get('Features'), list) else [dbutils.widgets.get('Features')]
type_of_params:str = dbutils.widgets.get('TypeOfParams')
train_model:bool = True if dbutils.widgets.get('TrainModel') == 'YES' else False
store_model:bool = True if dbutils.widgets.get('StoreModel') == 'YES' else False
store_predictions:bool = True if dbutils.widgets.get('StorePredictions') == 'YES' else False


logging.info(f'Your chosen parameters: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_params: "{type_of_params}", store_model: {store_model}, train_model: "{train_model}", store_predictions:"{store_predictions}"')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 
if (train_model == False) & (store_model == True): raise Exception('Set Train Model = "YES" if you are willing to store the model, otherwise set Store Trained Model = "NO". You will need to train the model before storing it!') 

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Functions
# MAGIC 
# MAGIC <br/>

# COMMAND ----------

# def split_data(df: pd.DataFrame, train_size:float=0.7, label:list=None):
#     """Splits training dataframe into training and test dataframes to train and validate ML models
#     Params
#     ------
#       :df: pd.DataFrame = data we are willing to use to train de model
#       :train_size: float = percentage of the dataframe we are willing to use to train
#       :label: list = contains the value/s willing to predict
      
#     Returns
#     -------
#       :X_train: str = data the model will use to train 
#       :X_test: str = unseen data the model will use to make predictions
#       :Y_train: str = values the model will use with the training set to train its predictions
#       :Y_test: str = unseen values the model will try to predict
#   """

#     # Separating values to be predicted from the data used to train
#     df_x = df[[col for col in df.columns if col not in label]]
#     df_y = df.select(label)

#     # Splitting the dataframe
#     X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_x, df_y, train_size=train_size, random_state=42)                        
  
#     return X_train, X_test, Y_train, Y_test
  
  
# def train_predict_ml_model(train_model_flag:bool, store_model:bool, model, X_train_data:pd.DataFrame=None, Y_train_data:pd.DataFrame=None, X_test_data:pd.DataFrame=None):
def train_load_ml_model(train_model_flag:bool, model, X_train_data:pd.DataFrame=None, Y_train_data:pd.DataFrame=None):
    """Trains a ML model and/or predicts data. It will store/load a ML model to/from azure experiments or execute the input model.
    Params
    ------
      :train_model_flag: bool = Flag to execute/skip training. If False, only predictions will take place (useful when validation)
      :store_model: bool = Flag to store (or not) the trained model into azure experiments.
      :model: [Object, str] = If we pass a string indicating the name of a model it will look for it at our azure experiments otherwise it will execute the input model
      :X_train_data: pd.DataFrame = data we are willing to use to train de model
      :Y_train_data: pd.DataFrame = label we are willing to use to predict on the training set
      :X_test_data: pd.DataFrame = data we are willing to use to make predictions
      
    Returns
    -------     
      :ml_model: float = score our model obtained
      :predictions: pd.DataFrame = predictions performed by our model and its actual value
  """
    
    if train_model_flag:
      print('Training model...')
#       with mlflow.start_run():
#         mlflow.autolog()
      # Training model with the input data  
      model_name = model['model_name']
      ml_model = model['model_to_train'].fit(X_train_data, Y_train_data)
#         if store_model:
#           run_id = mlflow.active_run().info.run_id
#           print('Registering model...')
#           # The default path where the MLflow autologging function stores the model
#           artifact_path = "model"
#           model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
#           model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
  
    else:
      if isinstance(model, str):
        print('Loading and executing stored model...')
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model)
        print('Loading latest version of your pretrained models: ', latest_version)
        ml_model = mlflow.pyfunc.load_model(latest_version[0].source)
        
      else:
        print('Executing trained model...')
        ml_model = model['model_to_train']
        

#     print('Performing predictions...')
#     predictions = ml_model.predict(X_test_data)

#     return ml_model, predictions

    return ml_model



def evaluate_model(ml_model, predictions:pd.DataFrame, y_test_data:pd.DataFrame, bins:int=40):
    """It will plot some plots showing performance of the model and feature importances (fscore).
    Params
    ------
      :ml_model: Object = ML model we are willing to evaluate
      :predictions: pd.DataFrame = obtained predictions from our model
      :y_test_data: pd.DataFrame = label we are willing to predict and will use to check performance of the model 
      
    Returns
    -------
      :results: pd.DataFrame = predictions performed by our model and its actual value
      :rmse: float = standard deviation of the residuals predicted from our model
      :mape: float = mean of the absolute percentage errors predicted from our model (note that bad predictions can lead to arbitrarily large MAPE values, especially if some y_true values are very close to zero.).
      :importance_scores: float = score given to each feature from our model based on fscore (number of times a variable is selected for splitting, weighted by the squared improvement to the model as a result of each split, and averaged over all trees)
  """

    # Get scores for model predictions
    rmse = np.sqrt(mean_squared_error(y_test_data, predictions))
    mape = mean_absolute_percentage_error(y_test_data, predictions)
    try:
      # Finding pollutant and target names in the y_test dataset columns
      label = [col.split('_') for col in y_test_data.columns][0]
      pollutant =  label[1] if len(label)>1 else label[0]
      target =  label[0] 
    except:
      print('Pollutant and target names could not be found')

    print(f"\n{pollutant}-{target} RMSE : {round(rmse, 3)}\n")
    print(f"\n{pollutant}-{target} MAPE : {round(mape, 3)}%\n")

    results = pd.concat([y_test_data.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    results.columns = ['actual', 'forecasted']

    # Plotting lines for actuals vs forecasts data
    fig = px.line(results)
    fig.show()
    
    # Plot histogram showing errors in predictions
    diff_results = results.actual - results.forecasted
    diff_results.hist(bins=bins, figsize = (20,10))
    plt.title('ML predict Errors distribution')

    try:
      # Feature importance of the trained model
      fig, ax = plt.subplots(figsize=(12,12))
      plot_importance(ml_model, ax=ax)
      importance_scores = ml_model.get_booster().get_fscore()
      plt.title(f'Feature importance (fscore) for target {target} & pollutant {pollutant}')
      plt.show();
    except:
      print('Feature importance could not be calculated!')

    return results, rmse, mape, importance_scores

  
def write_dataset_to_raster(raster_file, dataset, attribute, x_attrib='x', y_attrib='y',
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
        temp_file = os.path.join(temp_dir, temp_name) + os.path.splitext(raster_file)[1]
        
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
        if raster_file.startswith('/dbfs'):
            final_file = raster_file[5:]
            dbutils.fs.cp('file:' + temp_file, 'dbfs:' + final_file)
        else:
            import shutil
            shutil.copy2(temp_file, raster_file)
            
    return raster_file

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Execute training

# COMMAND ----------

for pollutant in pollutants:   
  
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]
    ml_models_config = MLModelsConfig(pollutant, type_of_params)                  
    
    if train_model:
      ml_data_handler = MLDataHandler(pollutant)
      logging.info('Training model...')

      # Collecting and cleaning data
      pollutant_train_data, pollutant_validation_data = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year, features)
      pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0))
      pollutant_validation_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0))
      logging.info('Data pollutant collected! Checking for duplicated data among your training and validation datasets...')

      # Making sure we do not have duplicates among training and validation datasets
      duplicated_rows = ml_data_handler.find_duplicates(df1=pollutant_train_data, df2=pollutant_validation_data, cols_to_compare=['GridNum1km','Year'])
      logging.warning(f'There are duplicates in your training and validation set: {duplicated_rows}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')

      # Preparing data for training/validating/predicting
      df_train = pollutant_train_data.drop('GridNum1km', 'Year','AreaHa').toPandas()                                          
      df_validation = pollutant_validation_data.drop('GridNum1km', 'Year','AreaHa').toPandas()                                         
      X_train , Y_train = df_train[[col for col in df_train.columns if col not in label]], df_train[[label]] 
      validation_X, validation_Y = df_validation[[col for col in df_validation.columns if col not in label]], df_validation[[label]]
      if not store_model: logging.info(f'Data is ready! Training & validating model with: \n{X_train.count()} \n') 

      # Executing selected ML model
      model_to_train, ml_params = ml_models_config.prepare_model()
      logging.info(f'Preparing training model {ml_models_config.model_str} for pollutant {pollutant} and {type_of_params.upper()} params: {ml_params}') if train_model else logging.info('Loading latest pretrained model to make predictions...')
      model_to_train_details = {'model_name': f"{pollutant}_{ml_models_config.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}",
                                'model_to_train' : model_to_train}
      
      if store_model:
      # Training final model: training + validation sets                                                                                                                      ????? shall we also concatenate preds dataset into the final model training????
        train_val_X, train_val_Y = pd.concat([X_train, validation_X]), pd.concat([Y_train, validation_Y])
        logging.info(f'Joining training and validation datasets... We will train the final model with: \n{train_val_X.count()} \n')

        with mlflow.start_run():
          mlflow.autolog()
          trained_model = train_load_ml_model(train_model_flag=True, model=model_to_train_details, X_train_data=train_val_X, Y_train_data=train_val_Y)                              
          run_id = mlflow.active_run().info.run_id
          print('Registering model...')
          # The default path where the MLflow autologging function stores the model
          artifact_path = "model"
          model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
          model_details = mlflow.register_model(model_uri=model_uri, name=model_to_train_details['model_name'])
          mlflow.end_run()

      else:
        logging.info(f'Performing predictions with features: {validation_X.count()}')
        trained_model = train_load_ml_model(train_model_flag=True, model=model_to_train_details, X_train_data=X_train, Y_train_data=Y_train)                              
        predictions = trained_model.predict(validation_X)

        results, rmse, mape, importance_scores = evaluate_model(trained_model, predictions, validation_Y, bins=100)

    else:
      ml_data_handler = MLDataHandler(pollutant)
      
      # Prediction inputs data                                                                                                                                                    ????? shall we also concatenate preds dataset into the final model training????
      pollutant_prediction_data, output_path = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, None, None, features)
      logging.info(f'Performing predictions with features: {pollutant_prediction_data.count()}')

      # Predicting data using a stored pretrained model
      model_name = f"{pollutant}_{ml_models_config.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}"
      logging.info('Performing predictions with loaded model: ', model_name)
      pollutant_prediction_data_pd = pollutant_prediction_data.toPandas()
      trained_model = train_load_ml_model(train_model_flag=False, model=model_name, X_train_data=None, Y_train_data=None)
      predictions = trained_model.predict(pollutant_prediction_data_pd)

      predictions_df = pd.DataFrame(predictions, columns=[pollutant.upper()])
      ml_outputs = pd.concat([pollutant_prediction_data_pd[['GridNum1km', 'Year']], predictions_df], axis=1)
      
      # Dealing with memory issues
      predictions_df = None
      pollutant_prediction_data_pd = None
      predictions = None
      
      if store_predictions:
        logging.info('Writing parquet file {}... '.format(output_path))
        ml_data_handler.data_handler.parquet_storer(ml_outputs, output_path)
        
        
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
        
        # # #Write to geotiff
        raster_file = '/dbfs' +  ml_data_handler.data_handler.file_system_path + '/StaticData/testgeotif444444.tiff'
        logging.info('Writing geotiff file {}... '.format(raster_file))
        write_dataset_to_raster(raster_file, ml_outputs_df_xy, attribute=pollutant, pixel_size_x=1000.0, pixel_size_y=1000.0)

        ml_outputs_df_xy.unpersist()
      
      
      
# Note if we add some more features/rows to the df, we will need to use SPARK xgboost regressor since pandas cannot support it. If we add it now, we might be using spark for few data (unneficient)
# #  REMEMBER TO: Perform training with the whole dataset (training + validation + prediction sets) once we have the final model


logging.info(f'Finished!')

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

validation_X

# COMMAND ----------



# COMMAND ----------

my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': ml_outputs_df_xy, 'attributes': [pollutant]})
display(my_map)

# COMMAND ----------


