# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets
# MAGIC

# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

"""
================================================================================
Notebook to train a ML model used for predictions of the pollutants. We should only need to modify the widgets for normal executions.
We will store (or not) a trained model into our AzureML experiments + metrics for evaluating results of the model

Arguments:
  + date_of_input: date used to build the path where we are storing our input data
  + features: cols we are willing to use to train our model (all vs selected at our config file)
  + pollutants: list of pollutants we are willing to forecast 
  + predval_end_year: last date for the interval we are willing to forecast (or use for validation)
  + predval_start_year: starting date for the period we are willing to forecast (or use for validation)
  + store_trained_model: bool to determine if we want to store our trained ML model or not 
  + train_end_year: last date we used to train the model (used for the naming of the ML model stored at Azure Experiments)
  + train_start_year: first date we used to train the model (used for the naming of the ML model stored at Azure Experiments)
  + train_model: bool to determine if we want to train a new model or use an existing (pretrained) one
  + trainset: list of the targets we are willing to predict 
  + type_of_params: parameters we will use in our training ML model (test vs optimized
  + version: version of the model (used for naming) 

================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157021
Author   : aiborra-ext@tracasa.es

================================================================================
"""

# Set default parameters for input widgets
DEFAULT_TRAIN_START = '2019'
DEFAULT_TRAIN_END = '2020'
DEFAULT_PREDVAL_START = '2021'
DEFAULT_PREDVAL_END = '2022'
# DEFAULT_VERSION = 'v0'
# DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b', 'CAMS', 'value_numeric']
DEFAULT_FEATURES_LIST = ['*', 'selected']
DEFAULT_PARAMS_LIST = ['optimized', 'test']
DEFAULT_STORE_MODEL_LIST = ['YES', 'NO']
DEFAULT_TRAIN_MODEL_LIST = ['Train', 'Pretrained']
DEFAULT_ADD_CITIES = ['YES', 'NO']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
# dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
# dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('TypeOfParams', 'optimized', DEFAULT_PARAMS_LIST, label='Type of params')  
dbutils.widgets.dropdown('StoreModel', 'NO', DEFAULT_STORE_MODEL_LIST, label='Store Trained Model')  
dbutils.widgets.dropdown('TrainPretrained', 'Train', DEFAULT_TRAIN_MODEL_LIST, label='Train new / Use Pretrained Model')               # We can list available pretrained models. Select any/"Train new model"
dbutils.widgets.dropdown('AddCities', 'NO', DEFAULT_ADD_CITIES, label='Add cities')  


# https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html
# https://docs.databricks.com/_extras/notebooks/source/xgboost-pyspark.html


# COMMAND ----------

# # To list all available pretrained models: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?tabs=python%2Cuse-local

# from azure.ai.ml import MLClient, Input
# from azure.ai.ml.entities import Model
# from azure.ai.ml.constants import AssetTypes
# from azure.identity import DefaultAzureCredential

# subscription_id = "<SUBSCRIPTION_ID>"
# resource_group = "<RESOURCE_GROUP>"
# workspace = "<AML_WORKSPACE_NAME>"

# ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# # List models
# models = ml_client.models.list()
# for model in models:
#     print(model.name)
    
# # List versions for each model 
# models = ml_client.models.list(name="run-model-example")
# for model in models:
#     print(model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables
# MAGIC

# COMMAND ----------

# MAGIC %run "../utils/Lib"
# MAGIC

# COMMAND ----------

import sys
import mlflow
import logging
import numpy as np
import pandas as pd

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import last
from pyspark.sql import Window
# from importlib import reload
# reload(logging)

sys.path.append('/dbfs/FileStore/scripts/eea/databricks')
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))


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
# date_of_input:str = dbutils.widgets.get('DateOfInput')
# version:str = dbutils.widgets.get('Version')
features:list = dbutils.widgets.get('Features') if isinstance(dbutils.widgets.get('Features'), list) else [dbutils.widgets.get('Features')]
type_of_params:str = dbutils.widgets.get('TypeOfParams')
train_model:bool = True if dbutils.widgets.get('TrainPretrained') == 'Train' else False
store_model:bool = True if dbutils.widgets.get('StoreModel') == 'YES' else False
add_cities:bool = True if dbutils.widgets.get('AddCities') == 'YES' else False


# logging.info(f'Your chosen parameters to TRAIN: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_params: "{type_of_params}", train_model: {train_model}, store_model: {store_model}')

logging.info(f'Your chosen parameters to TRAIN: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", pollutants: {pollutants}, trainset: {trainset}, features: {features}, type_of_params: "{type_of_params}", train_model: {train_model}, store_model: {store_model}')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 
if (train_end_year < train_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 
if train_model==False and type_of_params=='test': logging.warning('You have chosen to use a pretrained model so your testing parameters will not be used...')
if train_model==False and features[0]=='*': logging.warning('You have chosen to use a pretrained model so your features "*" will be filtered to the ones the model was trained...')
if train_model==False and store_model==True: logging.warning('You have chosen to use a pretrained model so it is stored already!')


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2. Execute training

# COMMAND ----------


if add_cities:
  from pyspark.sql.functions import col
  from pyspark.sql.types import StringType
  # from pyspark.ml.feature import StringIndexer
  from pyspark.sql.functions import monotonically_increasing_id

  sql_query = """
  SELECT b.level3_code, a.GridNum1km, b.adm_country
  FROM ml_input_from_jedi.aq_adminman_1000 a
  INNER JOIN hra_input_from_sqldbs.aq_admin_lookup b
  ON a.adminbound = b.adm_id
  """

  cities = spark.sql(sql_query).dropDuplicates() 
  cities = cities.withColumn('GridNum1km', col('GridNum1km').cast(StringType()))
  cities = cities.withColumn("country_encoded", monotonically_increasing_id())

  display(cities)
  print(cities.count())




# COMMAND ----------




# # pollutant_train_data_AUX = pollutant_train_data
# # cities = cities.withColumn('GridNum1km', col('GridNum1km').cast(StringType()))
# pollutant_train_data = pollutant_train_data.join(cities, on='GridNum1km', how='left')
# display(pollutant_train_data_AUX_CITIES)
# print(pollutant_train_data_AUX_CITIES.count())

from pyspark.ml.feature import StringIndexer

# indexer = StringIndexer(inputCol="adm_country", outputCol="country_encoded") 
# pollutant_train_data = indexer.fit(pollutant_train_data).transform(pollutant_train_data) 
# display(pollutant_train_data)


# COMMAND ----------

    
for pollutant in pollutants:   
# DEFAULT_POLLUTANTS_LIST = []
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [pollutant.upper()][0]
    
    ml_worker = MLWorker(pollutant, type_of_params='test')
    data_handler = DataHandler(pollutant)
    
    # Collecting and cleaning data
    selected_cols_pollutants = data_handler.config.select_cols(data_handler.pollutant) if features[0]=='selected' else ['*'] 
    pollutant_train_data = data_handler.parquet_reader(f'/ML_Input/HOURLY_DATA/episodes/data-{pollutant}_{train_start_year}_{train_end_year}-all_CAMS.parquet', features=selected_cols_pollutants).dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])
    pollutant_train_data = pollutant_train_data.select([col for col in pollutant_train_data.columns if col not in [pol for pol in DEFAULT_POLLUTANTS_LIST if pol != pollutant]])


    # pollutant_train_data = pollutant_train_data.sample(0.5)




    # pollutant_validation_data = data_handler.parquet_reader(f'/ML_Input/HOURLY_DATA/episodes/data-{pollutant}_{predval_start_year}_{predval_end_year}-all_CAMS.parquet', features=selected_cols_pollutants).dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])
    # # pollutant_validation_data = pollutant_validation_data.select([col for col in pollutant_validation_data.columns if col not in [pol for pol in DEFAULT_POLLUTANTS_LIST if pol != pollutant]])



    # pollutant_validation_data = pollutant_validation_data.sample(0.1)





    pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0) & (pollutant_train_data[label] <= data_handler.config.validate_pollutant_values(pollutant)))
    # pollutant_validation_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0) & (pollutant_validation_data[label] <= data_handler.config.validate_pollutant_values(pollutant)))
    

    # duplicated_rows = data_handler.find_duplicates(df1=pollutant_train_data, df2=pollutant_validation_data, cols_to_compare=['Gridnum1km','date', 'hour', f'CAMS_{pollutant}', pollutant])
    # logging.warning(f'¡¡¡ WARNING !!! There are duplicates in your training and validation set: {duplicated_rows.count()}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')


    if add_cities:
      logging.info('Joining regions with training gridnums...')
      no_cities_count = pollutant_train_data.count()
      no_cities_count_val = pollutant_validation_data.count()
      pollutant_train_data = pollutant_train_data.join(cities, on='Gridnum1km', how='left').na.drop(subset=[label]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour', f'CAMS_{pollutant}'])
      if pollutant_train_data.count() != no_cities_count: logging.warning(f'¡¡¡ WARNING !!! There is a different number of records after the cities join: {no_cities_count} VS {pollutant_train_data.count()}')

      logging.info('Joining regions with validation gridnums...')
      pollutant_validation_data = pollutant_validation_data.join(cities, on='Gridnum1km', how='left').na.drop(subset=[label]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour', f'CAMS_{pollutant}'])
      if pollutant_validation_data.count() != no_cities_count_val: logging.warning(f'¡¡¡ WARNING !!! There is a different number of records after the cities join: {no_cities_count_val} VS {pollutant_validation_data.count()}')


    # Preparing data for training/validating/predicting
    df_train = pollutant_train_data.drop('GridNum1km','AreaHa', 'level3_code', 'adm_country', 'date', 'datetime_end', 'datetime_begin').toPandas()         # DELETE DAY FROM DROP?
    if add_cities: df_train['country_encoded'] = df_train['country_encoded'].astype(int)
                                       
    # df_validation = pollutant_validation_data.drop('GridNum1km', 'Year','AreaHa', 'level3_code', 'adm_country', 'date', 'datetime_end', 'datetime_begin').toPandas()    
    if add_cities: df_validation['country_encoded'] = df_validation['country_encoded'].astype(int)
    
    
    X_train, X_test, Y_train, Y_test = ml_worker.split_data(df=df_train, train_size=0.7, label=pollutant)
    if not store_model: logging.info(f'Data is ready! Training model with: \n{X_train.count()} \n') 
    df_train = None
                   
    # X_train , Y_train = df_train[[col for col in df_train.columns if col not in label]], df_train[[label]] 
    # X_test, Y_test = df_validation[[col for col in df_validation.columns if col not in label]], df_validation[[label]]

    # # Executing selected ML model
    # model_to_train_details = {'model_name': f"{pollutant}_{ml_worker.ml_models_config.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}",
    #                           'model_to_train' : ml_worker.model_to_train} 


    # pollutant_train_data[pollutant] = pollutant_train_data[pollutant].ffill()
    # pollutant_validation_data[pollutant] = pollutant_validation_data[pollutant].ffill()
    # pollutant_train_data = pollutant_train_data.drop_duplicates(subset=['Gridnum1km','date', 'hour', f'CAMS_{pollutant}', pollutant], keep='first')

    
    # # display(pollutant_train_data)
    # X_train, X_test, Y_train, Y_test = ml_worker.split_data(df=pollutant_train_data, train_size=0.7, label=pollutant)

    # # pollutant_train_data, pollutant_validation_data = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year, features)
    # # pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0))
    # # pollutant_validation_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0))
    # logging.info('Data pollutant collected! Checking for duplicated data among your training and validation datasets...')

    # # # Making sure we do not have duplicates among training and validation datasets
    # trainset_df = pd.concat([X_train, Y_train], axis=1)
    # testset_df = pd.concat([validation_X, validation_Y], axis=1)
    # duplicated_rows = data_handler.find_duplicates(df1=spark.createDataFrame(trainset_df), df2=spark.createDataFrame(testset_df), cols_to_compare=['Gridnum1km','date', 'hour', f'CAMS_{pollutant}', pollutant])
    # # duplicated_rows = data_handler.find_duplicates(df1=spark.createDataFrame(X_train), df2=spark.createDataFrame(X_test), cols_to_compare=['GridNum1km','date', 'hour'])
    # logging.warning(f'¡¡¡ WARNING !!! There are duplicates in your training and validation set: {duplicated_rows.count()}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')

    # # # Preparing data for training/validating/predicting
    # X_train = X_train[[col for col in X_train.columns if col not in ['Gridnum1km', 'AreaHa', 'datetime_begin', 'datetime_end', 'resulttime', 'eucode', 'ns', 'polu', 'sta', 'Lat', 'Lon', 'AirQualityStationEoICode', 'date', 'level3_code', 'adm_country', '__index_level_0__']]]           # categorical features                             
    # X_test = X_test[[col for col in X_test.columns if col not in ['Gridnum1km', 'AreaHa', 'datetime_begin', 'datetime_end', 'resulttime', 'eucode', 'ns', 'polu', 'sta', 'Lat', 'Lon', 'AirQualityStationEoICode', 'date', 'level3_code', 'adm_country', '__index_level_0__']]]                 # categorical features 
    # Executing selected ML model
    model_to_train_details = {'model_name': f"{pollutant}_{ml_worker.ml_models_config.model_str.replace('()', '')}_hourly",
                              'model_to_train' : ml_worker.model_to_train} 
    
    if store_model:
      # Training final model: joining training + validation sets                             ????? shall we also concatenate preds dataset into the final model training????
      train_val_X, train_val_Y = pd.concat([X_train, X_test]), pd.concat([Y_train, Y_test])
      logging.info(f'Joining training and validation datasets... We will train the final model with: \n{train_val_X.count()}. Evaluations will not be performed \n')

      # Storing trained model into AzureML experiments
      with mlflow.start_run():
        mlflow.autolog()
        trained_model = ml_worker.train_load_ml_model(model_name=model_to_train_details['model_to_train'], X_train_data=train_val_X, Y_train_data=train_val_Y)                 
        run_id = mlflow.active_run().info.run_id
        logging.info(f'Registering model: {model_to_train_details["model_name"]}. A {ml_worker.ml_models_config.model_str} model trained with {type_of_params.upper()} params {ml_worker.ml_params}...')
        # The default path where the MLflow autologging function stores the model
        artifact_path = "model"
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_to_train_details['model_name'])
        mlflow.end_run()

    else:    
      if train_model:
        logging.info(f'Training model {ml_worker.ml_models_config.model_str} with {type_of_params.upper()} params {ml_worker.ml_params}')
        trained_model = ml_worker.train_load_ml_model(model_name=model_to_train_details['model_to_train'], X_train_data=X_train, Y_train_data=Y_train)  
      else:
        # In case we are willing to re-evaluate an existing pretrained model
        logging.info(f'Loading pretrained model {model_to_train_details["model_name"]}')
        logging.warning(f'Note you are using a pretrained model so, feature importances will not be plot.')
        trained_model = ml_worker.train_load_ml_model(model_name=model_to_train_details['model_name'], X_train_data=X_train, Y_train_data=Y_train)
          
      predictions = trained_model.predict(X_test)
      X_test = None
      results, rmse, mape, importance_scores = ml_worker.evaluate_model(trained_model, predictions, Y_test, bins=100)


logging.info(f'Finished training!')



# COMMAND ----------

display(X_train)

# COMMAND ----------

    
   
    results = pd.concat([Y_test.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    results.columns = ['actual', 'forecasted']
    results = results[results.actual<=1200]
    # Plot histogram showing errors in predictions
    diff_results = results.actual - results.forecasted
    diff_results.hist(bins=300, figsize = (20,10))
    plt.title('ML predict Errors distribution')


# COMMAND ----------

    
   
    results = pd.concat([Y_test.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    results.columns = ['actual', 'forecasted']
    results = results[results.actual<=1200]
    # Plot histogram showing errors in predictions
    diff_results = results.actual - results.forecasted
    diff_results.hist(bins=100, figsize = (20,10))
    plt.title('ML predict Errors distribution')


# COMMAND ----------

# Are we using as input the predictions from other pollutants but lagged? It is weird they are that relevant for predictions

# COMMAND ----------

    thresholds = {
      'no2':{'beta':2, 'urv95r': 0.24, 'rv': 200, 'alfa': 0.2, 'np': 5.2, 'nnp':5.5},
      'pm10':{'beta':2, 'urv95r': 0.28, 'rv': 50, 'alfa': 0.25, 'np': 20, 'nnp':1.5},
      'pm25':{'beta':2, 'urv95r': 0.36, 'rv': 25, 'alfa': 0.5, 'np': 20, 'nnp':1.5},
      'o3':{'beta':2, 'urv95r': 0.18, 'rv': 120, 'alfa': 0.79, 'np': 11, 'nnp':3},
    }

# COMMAND ----------

    
  def mqi_calculator(y_test_data, predictions, thresholds, pollutant):
    """
    Calculates the Model Quality Index (MQI) using the input test data and predictions.

    Parameters:
    -----------
    y_test_data : pd.DataFrame
        The test data for the model.
    predictions : array-like
        The predictions made by the model.

    Returns:
    --------
    mqi : array-like
        The calculated MQI for the input test data and predictions.
    """
    thresholds=thresholds[pollutant.lower()]
    y_test_rav = y_test_data.to_numpy().ravel()

    uncertainty = thresholds['urv95r']*np.sqrt(
                                              (1-np.square(thresholds['alfa']))
                                              *(np.square(np.mean(y_test_rav)) + np.square(np.std(y_test_rav)))
                                              +np.square(thresholds['alfa'])*np.square(thresholds['rv']))

    rmse = np.sqrt(mean_squared_error(y_test_data, predictions))
    mqi = rmse/(thresholds['beta']*uncertainty)

    return mqi

# COMMAND ----------

mqi_calculator(Y_test, predictions, thresholds, pollutant)

# COMMAND ----------

0.5480435975128095

# COMMAND ----------

pollutant_validation_data = data_handler.parquet_reader(f'/ML_Input/episodes/data-{pollutant}_{predval_start_year}_{predval_end_year}-all_episodes.parquet', features=selected_cols_pollutants).dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])

pollutant_validation_data.count()

# COMMAND ----------

pollutant_train_data = data_handler.parquet_reader(f'/ML_Input/episodes/data-{pollutant}_{train_start_year}_{train_end_year}-all_episodes.parquet', features=selected_cols_pollutants).dropna(subset=[pollutant]).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])
pollutant_train_data.count()

# COMMAND ----------

results[results.actual>=1200]

# COMMAND ----------



import holidays


def is_holiday(country, date):
    """
    Returns 1 if the date is a holiday in the given country, 0 if it is not,
    and None if the country is not found in the `country_holidays` dictionary.
    """
    try:
        holiday_list = holidays.CountryHoliday(country, prov=None, state=None)
    except KeyError:
        print(f"Country {country} not found")
        return None

    if date in holiday_list:
        return 1
    else:
        return 0

pollutant_train_data_test['is_holiday'] = pollutant_train_data_test.apply(lambda row: is_holiday(row['adm_country'], row['date']), axis=1)

# COMMAND ----------

display(pollutant_train_data_test)

# COMMAND ----------



# COMMAND ----------

display(pollutant_train_data)

# COMMAND ----------


