# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

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
  + type_of_params: parameters we will use in our training ML model (test vs optimized)
  + version: version of the model (used for naming) 

================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157021
Author   : aiborra-ext@tracasa.es

================================================================================
"""

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
DEFAULT_TRAIN_MODEL_LIST = ['Train', 'Pretrained']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('TypeOfParams', 'optimized', DEFAULT_PARAMS_LIST, label='Type of params')  
dbutils.widgets.dropdown('StoreModel', 'NO', DEFAULT_STORE_MODEL_LIST, label='Store Trained Model')  
dbutils.widgets.dropdown('TrainPretrained', 'Train', DEFAULT_TRAIN_MODEL_LIST, label='Train new / Use Pretrained Model')               # We can list available pretrained models. Select any/"Train new model"


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

# COMMAND ----------

# MAGIC %run "../utils/Lib"

# COMMAND ----------

import sys
import mlflow
import logging
import numpy as np
import pandas as pd

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
date_of_input:str = dbutils.widgets.get('DateOfInput')
version:str = dbutils.widgets.get('Version')
features:list = dbutils.widgets.get('Features') if isinstance(dbutils.widgets.get('Features'), list) else [dbutils.widgets.get('Features')]
type_of_params:str = dbutils.widgets.get('TypeOfParams')
train_model:bool = True if dbutils.widgets.get('TrainPretrained') == 'Train' else False
store_model:bool = True if dbutils.widgets.get('StoreModel') == 'YES' else False


logging.info(f'Your chosen parameters to TRAIN: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_params: "{type_of_params}", train_model: {train_model}, store_model: {store_model}')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 
if train_model==False and type_of_params=='test': logging.warning('You have chosen to use a pretrained model so your testing parameters will not be used...')
if train_model==False and features[0]=='*': logging.warning('You have chosen to use a pretrained model so your features "*" will be filtered to the ones the model was trained...')
if train_model==False and store_model==True: logging.warning('You have chosen to use a pretrained model so it is stored already!')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Execute training

# COMMAND ----------

for pollutant in pollutants:   
  
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]
    
    ml_worker = MLWorker(pollutant, type_of_params)
    ml_data_handler = MLDataHandler(pollutant)
    
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
    if not store_model: logging.info(f'Data is ready! Training model with: \n{X_train.count()} \n') 

    # Executing selected ML model
    model_to_train_details = {'model_name': f"{pollutant}_{ml_worker.ml_models_config.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}",
                              'model_to_train' : ml_worker.model_to_train} 
    
    if store_model:
      # Training final model: joining training + validation sets                             ????? shall we also concatenate preds dataset into the final model training????
      train_val_X, train_val_Y = pd.concat([X_train, validation_X]), pd.concat([Y_train, validation_Y])
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
        
      predictions = trained_model.predict(validation_X)
      results, rmse, mape, importance_scores = ml_worker.evaluate_model(trained_model, predictions, validation_Y, bins=100)


logging.info(f'Finished training!')
