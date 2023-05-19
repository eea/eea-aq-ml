# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

"""
================================================================================
Notebook optimize parameters to train a ML model used for predictions of the pollutants. We should only need to modify the widgets for normal executions.

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
# DEFAULT_TRAIN_START = '2016'
# DEFAULT_TRAIN_END = '2019'
# DEFAULT_PREDVAL_START = '2020'
# DEFAULT_PREDVAL_END = '2020'
# DEFAULT_VERSION = 'v0'
# DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b', 'CAMS', 'value_numeric']
DEFAULT_FEATURES_LIST = ['*', 'selected']
DEFAULT_ADD_CITIES = ['YES', 'NO']

# DEFAULT_PARAMS_LIST = ['optimized', 'test']
# DEFAULT_STORE_MODEL_LIST = ['YES', 'NO']
# DEFAULT_TRAIN_MODEL_LIST = ['Train', 'Pretrained']

# Set widgets for notebook
# dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
# dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
# dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
# dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
# dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
# dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('AddCities', 'YES', DEFAULT_ADD_CITIES, label='Add cities')  



# dbutils.widgets.dropdown('TypeOfParams', 'optimized', DEFAULT_PARAMS_LIST, label='Type of params')  
# dbutils.widgets.dropdown('StoreModel', 'NO', DEFAULT_STORE_MODEL_LIST, label='Store Trained Model')  
# dbutils.widgets.dropdown('TrainPretrained', 'Train', DEFAULT_TRAIN_MODEL_LIST, label='Train new / Use Pretrained Model')               # We can list available pretrained models. Select any/"Train new model"


# https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html
# https://docs.databricks.com/_extras/notebooks/source/xgboost-pyspark.html


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

# MAGIC %run "./Lib"

# COMMAND ----------

import sys
# import mlflow
import logging
import numpy as np
import pandas as pd

from hyperopt.pyll.base import scope
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
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
# train_start_year:str = dbutils.widgets.get('TrainStartDate')
# train_end_year:str = dbutils.widgets.get('TrainEndDate')
# predval_start_year:str = dbutils.widgets.get('PredValStartDate')
# predval_end_year:str = dbutils.widgets.get('PredValEndDate')
pollutants:list = dbutils.widgets.get('Pollutants').split(',')
trainset:list = dbutils.widgets.get('Trainset').split(',')
# date_of_input:str = dbutils.widgets.get('DateOfInput')
# version:str = dbutils.widgets.get('Version')
features:list = dbutils.widgets.get('Features') if isinstance(dbutils.widgets.get('Features'), list) else [dbutils.widgets.get('Features')]
add_cities:bool = True if dbutils.widgets.get('AddCities') == 'YES' else False

# type_of_params:str = dbutils.widgets.get('TypeOfParams')
# train_model:bool = True if dbutils.widgets.get('TrainPretrained') == 'Train' else False
# store_model:bool = True if dbutils.widgets.get('StoreModel') == 'YES' else False


# logging.info(f'Your chosen parameters to TRAIN: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_params: "{type_of_params}", train_model: {train_model}, store_model: {store_model}')

# logging.info(f'Your chosen parameters to TRAIN: pollutants: {pollutants}, trainset: {trainset}, features: {features}, type_of_params: "{type_of_params}", train_model: {train_model}, store_model: {store_model}')
logging.info(f'Your chosen parameters to TRAIN: pollutants: {pollutants}, trainset: {trainset}, features: {features}')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
# if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 
# if train_model==False and type_of_params=='test': logging.warning('You have chosen to use a pretrained model so your testing parameters will not be used...')
# if train_model==False and features[0]=='*': logging.warning('You have chosen to use a pretrained model so your features "*" will be filtered to the ones the model was trained...')
# if train_model==False and store_model==True: logging.warning('You have chosen to use a pretrained model so it is stored already!')


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Collect data

# COMMAND ----------


if add_cities:
  from pyspark.sql.functions import col
  from pyspark.sql.types import StringType
  from pyspark.ml.feature import StringIndexer

  sql_query = """
  SELECT b.level3_code, a.GridNum1km, b.adm_country
  FROM ml_input_from_jedi.aq_adminman_1000 a
  INNER JOIN hra_input_from_sqldbs.aq_admin_lookup b
  ON a.adminbound = b.adm_id
  """

  cities = spark.sql(sql_query).dropDuplicates() 
  cities = cities.withColumn('GridNum1km', col('GridNum1km').cast(StringType()))

  display(cities)
  cities.count()




# COMMAND ----------

train_start_year:str = '2016'
train_end_year:str = '2019'
predval_start_year:str = '2020'
predval_end_year:str = '2020'
date_of_input:str = '20230201'
version:str = 'v0'
pollutants = ['PM10']
trainset=['eRep']
features=  ['AreaHa',
            'GridNum1km',
            'Year',
            'avg_eudem',
            'avg_smallwoody_mean_sur',
            'biogeoregions_4',
            'biogeoregions_7',
            'cams_PM10',
            'carbonseqcapacity_mean_sur',
            'climate_HU',
            'climate_RR',
            'climate_TG',
            'climate_TX',
            'droughtimp',
            'eRep_PM10',
            'ecoclimaregions_1',
            'ecoclimaregions_7',
            'max_fga2015',
            'max_imp2015',
            'pop2018',
            'std_eudem',
            'sum_clc18_112',
            'sum_clc18_231_mean_sur',
            'sum_clc18_312_mean_sur',
            'sum_elevbreak_Inlands',
            'sum_elevbreak_Mountains',
            'sum_envzones_BOR',
            'sum_envzones_PAN',
            'sum_urbdegree_11_mean_sur',
            'sum_urbdegree_30',
            'weight_urb',
            'windspeed_mean_sur']

# COMMAND ----------

    
for pollutant in pollutants:   
  
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]
    
    ml_worker = MLWorker(pollutant)
    data_handler = DataHandler(pollutant)
    ml_data_handler = MLDataHandler(pollutant)
    # Collecting and cleaning data
    # selected_cols_pollutants = data_handler.config.select_cols(data_handler.pollutant) if features[0]=='selected' else ['*'] 


    # pollutant_train_data = data_handler.parquet_reader(f'/ML_Input/episodes/data-{pollutant}_2019_2020.parquet', features=features).drop_duplicates([pollutant, 'Gridnum1km', 'date', 'hour'])
    print(features)
    pollutant_train_data = data_handler.parquet_reader(f'/ML_Input/data-PM10_2020-2020/20230201_v0/training_input_eRep_PM10_2016-2019.parquet', features=features)
    pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0))
    display(pollutant_train_data)


    pollutant_train_data = pollutant_train_data.toPandas()
    # pollutant_train_data[pollutant] = pollutant_train_data[pollutant].ffill()
    # pollutant_train_data = pollutant_train_data.drop_duplicates(subset=['GridNum1km','Year'], keep='first')
    # display(pollutant_train_data)
    X_train, X_test, Y_train, Y_test = ml_worker.split_data(df=pollutant_train_data, train_size=0.7, label=f'eRep_{pollutant}')

    # pollutant_train_data, pollutant_validation_data = ml_data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year, features)
    # pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0))
    # pollutant_validation_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0))
    logging.info('Data pollutant collected! Checking for duplicated data among your training and validation datasets...')

    # Making sure we do not have duplicates among training and validation datasets
    trainset_df = pd.concat([X_train, Y_train], axis=1)
    testset = pd.concat([X_test, Y_test], axis=1)
    duplicated_rows = data_handler.find_duplicates(df1=spark.createDataFrame(trainset_df), df2=spark.createDataFrame(testset), cols_to_compare=['GridNum1km','Year'])
    # duplicated_rows = data_handler.find_duplicates(df1=spark.createDataFrame(X_train), df2=spark.createDataFrame(X_test), cols_to_compare=['GridNum1km','date', 'hour'])
    logging.warning(f'¡¡¡ WARNING !!! There are duplicates in your training and validation set: {duplicated_rows.count()}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')

    # # Preparing data for training/validating/predicting
    X_train = X_train[[col for col in X_train.columns if col not in ['Gridnum1km', 'AreaHa', 'datetime_begin', 'datetime_end', 'resulttime', 'eucode', 'ns', 'polu', 'sta', 'Lat', 'Lon', 'AirQualityStationEoICode', 'date', 'level3_code', 'adm_country', '__index_level_0__']]]           # categorical features                             
    X_test = X_test[[col for col in X_test.columns if col not in ['Gridnum1km', 'AreaHa', 'datetime_begin', 'datetime_end', 'resulttime', 'eucode', 'ns', 'polu', 'sta', 'Lat', 'Lon', 'AirQualityStationEoICode', 'date', 'level3_code', 'adm_country', '__index_level_0__']]]                 # categorical features 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2. Execute GridSearchCV

# COMMAND ----------

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

originals = {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 0.3, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7}


parameters = {
              'learning_rate': [.07, .2, .5], #so called `eta` value
              'max_depth': [5, 7, 10],
              'min_child_weight': [2, 4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500, 1000]}





parameters = {
  'colsample_bytree': list(np.linspace(0.6, 1, 100)),
  'learning_rate': list(np.logspace(np.log10(0.01), np.log10(0.9), base = 10, num = 1000)), 
  'max_depth': list(range(3, 15, 2)),
  'min_child_weight': list(range(5, 50, 5)), 
  'n_estimators': list(range(700, 5000, 200)),
  'nthread': [4], 
  'subsample': list(np.linspace(0.6, 1, 20)), 
  'gamma': list(range(5, 15, 3)),
  'reg_alpha': list(np.linspace(0, 1)), 
  'reg_lambda': list(np.linspace(0, 1)),

}



# parameters = {
#   'colsample_bytree': [0.7], 
#   'learning_rate':[0.2], 
#   'max_depth': [10], 
#   'min_child_weight': [7], 
#   'n_estimators': [700], 
#   'subsample': [0.7], 
#   'gamma': [5, 7],
#   'reg_alpha': [0.5, 0.7], 
#   'reg_lambda': [1, 3]
#   }




# COMMAND ----------

xgb1 = XGBRegressor()
print(parameters)
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose=2)

xgb_grid.fit(X_train,
         Y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

# COMMAND ----------

xgb1.get_params()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # 3. Bayesian optimization

# COMMAND ----------

space= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'n_estimators': hp.quniform('n_estimators', 5, 35, 1),
    'num_leaves': hp.quniform('num_leaves', 5, 50, 1),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

# COMMAND ----------

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1, 9),
        'reg_alpha' : hp.quniform('reg_alpha', 40, 180, 1),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 34
    }

space = {
  'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
  'learning_rate': hp.uniform('learning_rate', 0.5, 1), 
  'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
  'min_child_weight' :  scope.int(hp.quniform('min_child_weight', 0, 10, 1)),
  'n_estimators' :  scope.int(hp.quniform('n_estimators', 300, 1000, 100)),
  'subsample': hp.uniform('subsample', 0.6, 1), 

  'gamma': hp.uniform ('gamma', 1, 9),
  'reg_alpha' :  scope.int(hp.quniform('reg_alpha', 40, 180, 1)),
  'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
}

space = {
  'colsample_bytree' : hp.uniform('colsample_bytree', 0.2, 1),
    'learning_rate': hp.uniform('learning_rate', 0, 1),
  'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
  'n_estimators' :  scope.int(hp.quniform('n_estimators', 300, 1000, 100)),
  'subsample': hp.uniform('subsample', 0.6, 1), 
    'gamma': hp.loguniform('gamma', -10, 10),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'random_state': 34
}


space = {
  'colsample_bytree' : hp.uniform('colsample_bytree', 0.1, 1),
  'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
  'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 20),
  'n_estimators' :  scope.int(hp.quniform('n_estimators', 300, 2000, 100)),
  'subsample': hp.uniform('subsample', 0.2, 1), 
  'gamma': hp.loguniform('gamma', -1, 10),
  'reg_alpha': hp.loguniform('reg_alpha', -1, 10),
  'reg_lambda': hp.loguniform('reg_lambda', -1, 10),
  'random_state': 34
}




# COMMAND ----------

def XGB_hyperparameter_tuning(space):
    print(space)
    model = XGBRegressor()
    model.set_params(**space)
    evaluation = [( X_train, Y_train), ( X_test, Y_test)]
    
    model.fit(X_train, Y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10, verbose=False)

    pred = model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(Y_test, pred))
    print ("RMSE SCORE:", rmse)

    return {'loss':rmse, 'status': STATUS_OK, 'model': model}

# COMMAND ----------

trials = Trials()
best = fmin(fn=XGB_hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print ('\n BEST: ', best)

# COMMAND ----------

scores_df = pd.DataFrame()
for result in trials.trials:
  loss = pd.DataFrame([result['result']['loss']], columns=['rmse'])
  params = pd.DataFrame([result['misc']['vals']]).applymap(lambda x: float(x[0]))
  scores_df = pd.concat([scores_df, pd.concat([params, loss], axis=1)])

try:
  existing_params = pd.read_csv('/dbfs'+data_handler.file_system_path + f'/ML_Input/episodes/hyper_params-{pollutant}_2019_2020.csv')
  new_params = pd.concat([existing_params, scores_df])
  new_params.to_csv('/dbfs'+data_handler.file_system_path + f'/ML_Input/episodes/hyper_params-{pollutant}_2019_2020.csv', index=False) 

except FileNotFoundError:
  scores_df.to_csv('/dbfs'+data_handler.file_system_path + f'/ML_Input/episodes/hyper_params-{pollutant}_2019_2020.csv', index=False) 



# COMMAND ----------

existing_params = pd.read_csv('/dbfs'+data_handler.file_system_path + f'/ML_Input/episodes/hyper_params-{pollutant}_2019_2020.csv')
display(existing_params)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

X = existing_params[[col for col in existing_params.columns if col != 'rmse']]
y = existing_params[[col for col in existing_params.columns if col == 'rmse']]
# display(X)


# define the model
model = LinearRegression()
model.fit(X, y)
importance = pd.DataFrame(model.coef_[0])
features = pd.DataFrame([col for col in existing_params.columns if col != 'rmse'], columns=['Features'])
importance = pd.concat([importance, features], axis=1)


importance.set_index('Features').plot(kind='bar', legend=False, title='Params feature importance', figsize=(15,10));

# COMMAND ----------


