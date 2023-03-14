# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

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

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ?????????????????? do we need to check the db every time to get the dateofinput? otherwise, how could we know the path to the data we need?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.dropdown('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', 'selected', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('TypeOfParams', 'optimized', DEFAULT_PARAMS_LIST, label='Type of params')  
dbutils.widgets.dropdown('StoreModel', 'NO', DEFAULT_STORE_MODEL_LIST, label='Store Trained Model')  
dbutils.widgets.dropdown('TrainModel', 'NO', DEFAULT_TRAIN_MODEL_LIST, label='Train Model')  



# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

# MAGIC %run "../utils/Lib"

# COMMAND ----------

# MAGIC %run "../config/ConfigFile"

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


logging.info(f'Your chosen parameters: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_params: "{type_of_params}", store_model: {store_model}, train_model: "{train_model}"')

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
  
  
def train_predict_ml_model(train_model_flag:bool, store_model:bool, model, X_train_data:pd.DataFrame=None, Y_train_data:pd.DataFrame=None, X_test_data:pd.DataFrame=None):
    """Trains a ML model and/or predicts data. It will store/load a ML model to/from azure experiments or execute the input model.
    Params
    ------
      :train_model_flag: bool = Flag to execute/skip training. If False, only predictions will take place (useful when validation)
      :model: [Object, str] = If we pass a string indicating the name of a model it will look for it at our azure experiments otherwise it will execute the input model
      :X_train_data: pd.DataFrame = data we are willing to use to train de model
      :Y_train_data: pd.DataFrame = label we are willing to use to predict on the training set
      :X_test_data: pd.DataFrame = data we are willing to use to make predictions
    Returns
    -------
      :predictions: pd.DataFrame = predictions performed by our model and its actual value
      :ml_model: float = score our model obtained
  """
    
    if train_model_flag:
      print('Training model...')
      with mlflow.start_run():
        mlflow.autolog()
        # Training model with the input data      
        ml_model = model.fit(X_train_data, Y_train_data)
        if store_model:
          run_id = mlflow.active_run().info.run_id
          print('Registering model...')
          # The default path where the MLflow autologging function stores the model
          artifact_path = "model"
          model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
          model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
  
    else:
      if isinstance(model, str):
        print('Loading and executing stored model...')
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name)
        print('Loading latest version of your pretrained models: ', latest_version)
        ml_model = mlflow.pyfunc.load_model(latest_version[0].source)
        
      else:
        print('Executing trained model...')
        ml_model = model
        

    print('Performing predictions...')
    predictions = ml_model.predict(X_test_data)

    return ml_model, predictions

  

def evaluate_model(ml_model, predictions:pd.DataFrame, y_test_data:pd.DataFrame):
    """It will plot some plots showing performance of the model and feature importances (fscore).
    Params
    ------
      :ml_model: Object = ML model we are willing to evaluate
      :predictions: pd.DataFrame = obtained predictions from our model
      :y_test_data: pd.DataFrame = label we are willing to predict and will use to check performance of the model      
    Returns
    -------
      :results: pd.DataFrame = predictions performed by our model and its actual value
      :rmse: float = score our model obtained
  """

    # Evaluating model
    rmse = np.sqrt(mean_squared_error(y_test_data, predictions))
    mape = mean_absolute_percentage_error(y_test_data, predictions)
    try:
      # Finding pollutant and target names in the y_test dataset columns
      label = [col.split('_') for col in y_test_data.columns][0]
      pollutant =  label[1] if len(label)>1 else label[0]
      target =  label[0] 
    except:
      print('Pollutant and target names could not be found')

    print(f"\n{pollutant}-{target} RMSE : {rmse}\n")
    print(f"\n{pollutant}-{target} MAPE : {mape}\n")

    results = pd.concat([y_test_data.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    results.columns = ['actual', 'forecasted']

    # Plotting lines for actuals vs forecasts data
    fig = px.line(results)
    fig.show()

    diff_results = results.actual - results.forecasted
    diff_results.hist()
    plt.title('ML predict Errors distribution')

    try:
      # Feature importance based on fscore (number of times a variable is selected for splitting, weighted by the squared improvement to the model as a result of each split, and averaged over all trees)
      fig, ax = plt.subplots(figsize=(12,12))
      plot_importance(ml_model, ax=ax)
      importance_scores = ml_model.get_booster().get_fscore()
      plt.title(f'Feature importance (fscore) for target {target} & pollutant {pollutant}')
      plt.show();
    except:
      print('Feature importance could not be calculated!')

    return results, rmse, mape, importance_scores


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Execute training

# COMMAND ----------

# Set input params for header
collect_data = CollectData()
file_system_path = header(collect_data.storage_account_name, collect_data.blob_container_name)

cols_to_compare_duplicates = ['GridNum1km','Year']

selected_cols_pollutants = ColsPollutants()
# train_size = 0.99                                         # Are we sure we want to split dataset by size? I think the split should follow a dates schema as we are working with cyclic and seasonal time series. We might be biasing the model if we do not provide whole seasons when we train the model 

for pollutant in pollutants:      
  # Selecting desired cols (*:all vs selected: TrainConfig file)
  pollutant_cols =  eval('selected_cols_pollutants.'+pollutant.lower()) if features[0] == 'selected' else features 
  # In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]
    
    if train_model:
      # Collecting and cleaning data
      path_to_training_parquet:str = f'/ML_Input/data-{pollutant}_{predval_start_year}-{predval_end_year}/{date_of_input}_{version}/training_input_{target}_{pollutant}_{train_start_year}-{train_end_year}.parquet'  
      pollutant_train_data = parquet_reader(file_system_path, path_to_parket=path_to_training_parquet, cols_to_select=pollutant_cols)
      pollutant_train_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0))
      
      path_to_val_parquet:str = f'/ML_Input/data-{pollutant}_{predval_start_year}-{predval_end_year}/{date_of_input}_{version}/validation_input_{pollutant}_{predval_start_year}-{predval_end_year}.parquet' 
      pollutant_validation_data = parquet_reader(file_system_path, path_to_parket=path_to_val_parquet, cols_to_select=pollutant_cols)
      pollutant_validation_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0))
      logging.info('Data pollutant collected! Checking for duplicated data among your training and validation datasets...')

      # Making sure we do not have duplicates among train and val datasets
      duplicated_rows = find_duplicates(df1=pollutant_train_data, df2=pollutant_validation_data, cols_to_compare=cols_to_compare_duplicates)
      logging.warning(f'There are duplicates in your training and validation set: {duplicated_rows}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')

      # Preparing data for training/validating/predicting
      df_train = pollutant_train_data.drop('GridNum1km', 'Year','AreaHa').toPandas()                                          
      df_validation = pollutant_validation_data.drop('GridNum1km', 'Year','AreaHa').toPandas()                                         
      X_train , Y_train = df_train[[col for col in df_train.columns if col not in label]], df_train[[label]] 
      validation_X, validation_Y = df_validation[[col for col in df_validation.columns if col not in label]], df_validation[[label]]
      logging.info(f'Data ready! Training & validating model with: \n{X_train.count()} \nPreparing ML model...')

      # Executing selected ML model
      ml_models = MLModels(pollutant)
      model_to_train, ml_params = ml_models.prepare_model()
      logging.info(f'Preparing training model {ml_models.model_str} for pollutant {pollutant} and {type_of_params.upper()} params: {ml_params}') if train_model else logging.info('Loading latest pretrained model to make predictions...')
    
      # Training model + validation
      trained_model, predictions = train_predict_ml_model(train_model_flag=True, store_model=store_model, model=model_to_train, X_train_data=X_train, Y_train_data=Y_train, X_test_data=validation_X)
      results, rmse, mape, importance_scores = evaluate_model(trained_model, predictions, validation_Y)

    else:
      # Prediction inputs data
      path_to_prediction_parquet:str = f''  
      pollutant_prediction_data = parquet_reader(file_system_path, path_to_parket=path_to_prediction_parquet, cols_to_select=pollutant_cols)
      
      # Predicting data using a stored pretrained model
      model_name = f"{pollutant}_{ml_models.model_str.replace('()', '')}_trained_from_{train_start_year}_to_{train_end_year}_{version}"
      _, predictions = train_predict_ml_model(train_model_flag=False, store_model=store_model, model=model_name, X_train_data=None, Y_train_data=None, X_test_data=pollutant_prediction_data)

    
    

# #  REMEMBER TO: Perform training with the whole dataset (training + validation + prediction sets) once we have the final model


logging.info(f'Finished!')

# COMMAND ----------

df_validation[[label]]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 4. Execute training with reference data

# COMMAND ----------


# # Set input params for header
# collect_data = CollectData()
# file_system_path = header(collect_data.storage_account_name, collect_data.blob_container_name)


# path_to_reference_parquet:str = f'/ETC_maps/aq_grids_year_all_with2020.parquet'                    # Are we using this for anything???
# cols_to_compare_duplicates = ['GridNum1km','Year']

# selected_cols_pollutants = ColsPollutants()
# train_size = 0.99                                                                                  # Are we sure we want to split dataset by size? I think the split should follow a dates schema as we are working with cyclic and seasonal time series. We might be biasing the model if we do not provide same seasons when we train the model 

# for pollutant in pollutants:      

#   # Loading reference data for pollutant
#   reference_data = parquet_reader(file_system_path, path_to_reference_parquet, cols_to_select=['GridNum1km', 'Year', pollutant.lower()+'_avg']).withColumnRenamed(pollutant.lower()+'_avg', pollutant)                    # DELETE _avg. Why reference data has avg values for some pollutants and doesn't for others????????  
  
#   # Selecting desired cols (*:all vs selected: TrainConfig file)
#   pollutant_cols =  eval('selected_cols_pollutants.'+pollutant.lower()) if features[0] == 'selected' else features 
  
#   # In case have different target variables i.e.: eRep and e1b.
#   for target in trainset:
#     logging.info(f'Processing pollutant: {pollutant} target {target}.')
  
#     # Reading parquet files
#     path_to_training_parquet:str = f'/ML_Input/data-{pollutant}_{predval_start_year}-{predval_end_year}/{date_of_input}_{version}/training_input_{target}_{pollutant}_{train_start_year}-{train_end_year}.parquet'  
#     pollutant_train_data = parquet_reader(file_system_path, path_to_parket=path_to_training_parquet, cols_to_select=pollutant_cols)

#     path_to_validation_parquet:str = f'/ML_Input/data-{pollutant}_{predval_start_year}-{predval_end_year}/{date_of_input}_{version}/validation_input_{pollutant}_{predval_start_year}-{predval_end_year}.parquet' 
#     pollutant_validation_data = parquet_reader(file_system_path, path_to_parket=path_to_validation_parquet, cols_to_select=pollutant_cols)
#     logging.info('Data pollutant collected!')
    
#     # Checking duplicate rows between training and validation datasets
#     duplicated_rows = find_duplicates(df1=pollutant_train_data, df2=pollutant_validation_data, cols_to_compare=cols_to_compare_duplicates)
#     logging.warning(f'There are duplicates in your training and validation set: {duplicated_rows}') if not duplicated_rows.rdd.isEmpty() else logging.info(f'There are no duplicates!')

#     logging.info('Filtering pollutant data for specified train/pred years and existing reference data...')
#     dfref_train = reference_data.filter((reference_data['Year'] >= train_start_year) & (reference_data['Year'] <= train_end_year) & (reference_data[pollutant] > 0))
#     dfref_val = reference_data.filter((reference_data['Year'] >= predval_start_year) & (reference_data['Year'] <= predval_end_year) & (reference_data[pollutant] > 0))   

#     logging.info('Joining training and validation dataframes with the ETC datasets containing reference data...')
#     dftrain = pollutant_train_data.join(dfref_train, ['GridNum1km', 'Year'], how="inner").drop('GridNum1km', 'Year','AreaHa')
#     dfval = pollutant_validation_data.join(dfref_val, ['GridNum1km', 'Year'], how="inner").drop('GridNum1km', 'Year','AreaHa')

#     logging.info('Splitting data in train/test dataframes for ML training...')
#     label = [pollutant, target + '_' + pollutant.upper()]
#     X_train_data, X_test_data, Y_train_data_ref, Y_test_data_ref = split_data(df=dftrain, train_size=train_size, label=label)

#     # Splitting the target-reference data sets for training, cv and validation into separate datasets
#     y_train_data = Y_train_data_ref[[target + '_' + pollutant]] # target for training error checks
#     y_train_ref = Y_train_data_ref[[pollutant]] # reference for training error checks
    
#     y_test_data = Y_test_data_ref[[target + '_' + pollutant]]  # target for cv error checks
#     y_test_ref = Y_test_data_ref[[pollutant]]  # reference for cv error checks

#     y_val_train = dfval[[col for col in dfval.columns if col not in [target + '_' + pollutant, pollutant]]].toPandas()
#     y_val_data = dfval[[target + '_' + pollutant]].toPandas()  # target for validation error checks
#     y_val_ref = dfval[[pollutant]].toPandas()  # reference for validation error checks

#     logging.info('Data ready for training! Preparing ML model...')
#     # Executing selected ML model
#     ml_models = MLModels(pollutant)
#     model_to_train, ml_params = ml_models.prepare_model()
#     logging.info(f'Training model {ml_models.model_str} for pollutant {pollutant} and {type_of_params} params: {ml_params}')

#     # Execute training
#     train_model = True
#     trained_model, results, rmse = train_predict_ml_model(train_model, model_to_train, X_train_data, y_train_data, X_test_data, y_test_data)

#     logging.info(f'Validating model {str(trained_model)} for pollutant {pollutant}')
#     # Predict validation dataset
#     train_model = False
#     ml_model, results, rmse = train_predict_ml_model(train_model, trained_model, X_train_data=None, y_train_data=None, X_test_data=y_val_train, y_test_data=y_val_data)



    
    
# #  Perform training with the whole dataset
# # We shouldn't be using a validation dataset since we are not using many hyperparameters (low risk of overfitting) and hence we are losing amount of data we should be using for training. In case we still want to use validation dataset, we should not be splitting it in different dbs (not scalable). Store everything into the same database and split it as desired using the split function. Otherwise, how are we updating our db? we move part of the predicting db to the training and update the predicting with the new datapoints?????
# logging.info(f'Finished!')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

display(reference_data)
print(reference_data.count())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Learning curve on data amount vs training and cv errors   ---> Why are they doing this? training on 1% data??

# COMMAND ----------



# COMMAND ----------

# Generating input for learning curve on data amount vs training and cv errors: first view on bias & variance

# Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutant = 'O3_SOMO10' #,'PM10','PM25','O3_SOMO35','O3_SOMO10','NO2'
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0

i = 0.01

# Create the empty initial pandas DataFrame for storing the score changes with growing amount of data
dfscores = pd.DataFrame({'rmse_train': [0], 'rmse_cv': [0], 'rmse_ref': [0], 'data':[0]} )

lr,md,gm,al,la,sub = bestModelParameters(pollutant)
model = XGBRegressor(learning_rate = lr, max_depth = md, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)

while i <= 1.0:

  #load data using loadAQMLData function with sample fraction defined by i
  X_train, X_cv, X_val, y_train_ref, y_train, y_cv, y_cv_ref, y_val, y_val_ref = loadAQMLData(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput,frac=i)
  
  # NN: fitting model to data
#   model.fit(X_train,y_train,
#     epochs=50)
  
  # SGD or XGB: fitting model to data
  model.fit(X_train,y_train)
  
  #getting metrics for training data and reference data (reference y vs y_train)
  rmse_train_ref, rmse_train = validateModel(model,X_train,y_train,y_train_ref)
  #getting metrics for cv data and reference data (reference y vs y_cv)
  rmse_cv_ref, rmse_cv = validateModel(model,X_cv,y_cv,y_cv_ref)
  rmse_ref = (rmse_train_ref + rmse_cv_ref)/2
    
  dfscores = dfscores.append(pd.DataFrame({'rmse_train': [rmse_train], 'rmse_cv': [rmse_cv], 'rmse_ref': [rmse_ref], 'data':[i * 100]} ))
  print('Calculations for : ' + str(i * 100) + ' % of data completed.')
  
  if i == 0.01:
    i += 0.09
  else:
    i += 0.1
    
print('Calculations completed.')

# COMMAND ----------



# ML predictions

# Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutants = ['PM10','PM25','O3_SOMO35','O3_SOMO10','NO2']
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0












for pollutant in pollutants:
  
  print('Pollutant: ' + pollutant)

  # Load data, fit model and prep data for error calculations

  X_train, X_cv, X_val, y_train, y_cv, y_val = loadAQMLDataValTrain(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput)
  lr,md,gm,al,la,sub = bestModelParameters(pollutant)
  model = XGBRegressor(learning_rate = lr, max_depth = md, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)
  model.fit(X_train,y_train)

  X_pred = loadAQMLPredData(path,predval_start_year,predval_end_year,pollutant,dateOfInput)
  predictions = model.predict(X_pred.iloc[:, 3:])

  X_maps = X_pred.iloc[:, :2]
  preds = pd.DataFrame({pollutant: predictions})
  ML_maps = pd.concat([X_maps,preds], axis=1)
  ML_maps_spark = spark.createDataFrame(ML_maps) 

  file_name = '/dbfs' + aq_predictions_path+"/ML_Output/" + pollutant + "_" + predval_start_year + "_" + predval_end_year + "_" + dateOfInput+"_maps.parquet"
  ML_maps_spark.toPandas().to_parquet(file_name, compression='snappy')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


