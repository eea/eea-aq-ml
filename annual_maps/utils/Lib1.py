# Databricks notebook source
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from patsy import dmatrix
from pyspark.sql.functions import col

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

# COMMAND ----------

# MAGIC %run "../config/ConfigFile"

# COMMAND ----------

# General functions to manipulate data

class DataHandler(DataHandlerConfig):
  """Class containing all needed functions to collect data"""
  
  def __init__(self, pollutant:str):
    
    self.config = DataHandlerConfig()
    storage_account_name, blob_container_name, sas_key = self.config.select_container()
    self.file_system_path = self.header(storage_account_name, blob_container_name, sas_key)
    self.pollutant = pollutant.upper()

    
  @staticmethod
  def header(storage_account_name, blob_container_name, sas_key):
    """Mounts the Azure Blob Storage Container as a File System.
    Params
    ------
      :storage_account_name: str = Name for the storage account we are willing to connect
      :blob_container_name: str = Name for the container storing the desired data
      :sas_key: str = API key

    Returns
    -------
      :file_system_path: str = Path to /mnt datalake 
    """

    file_system_path = fsutils.mount_azure_container(
    storage_account_name = storage_account_name, 
    container_name = blob_container_name, 
    sas_key = sas_key
    )

    return file_system_path    
    
    
  def parquet_reader(self, path_to_parket:str, features:list=['*']):
    """Connects to the datasources and queries the desired parquet file to return a dataframe
    Params
    ------
      :path_to_parket: str = Name of the parquet file storing the desired data
      :features: str = Columns' name we are willing to query

    Returns
    -------
      :temp_df_filtered: str = Dataframe stored in the target parquet file
    """
    
    temp_df = spark.read.parquet(self.file_system_path+path_to_parket)
    temp_df_filtered = temp_df.select(features)
    
    return temp_df_filtered
    

  def tiff_reader(self, path_to_tiff:str):
    """Connects to the datasources and loads the desired Geotiff file
    Params
    ------
      :path_to_tiff: str = path to the stored tiff data

    Returns
    -------
      :raster: array = values for the data at tiff 
      :rasterXsize: int = size of raster
      :transform: tuple = containing values used for the geotransformation
      :no_data: str = missing data
    """
        
    # Convert tiff to GDAL dataset
    dataset = gdal.Open('/dbfs/' + self.file_system_path + path_to_tiff, gdal.GA_ReadOnly)
    
    # Convert GDAL Dataset into a Pandas Table.
    transform = dataset.GetGeoTransform()
    rasterXsize, rasterYSize = dataset.RasterXSize, dataset.RasterYSize
    band_r    = dataset.GetRasterBand(1)
    no_data   = band_r.GetNoDataValue()
    band_r    = None
    raster    = dataset.ReadAsArray(0, 0, rasterXsize, rasterYSize)
    dataset = None
  
    return raster, rasterXsize, transform, no_data   
  
  
  def parquet_storer(self, data:pd.DataFrame, output_path:str, compression:str='snappy', index:bool=False):                       
    """Stores dataframe into parquet
    Params
    -------
      :data: str = Dataframe containing data we are willing to store
      :output_path: str = path to store our df
      :compression: str = type of compression we are willing to use
      :index: bool = willing to set new index or not
    """
    
    data.to_parquet('/dbfs'+self.file_system_path+output_path, compression=compression, index=index)
    
    return None
  
  
  def csv_storer(self, data:pd.DataFrame, output_path:str, compression:str='infer', index:bool=False):
    """Stores dataframe into csv
    Params
    -------
      :data: str = Dataframe containing data we are willing to store
      :output_path: str = path to store our df
      :compression: str = type of compression we are willing to use
      :index: bool = willing to set new index or not
    """
    
    data.to_csv('/dbfs'+self.file_system_path+output_path, compression=compression, index=index)
    
    return None
  
  
  @staticmethod
  def find_duplicates(df1:pd.DataFrame, df2:pd.DataFrame, cols_to_compare:list=['*']):
    """Find duplicated values among two different dataframes
    Params
    ------
      :df1: pd.DataFrame = Dataframe you are willing to compare against df2
      :df2: pd.DataFrame = Dataframe you are willing to compare against df1
      :cols_to_compare: list = Columns you are willing to compare

    Returns
    -------
      :duplicated_rows_df: pd.DataFrame = Duplicated rows
    """

    duplicated_rows_df = df1[cols_to_compare].intersect(df2[cols_to_compare])

    return duplicated_rows_df


# COMMAND ----------

# Specific functions to manipulate data with ML purposes

class MLDataHandler(DataHandler):
  
  
  def __init__(self, pollutant:str):

    self.data_handler = DataHandler(pollutant)
    self.train_path_struct, self.validation_path_struct, self.prediction_path_struct, self.parquet_output_path_struct, self.raster_outputs_path_struct = self.data_handler.config.select_ml_paths() 

  
  def build_path(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str): 
    """Builds path where we are storing our datafile by following the structure determined at init
    """
    if train_start_year:
      train_path:str = self.train_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, target, self.data_handler.pollutant, train_start_year, train_end_year)
      validation_path:str = self.validation_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.data_handler.pollutant, predval_start_year, predval_end_year)

      return train_path, validation_path
    
    else:
      prediction_path:str = self.prediction_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.data_handler.pollutant, predval_start_year, predval_end_year)
      output_parquet_path:str = self.parquet_output_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input)
  
      code_pollutant = {'NO2':'8', 'PM10':'5', 'PM25':'6001', 'O3_SOMO35':'7', 'O3_SOMO10':'7'}
      agg_pollutant = {'NO2':'P1Y', 'PM10':'P1Y', 'PM25':'P1Y', 'O3_SOMO35':'SOMO35', 'O3_SOMO10':'SOMO10'}
      
      raster_output_path:str = self.raster_outputs_path_struct.format(predval_end_year, code_pollutant[self.data_handler.pollutant], agg_pollutant[self.data_handler.pollutant], predval_end_year, code_pollutant[self.data_handler.pollutant], agg_pollutant[self.data_handler.pollutant])                 # predyear, code, agg, predyear, code, agg
    
      return prediction_path, output_parquet_path, raster_output_path

  
  def data_collector(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str, features:list=['*']):
    """Pipeline to execute previous functions so we can collect desired data by calling just one function.
    
    Returns
    -------
      :train_data: str = Dataframe stored in the target parquet file
      :validation_data: str = Dataframe stored in the target parquet file
      
      OR
      :prediction_data: str = Dataframe stored in the target parquet file
      :output_path: str = path to store the dataframe into parquet file
    """
    selected_cols_pollutants = self.data_handler.config.select_cols(self.data_handler.pollutant) if features[0]=='selected' else ['*'] 

    if train_start_year:
      train_path, validation_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year)
      
      train_data = self.data_handler.parquet_reader(train_path, selected_cols_pollutants)
      validation_data = self.data_handler.parquet_reader(validation_path, selected_cols_pollutants)
      
      return train_data, validation_data
    
    else:
      prediction_path, output_parquet_path, raster_output_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, None, None)
      
      selected_cols_pollutants = [col for col in selected_cols_pollutants if not (col.startswith('eRep') | col.startswith('e1b'))]
      prediction_data = self.data_handler.parquet_reader(prediction_path, features=selected_cols_pollutants)
      
      return prediction_data, output_parquet_path, raster_output_path

# COMMAND ----------

# Specific functions to manipulate data with PREPROCESSING purposes

class PreProcessDataHandler(DataHandler):
  
  def __init__(self, pollutant:str):
    self.data_handler = DataHandler(pollutant)
    self.preprocess_input_data_path, self.preprocess_output_data_path = self.data_handler.config.select_preprocess_paths()
    
  
  def build_path(self, date:str):
    year = date.year
    month = date.month if len(str(date.month))>1 else '0'+str(date.month)
    day = date.day if len(str(date.day))>1 else '0'+str(date.day)

    path_to_tiff = self.preprocess_input_data_path.format(self.data_handler.pollutant, year, month, self.data_handler.pollutant, year, month, day)
    path_to_csv = self.preprocess_output_data_path.format(self.data_handler.pollutant, year, month, self.data_handler.pollutant, year, month, day)

    return path_to_tiff, path_to_csv


  def data_collector(self, date):
    
    path_to_tiff, path_to_csv = self.build_path(date)
    raster, rasterXsize, transform, no_data = self.data_handler.tiff_reader(path_to_tiff)

    return raster, rasterXsize, transform, no_data, path_to_csv
    
    

# COMMAND ----------

class PrepareTrainValPredDfs:
  """Necesitamos generar un dataframe final que junte X días para el train, Y días para el val y Z días para el predict 
  """
  pass

# COMMAND ----------

class MLWorker(MLModelsConfig):
  
  def __init__(self, pollutant, type_of_params):
    self.ml_models_config = MLModelsConfig(pollutant, type_of_params)
    self.model_to_train, self.ml_params = self.ml_models_config.prepare_model()

  @staticmethod
  def split_data(df: pd.DataFrame, train_size:float=0.7, label:list=None):
    """Splits training dataframe into training and test dataframes to train and validate ML models
    Params
    ------
      :df: pd.DataFrame = data we are willing to use to train de model
      :train_size: float = percentage of the dataframe we are willing to use to train
      :label: list = contains the value/s willing to predict
      
    Returns
    -------
      :X_train: str = data the model will use to train 
      :X_test: str = unseen data the model will use to make predictions
      :Y_train: str = values the model will use with the training set to train its predictions
      :Y_test: str = unseen values the model will try to predict
  """

    # Separating values to be predicted from the data used to train
    df_x = df[[col for col in df.columns if col not in label]]
    df_y = df.select(label)

    # Splitting the dataframe
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_x, df_y, train_size=train_size, random_state=42)                        
  
    return X_train, X_test, Y_train, Y_test
  
  @staticmethod   
  def train_load_ml_model(model_name:str = None, X_train_data:pd.DataFrame=None, Y_train_data:pd.DataFrame=None):
    """Trains/loads (from azure experiments) a ML model.
    Params
    ------
      :train_model_flag: bool = Flag to execute/skip training. If False, only predictions will take place (useful when validation)
      :model: [Object, str] = If we pass a string indicating the name of a model it will look for it at our azure experiments otherwise it will execute the input model
      :X_train_data: pd.DataFrame = data we are willing to use to train de model
      :Y_train_data: pd.DataFrame = label we are willing to use to predict on the training set
      
    Returns
    -------     
      :ml_model: float = score our model obtained
  """
    
    if isinstance(model_name, str):
      client = mlflow.tracking.MlflowClient()
      latest_version = client.get_latest_versions(model_name)
      print(f'Loading and executing stored model {model_name} version {latest_version}...')
      ml_model = mlflow.pyfunc.load_model(latest_version[0].source)
  
    else:
#       print(f'Training model {str(self.ml_models_config.model_str)} with {self.ml_models_config.type_of_params} params: {self.ml_params}')
      ml_model = model_name.fit(X_train_data, Y_train_data)

    return ml_model
    
  @staticmethod   
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
  
  
  def ml_executor(self):
    """Add every above functions to generate a final pipeline where split data, training, validating, evaluation"""
    pass
