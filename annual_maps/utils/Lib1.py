# Databricks notebook source
import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
      :cols_to_select: str = Columns' name we are willing to query

    Returns
    -------
      :temp_df_filtered: str = Dataframe stored in the target parquet file
    """
    temp_df = spark.read.parquet(self.file_system_path + path_to_parket)
    temp_df_filtered = temp_df.select(features)
    
    return temp_df_filtered
    
  
  def parquet_storer(self, data:pd.DataFrame, path_to_store:str, compression:str='snappy', index:bool=False):
    """Stores dataframe into parquet
    Params
    -------
      :data: str = Dataframe containing data we are willing to store
      :path_to_store: str = path to store our df
      :compression: str = type of compression we are willing to use
      :index: bool = willing to set new index or not
    """
    
    data.to_parquet('/dbfs'+self.file_system_path+output_path, compression=compression, index=index)
    
    return None
  

  def tiff_reader(self, path_to_tiff:str):
    """Connects to the datasources and loads the desired Geotiff file
    Params
    ------
      :path_to_tiff: str = path to the stored tiff data

    Returns
    -------
      :dataset: str = dataset with the pollutant's data
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
    self.train_path_struct, self.validation_path_struct, self.prediction_path_struct, self.output_path_struct = self.data_handler.config.select_ml_paths()

  
  def build_path(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str): 
    """Builds path where we are storing our datafile by following the structure determined at init
    """
    if train_start_year:
      train_path:str = self.train_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, target, self.data_handler.pollutant, train_start_year, train_end_year)
      validation_path:str = self.validation_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.data_handler.pollutant, predval_start_year, predval_end_year)

      return train_path, validation_path
    
    else:
      prediction_path:str = self.prediction_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.data_handler.pollutant, predval_start_year, predval_end_year)
      output_path:str = self.output_path_struct.format(self.data_handler.pollutant, predval_start_year, predval_end_year, date_of_input)
      
      return prediction_path, output_path

  
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
      prediction_path, output_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, None, None)
      
      selected_cols_pollutants = [col for col in selected_cols_pollutants if not (col.startswith('eRep') | col.startswith('e1b'))]
      prediction_data = self.data_handler.parquet_reader(prediction_path, features=selected_cols_pollutants)
      
      return prediction_data, output_path

# COMMAND ----------

# Specific functions to manipulate data with PREPROCESSING purposes

class PreProcessDataHandler(DataHandler):
  
  def __init__(self, pollutant:str):

    self.data_handler = DataHandler(pollutant)
    self.preprocess_path_struct = self.data_handler.config.select_preprocess_paths()
    
  
  def build_path(self, date:str):
    year = date.year
    month = date.month if len(str(date.month))>1 else '0'+str(date.month)
    day = date.day if len(str(date.day))>1 else '0'+str(date.day)

    path_to_tiff = self.preprocess_path_struct.format(self.data_handler.pollutant, year, month, self.data_handler.pollutant, year, month, day)

    return path_to_tiff


  def data_collector(self, date):
    
    path_to_tiff = self.build_path(date)
    raster, rasterXsize, transform, no_data = self.data_handler.tiff_reader(path_to_tiff)

    return raster, rasterXsize, transform, no_data
    
    

# COMMAND ----------

class PrepareTrainValPredDfs:
  """Necesitamos generar un dataframe final que junte X días para el train, Y días para el val y Z días para el predict 
  """
  pass
