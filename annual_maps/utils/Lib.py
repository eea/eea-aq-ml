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

class CollectData(CollectDataConfig):
  """Paths storing data"""
  def __init__(self, pollutant:str):
    
    config = CollectDataConfig()
    self.storage_account_name, self.blob_container_name, self.sas_key = config.select_container()
    self.train_path_struct, self.validation_path_struct, self.prediction_path_struct = config.select_paths()
    self.selected_cols_pollutants = config.select_cols(pollutant)
    self.pollutant = pollutant.upper()

  def header(self):
    """Mounts the Azure Blob Storage Container as a File System.
    Params
    ------
      :self.storage_account_name: str = Name for the storage account we are willing to connect
      :self.blob_container_name: str = Name for the container storing the desired data
      :self.sas_key: str = API key

    Returns
    -------
      :file_system_path: str = Path to /mnt datalake 
    """

    file_system_path = fsutils.mount_azure_container(
    storage_account_name = self.storage_account_name, 
    container_name = self.blob_container_name, 
    sas_key = self.sas_key
    )

    return file_system_path    
    
  def build_path(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str): 
    """Builds path where we are storing our datafile by following the structure determined at init
    """
    if train_start_year:
      train_path:str = self.train_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, target, self.pollutant, train_start_year, train_end_year)
      validation_path:str = self.validation_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.pollutant, predval_start_year, predval_end_year)

      return train_path, validation_path
    
    else:
      prediction_path:str = self.prediction_path_struct.format()
      
      return prediction_path, _

  def parquet_reader(self, file_system_path:str, path_to_parket:str, features:list=['*']):
    """Connects to the datasources and queries the desired parquet file to return a dataframe
    Params
    ------
      :file_system_path: str = path to /mnt datalake
      :path_to_parket: str = Name of the parquet file storing the desired data
      :cols_to_select: str = Columns' name we are willing to query

    Returns
    -------
      :temp_df_filtered: str = Dataframe stored in the target parquet file
    """
    
    temp_df = spark.read.parquet(file_system_path+path_to_parket)
    temp_df_filtered = temp_df.select(self.selected_cols_pollutants)
    
    return temp_df_filtered
  
  def data_collector(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str, features:list=['*']):
    """Pipeline to execute previous functions so we can collect desired data by calling just one function.
    
    Returns
    -------
      :train_data: str = Dataframe stored in the target parquet file
      :validation_data: str = Dataframe stored in the target parquet file
      
      OR
      :prediction_data: str = Dataframe stored in the target parquet file
    """
      
    file_system_path = self.header()
    if train_start_year:
      train_path, validation_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year)
      
      train_data = self.parquet_reader(file_system_path, train_path, features)
      validation_data = self.parquet_reader(file_system_path, validation_path, features)
      
      return train_data, validation_data
    
    else:
      prediction_path, _ = self.build_path()
      
      prediction_data = self.parquet_reader(file_system_path, prediction_path, features)
      
      return prediction_data


# COMMAND ----------

  
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

# pollutant_train_data, pollutant_validation_data = CollectData('PM10').data_collector(predval_start_year='2020', predval_end_year='2020', date_of_input='20230201', version='v0', target='eRep', train_start_year='2016', train_end_year='2019', features=['selected'])


# COMMAND ----------


