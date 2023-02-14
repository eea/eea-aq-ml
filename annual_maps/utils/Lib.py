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


def header(storage_account_name:str, blob_container_name:str):
    """Mounts the Azure Blob Storage Container as a File System.
    Params
    ------
      :storage_account_name: str = Name for the storage account we are willing to connect
      :blob_container_name: str = Name for the container storing the desired data
      
    Returns
    -------
      :path: str = Path to /mnt datalake 
  """
    
    path = fsutils.mount_azure_container(
    storage_account_name = storage_account_name, 
    container_name = blob_container_name, 
    sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
  )
  
    return path
  
  
  
# Class DataLoader

def parquet_reader(file_system_path:str, path_to_parket:str, cols_to_select:list=['*']):
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
    temp_df_filtered = temp_df.select(cols_to_select)
    
    return temp_df_filtered
  
  
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



# COMMAND ----------



# COMMAND ----------


