# Databricks notebook source
"""
================================================================================
Notebook showing results in a Folium map. We should only need to modify the widgets for normal executions.

Arguments:
  + date_of_input: date used to build the path where we are storing our input data
  + pollutants: list of pollutants we are willing to forecast 
  + predval_end_year: last date for the interval we are willing to forecast
  + predval_start_year: starting date for the period we are willing to forecast
  + trainset: list of the targets we are willing to predict 

================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157021
Author   : aiborra-ext@tracasa.es

================================================================================
"""

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

DEFAULT_PREDVAL_START = '2020'
DEFAULT_PREDVAL_END = '2020'
DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b']

dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset') 

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

import logging
import pyspark.sql.functions as F

from pyspark.sql.types import LongType

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))

gridid2laea_x_udf = spark.udf.register('gridid2laea_x', CalcGridFunctions.gridid2laea_x, LongType())
gridid2laea_y_udf = spark.udf.register('gridid2laea_y', CalcGridFunctions.gridid2laea_y, LongType())

# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)

predval_start_year:str = dbutils.widgets.get('PredValStartDate')
predval_end_year:str = dbutils.widgets.get('PredValEndDate')
pollutants:list = dbutils.widgets.get('Pollutants').split(',')
trainset:list = dbutils.widgets.get('Trainset').split(',')
date_of_input:str = dbutils.widgets.get('DateOfInput')

logging.info(f'Your chosen parameters to PLOT: predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}"')

# COMMAND ----------

# MAGIC %run "../utils/Lib"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Executing script

# COMMAND ----------

for pollutant in pollutants:   
  
  # In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    # Collecting forecasted data
    data_handler = DataHandler(pollutant)
    predictions_path = data_handler.config.select_ml_paths(path_to_return='output_parquet_path_struct').format(pollutant, predval_start_year, predval_end_year, date_of_input)
    logging.info(f'Collecting data from {predictions_path}')
    predictions_data = data_handler.parquet_reader(path_to_parket=predictions_path)

    # Adding XY location using 'GridNum1km' attribute (For didactical purpose).
    logging.info('Data collected! Adding coordinates X and Y to plot maps...')
    ml_outputs_df_xy = predictions_data \
                                  .withColumnRenamed('x', 'x_old') \
                                  .withColumnRenamed('y', 'y_old') \
                                  .withColumn('x', gridid2laea_x_udf('GridNum1km') + F.lit(500)) \
                                  .withColumn('y', gridid2laea_y_udf('GridNum1km') - F.lit(500))

    # Plot predictions
    my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': ml_outputs_df_xy, 'attributes': [pollutant]})
    display(my_map)

logging.info('Finished plots!')


# COMMAND ----------


