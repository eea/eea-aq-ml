# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# DEFAULT_PREPROCESS_INPUT_DATA_FLAG = False
DEFAULT_START_DATE = 2020
DEFAULT_END_DATE = 2020
DEFAULT_TRAIN_FLAG = False
# DEFAULT_MODEL_FINE_TUNE_FLAG = False 



# dbutils.widgets.text(name='PreprocessInputDataFlag', defaultValue=str(DEFAULT_PREPROCESS_INPUT_DATA_FLAG), label='Preprocess input data flag')
dbutils.widgets.text(name='TrainingStartDateLoad', defaultValue=str(DEFAULT_START_DATE), label='Starting date')
dbutils.widgets.text(name='TrainingEndDateLoad', defaultValue=str(DEFAULT_END_DATE), label='End date')
dbutils.widgets.text(name='TrainFlag', defaultValue=str(DEFAULT_TRAIN_FLAG), label='Train flag')
dbutils.widgets.text(name='ModelFineTunningFlag', defaultValue=str(DEFAULT_MODEL_FINE_TUNE_FLAG), label='Model fine tunning flag')

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

import sys
import logging

# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))


# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)


# Adding input variables from widgets
# preprocess_input_data:bool = dbutils.widgets.get('PreprocessInputDataFlag')
start_date_load:str = dbutils.widgets.get('StartDateLoad')
end_date_load:str = dbutils.widgets.get('EndDateLoad')
train_flag:bool = dbutils.widgets.get('TrainFlag')
model_fine_tune:bool = dbutils.widgets.get('ModelFineTunningFlag') if train_flag == True else False



logging.info(f'Your chosen parameters: PREPROCESS_INPUT_DATA: "{preprocess_input_data}", START_DATE_LOAD: "{start_date_load}", END_DATE_LOAD:"{end_date_load}", TRAIN_FLAG: "{train_flag}", MODEL_FINE_TUNE: "{model_fine_tune}"')





# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Main execution

# COMMAND ----------

import glob
import os

spark.conf.set("spark.databricks.io.cache.enabled", "true")
serialize_settings = \
{
  "OutputFolder": "ML_Input"
}


# Timeout in seconds of execution of child Notebooks.
CHILD_TIMEOUT = 60*60*24

# COMMAND ----------



# COMMAND ----------

# Duda 0: OOP?? Scripting??
# Duda: Cargar datos: Load data VS FeatureSelection    -->  (como se seleccionan las fechas???)
#  Load data (opcional): actualizar datos hasta fecha mas reciente
#  FeatureSelection: carga los datos para ML
#  TrainData (opcional): Reentrenar modelo + hyperparameter tunning (opcional)
#  PredictData: Predicciones
#  PlotMaps


#  DataPipeline class used in load data
#  Duda 1: revisar variables con las que se monta el path a los MLINPUT en featureselection. Cuando se generan los paths a los pollutants, "example-0X" que es?? por que se utiliza?
# Duda 2: ## El train/predict se divide por fechas obligatoriamente???? Podemos hacerlo por % del DF??? Automatizar fechas con comando update???
# Duda2: Cireteria used for selected features? VIX and corr????? Can we automate it? or it is not based on scores obtained?

