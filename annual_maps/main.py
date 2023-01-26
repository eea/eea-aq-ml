# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Import required packages & variables

# COMMAND ----------

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)

logger.addHandler(c_handler)



# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

DEFAULT_PREPROCESS_INPUT_DATA_FLAG = False
DEFAULT_START_DATE = 2020
DEFAULT_END_DATE = 2020
DEFAULT_TRAIN_FLAG = False
DEFAULT_MODEL_FINE_TUNE_FLAG = False 



dbutils.widgets.text(name='PreprocessInputDataFlag', defaultValue=str(DEFAULT_PREPROCESS_INPUT_DATA_FLAG), label='Preprocess input data flag')
dbutils.widgets.text(name='TrainingStartDateLoad', defaultValue=str(DEFAULT_START_DATE), label='Starting date')
dbutils.widgets.text(name='TrainingEndDateLoad', defaultValue=str(DEFAULT_END_DATE), label='End date')
dbutils.widgets.text(name='TrainFlag', defaultValue=str(DEFAULT_TRAIN_BOOL), label='Train flag')
dbutils.widgets.text(name='ModelFineTunningFlag', defaultValue=str(DEFAULT_MODEL_FINE_TUNE_FLAG), label='Model fine tunning flag')


# COMMAND ----------

preprocess_input_data:bool = dbutils.widgets.get('PreprocessInputDataFlag')
start_date_load:str = dbutils.widgets.get('StartDateLoad')
end_date_load:str = dbutils.widgets.get('EndDateLoad')
train_flag:bool = dbutils.widgets.get('TrainFlag')
model_fine_tune:bool = dbutils.widgets.get('ModelFineTunningFlag') if train_flag == True else False



logger.info(f' ##  You are about to {"PREPROCESS" if preprocess_input_data==True else "PULL"} data from {start_date_load} until {end_date_load}, {"TRAIN" if train_flag else "PREDICT"} and {"INCLUDE FINE TUNNING" if model_fine_tune==True else "EXCLUDE FINE TUNNING"} for the model.\n\n#########  INPUT PARAMS  ######### \n  PREPROCESS_INPUT_DATA: {preprocess_input_data}\n  START_DATE_LOAD: {start_date_load}\n  END_DATE_LOAD: {end_date_load}"\n  TRAIN_FLAG: {train_flag}\n  MODEL_FINE_TUNE: {model_fine_tune}\n############################')





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



# COMMAND ----------



# COMMAND ----------

logging.info('Starting main execution...')

if preprocess_input_data:
  if train_flag:                                 # Duda: Esto deberia de ejecutar solo el predict, no????????????? en caso de querer entrenar los pollutants, meternos en el NB y hacerlo desde ahi.
    logger.info('Executing Notebook "LoadData" to preprocess the needed pollutants...')
    preprocessed_data = dbutils.notebook.run('LoadData', CHILD_TIMEOUT, {'TrainingStartYear': start_date_load, 'TrainingEndYear': end_date_load})
    
  else:          ## No estoy seguro que esto funcione. El NB depende de fechas de start/end para train + predict. O si solo pasamos predict, ejecuta solo parte de predict.
    preprocessed_data = dbutils.notebook.run('LoadData', CHILD_TIMEOUT, {'PredictStartYear': start_date_load, 'PredictEndYear': end_date_load})       


else:
  for index in range(0, 4):
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    file_pattern = '/dbfs/mnt/dis2datalake_airquality-predictions/{}/example-{:02d}*/{}/*.parquet'.format(serialize_settings.get('OutputFolder'), index+1, current_date)
    
    print('------------------------------------------------------------------')
    print('Tables of Example {:02d} (Date: {}):'.format(index+1, current_date))
    print('------------------------------------------------------------------')
    
    for file_name in glob.glob(file_pattern, recursive=False):
        temp_df = spark.read.parquet(file_name[5:])
        print('> Table: "{}", Date: {}, Count: {}'.format(os.path.splitext(os.path.basename(file_name))[0], current_date, temp_df.count()))
        display(temp_df)
      
        
print('OK!')


# COMMAND ----------

display(temp_df)

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

