# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# dbutils.widgets.removeAll()


# COMMAND ----------

# Set default parameters for input widgets
DEFAULT_TRAIN_START = '2016'
DEFAULT_TRAIN_END = '2019'
DEFAULT_PREDVAL_START = '2020'
DEFAULT_PREDVAL_END = '2020'
DEFAULT_VERSION = 'v0'
DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_PREPROCESS_DATA_LIST = ['YES', 'NO']
DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b']
DEFAULT_STORE_PREDICTIONS_LIST = ['YES', 'NO']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                       # We need this to load the pretrained model
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')                             # We need this to load the pretrained model
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.dropdown('PreprocessData', 'NO', DEFAULT_PREPROCESS_DATA_LIST, label='Preprocess Data')
dbutils.widgets.multiselect('Pollutants', 'PM10', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                          
dbutils.widgets.dropdown('StorePredictions', 'NO', DEFAULT_STORE_PREDICTIONS_LIST, label='Store Predictions')  


# https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html
# https://docs.databricks.com/_extras/notebooks/source/xgboost-pyspark.html


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

import time
import logging
import traceback


# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+'/mnt/dis2datalake_airquality-predictions', logging_mode='w')

# Preparing logs configuration
logging.basicConfig(
    format = '%(asctime)s %(levelname)-8s %(message)s', 
    level  = logging.INFO,
)
logging.getLogger("py4j").setLevel(logging.ERROR)


# Timeout in seconds of execution of child Notebooks.
CHILD_TIMEOUT = 60*60*24

# Adding input variables from widgets
preprocess_data:bool = True if dbutils.widgets.get('PreprocessData') == 'YES' else False
train_start_year:str = dbutils.widgets.get('TrainStartDate')
train_end_year:str = dbutils.widgets.get('TrainEndDate')
predval_start_year:str = dbutils.widgets.get('PredValStartDate')
predval_end_year:str = dbutils.widgets.get('PredValEndDate')
pollutants:list = dbutils.widgets.get('Pollutants')#.split(',')
trainset:list = dbutils.widgets.get('Trainset')#.split(',')
date_of_input:str = dbutils.widgets.get('DateOfInput')
version:str = dbutils.widgets.get('Version')
store_predictions:bool = dbutils.widgets.get('StorePredictions') 

if preprocess_data: 
  notebooks_list = ['DataPreprocessing', 'PredictData', 'PlotMaps'] 
else:
  notebooks_list = ['PredictData', 'PlotMaps']

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 

logging.info(f'Your chosen parameters to execute the PREDICT PIPELINE: preprocess_data:"{preprocess_data}", train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", store_predictions:"{store_predictions}", store_predictions:"{store_predictions}", notebooks_list:{notebooks_list}')


# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Main execution

# COMMAND ----------

logging.info('Starting execution of child Notebooks...')
try:
  for notebook_name in notebooks_list:
    start_time = time.time()
    logging.info('Executing Notebook "{}"...'.format(notebook_name))

    # Run current child Notebook.
    result = dbutils.notebook.run('pipeline/'+notebook_name, CHILD_TIMEOUT, {'TrainStartDate': str(train_start_year), 'TrainEndDate': str(train_end_year), 'PredValStartDate': str(predval_start_year),'PredValEndDate': str(predval_end_year),'Pollutants': str(pollutants),'Trainset': str(trainset),'DateOfInput': str(date_of_input),'Version': str(version),'StorePredictions': str(store_predictions)})
    
    # if result.get('logging_file_name'):
    #     notebook_mgr.digest_logging_file(result.get('logging_file_name'))
    # if result['status']!='SUCCESS': 
    #     raise Exception(result['message'])
            
    elapsed_time = time.time() - start_time
    elapsed_time = datetime.datetime.utcfromtimestamp(elapsed_time)
    elapsed_text = str(elapsed_time)
    elapsed_text = elapsed_text[elapsed_text.index(' ') + 1:]
    logging.info('Notebook successfully processed! Elapsed=[{0}]'.format(elapsed_text))

        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'TrainStartDate': str(train_start_year), 'TrainEndDate': str(train_end_year), 'PredValStartDate': str(predval_start_year),'PredValEndDate': str(predval_end_year),'Pollutants': str(pollutants),'Trainset': str(trainset),'DateOfInput': str(date_of_input),'Version': str(version),'StorePredictions': str(store_predictions)})


logging.info('Child Notebooks successfully processed!')



# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Finishing Job

# COMMAND ----------

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'TrainStartDate': str(train_start_year), 'TrainEndDate': str(train_end_year), 'PredValStartDate': str(predval_start_year),'PredValEndDate': str(predval_end_year),'Pollutants': str(pollutants),'Trainset': str(trainset),'DateOfInput': str(date_of_input),'Version': str(version),'StorePredictions': str(store_predictions)})

notebook_mgr = None


# COMMAND ----------


