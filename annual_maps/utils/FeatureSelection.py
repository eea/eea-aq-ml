# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Adding Notebook Input widgets

# COMMAND ----------

# dbutils.widgets.removeAll()

# Set default parameters for input widgets
DEFAULT_TRAIN_START = '2020'
DEFAULT_TRAIN_END = '2020'
DEFAULT_PREDVAL_START = '2020'
DEFAULT_PREDVAL_END = '2020'
DEFAULT_VERSION = 'v0'
DEFAULT_DATE_OF_INPUT = '20230201'

DEFAULT_POLLUTANTS_LIST = ['PM10', 'PM25', 'O3', 'O3_SOMO10', 'O3_SOMO35', 'NO2']
DEFAULT_TRAINSET_LIST = [ 'eRep', 'e1b']
DEFAULT_FEATURES_LIST = ['*', 'selected']
DEFAULT_TYPE_DATA_LIST = ['training', 'validation', 'prediction']

# Set widgets for notebook
dbutils.widgets.text(name='TrainStartDate', defaultValue=str(DEFAULT_TRAIN_START), label='Train Start Year')                  
dbutils.widgets.text(name='TrainEndDate', defaultValue=str(DEFAULT_TRAIN_END), label='Train End Year')
dbutils.widgets.text(name='PredValStartDate', defaultValue=str(DEFAULT_PREDVAL_START), label='Pred-Val Start Year')
dbutils.widgets.text(name='PredValEndDate', defaultValue=str(DEFAULT_PREDVAL_END), label='Pred-Val End Year')
dbutils.widgets.text(name='Version', defaultValue=str(DEFAULT_VERSION), label='Version')
dbutils.widgets.text(name='DateOfInput', defaultValue=str(DEFAULT_DATE_OF_INPUT), label='Date of Input')                            # ? Check the db every time to get the dateofinput?  # Idea generate a droprdown widget + listdir from db

dbutils.widgets.multiselect('Pollutants', 'NO2', DEFAULT_POLLUTANTS_LIST, label='Pollutants')
dbutils.widgets.multiselect('Trainset', "eRep", DEFAULT_TRAINSET_LIST, label='Trainset')                         
dbutils.widgets.dropdown('Features', '*', DEFAULT_FEATURES_LIST, label='Features')  
dbutils.widgets.dropdown('TypeOfData', 'training', DEFAULT_TYPE_DATA_LIST, label='Type of Data')



# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Import required packages & variables

# COMMAND ----------

# MAGIC %run "../utils/Lib"

# COMMAND ----------

# MAGIC %run "../config/ConfigFile"

# COMMAND ----------

import sys
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
type_of_data: str = dbutils.widgets.get('TypeOfData')


logging.info(f'Your chosen parameters: train_start_year: "{train_start_year}", train_end_year: "{train_end_year}", predval_start_year: "{predval_start_year}", predval_end_year: "{predval_end_year}", pollutants: {pollutants}, trainset: {trainset}, date_of_input: "{date_of_input}", version: "{version}", features: {features}, type_of_data: "{type_of_data}"')

if len(trainset)>1: logging.warning(f'You have chosen more than 1 values for Trainset: {trainset}')
if (train_end_year < train_start_year) or (predval_end_year < predval_start_year): raise Exception('End dates cannot be earlier than starting dates. Double check!') 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Loading, processing and plotting data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation and VIF for features selection

# COMMAND ----------

def get_corr_vif(data: pd.DataFrame, cols_to_exclude:list=[], target_col:str=trainset):
    """Calculate correlation and VIF of the features against a target variable
  Params
  ------
    :data: pd.DataFrame = Dataframe containing the features you are willing to analyse
    :cols_to_exclude: list = Cols you are willing to exclude from the analysis
    :target_col: str = Column you are willing to focus on

  Returns
  -------
    :corr_vif_df: pd.DataFrame = Feature, correlation and VIF
  """
    
    # Filtering cols we cannot include into the vif/corr analysis
    filtered_data = data[[col for col in data.columns if col not in cols_to_exclude]].toPandas()

    # Setting target variable we want to get the correlation
    target_variable = [col for col in data.columns if col.startswith(target_col)][0]

    # Calculating correlation
    corr_features = filtered_data.corrwith(data.toPandas()[target_variable]).to_frame().reset_index().rename(columns={'index':'Feature', 0:'Corr'})

    # Calculating VIF
    vif_features = pd.DataFrame([variance_inflation_factor(filtered_data.values, i) for i in range(filtered_data.shape[1])],  columns=['VIF'])

    # Joining correlation and VIF dfs
    corr_vif_df = pd.concat([corr_features, vif_features], axis=1)

    return corr_vif_df


def plot_corr_vif (corr_vif_features: pd.DataFrame, pollutant:str, target:str = 'eRep', chunk_size_to_plot: int = 50):
  """Plot charts showing correlation and VIF
  Params
  ------
    :corr_vif_features: pd.DataFrame = Dataframe containing the correlation and VIF values for the selected features
    :pollutant: str = Pollutant being analysed
    :chunk_size_to_plot: int = Number of features plot at each chart

  """
    
  # Seating features as index so they appear at our x-axis
  corr_vif_features = corr_vif_features.set_index('Feature').sort_index()

  # Forcing to iterate every feature and limiting plots per chart so we can properly view results
  i=1
  while i*chunk_size_to_plot<=len(corr_vif_features)+chunk_size_to_plot:
    fig, ax1 = plt.subplots(figsize=(50, 10))
    ax2 = ax1.twinx()

    ax1.bar(corr_vif_features.index[(i-1)*chunk_size_to_plot:i*chunk_size_to_plot], height=corr_vif_features['Corr'].iloc[(i-1)*chunk_size_to_plot:i*chunk_size_to_plot], color='y')
    ax2.plot(corr_vif_features['VIF'].iloc[(i-1)*chunk_size_to_plot:i*chunk_size_to_plot])

    ax1.set_ylabel('Corr', fontsize=20, color='y')
    ax1.tick_params(axis='y', colors='y', labelsize=20)

    ax2.set_ylabel('VIF', fontsize=20, color='b')
    ax2.tick_params(axis='y', colors='b', labelsize=20)

    ax1.set_xticklabels(corr_vif_features['Corr'].index[(i-1)*chunk_size_to_plot:i*chunk_size_to_plot], rotation=90, ha='right', fontsize=20)
    plt.title(f'Corr-VIF {pollutant.upper()} for {target}', fontsize=35)
    plt.show()
    i+=1
    
  return None


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Execute pipeline 

# COMMAND ----------

for pollutant in pollutants:   
  data_handler = DataHandler(pollutant)
  
# In case we have different target variables i.e.: eRep and e1b.
  for target in trainset:
    logging.info(f'Processing pollutant: {pollutant} target {target}.')
    label = [target + '_' + pollutant.upper()][0]
    # Collecting data
    pollutant_train_data, pollutant_validation_data = data_handler.data_collector(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year, features)
    logging.info('Data pollutant collected!')


    try:
      # Excluding cols from corr-VIF analysis
      cols_to_exclude:list = ['GridNum1km','Year','AreaHa', target + '_' + pollutant.upper()] 
      
      # Choose input data from training/validation/prediction
      if type_of_data == 'training':
        # Cleaning data
        features_data = pollutant_train_data.filter((pollutant_train_data['Year'] >= train_start_year) & (pollutant_train_data['Year'] <= train_end_year) & (pollutant_train_data[label] > 0)).cache()
        
      if type_of_data == 'validation':
        # Cleaning data
        features_data = pollutant_validation_data.filter((pollutant_validation_data['Year'] >= predval_start_year) & (pollutant_validation_data['Year'] <= predval_end_year) & (pollutant_validation_data[label] > 0)).cache()

      logging.info(f'Calculating Correlation and VIF for {pollutant.upper()}, target {target} based on {type_of_data.upper()} dataset...')
      corr_vif_features = get_corr_vif(features_data, cols_to_exclude=cols_to_exclude, target_col=target)

      logging.info('Correlation and VIF calculated succesfully! Plotting charts...')
      plot_corr_vif(corr_vif_features, pollutant, target)
      
      features_data.unpersist()
    except:
      if not features_data: logging.info(f'Something went wrong! Please, revisit input params...')

      
      
logging.info(f'Finished!')




# COMMAND ----------


