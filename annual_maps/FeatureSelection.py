# Databricks notebook source


# COMMAND ----------

from patsy import dmatrix
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pyspark.sql.functions import col



# COMMAND ----------

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))

spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")


# NOTE:
# We can't use a simple import statement because of there is a bug in Databricks, 
# It seems that 'dbutils.fs' can't be accessed by local python files.

# # Mount the 'JEDI' Azure Blob Storage Container as a File System.
# jedi_path = fsutils.mount_azure_container(
#   storage_account_name = 'cwsblobstorage01', 
#   container_name = 'cwsblob01', 
#   sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-01-01T19:32:00Z&st=2020-11-28T11:32:00Z&spr=https&sig=6X3z%2Bi1V88p2af7DgyGdGCHeLkK1UxEMeiO8H%2FVPKgo%3D'
# )

# # Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
# aq_predictions_path = fsutils.mount_azure_container(
#   storage_account_name = 'dis2datalake', 
#   container_name = 'airquality-predictions', 
#   sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
# )

# # Mount the 'AQ Climate ML' Azure Blob Storage Container as a File System.
# aq_climate_path = fsutils.mount_azure_container(
#   storage_account_name = 'dis2datalake', 
#   container_name = 'airquality-climate', 
#   sas_key = 'sp=r&st=2021-04-22T12:54:13Z&se=2021-04-22T20:54:13Z&spr=https&sv=2020-02-10&sr=c&sig=0cqTsxw75O4EykAp4LwE2cKFp8MjNlsQXiG%2B2oQcQFc%3D'
# )

# # Mount the 'AQ CAMS ML' Azure Blob Storage Container as a File System.
# aq_cams_path = fsutils.mount_azure_container(
#   storage_account_name = 'dis2datalake', 
#   container_name = 'airquality-cams', 
#   sas_key = 'sp=r&st=2021-05-10T15:06:41Z&se=2021-05-10T23:06:41Z&spr=https&sv=2020-02-10&sr=c&sig=jfWhTCiUPEZ0TQTVE27%2BX6mkNT6xotAXqKpR6f9VvZ0%3D'
# )



# # Initialize a Context Dictionary with useful data.
# context_args = {
#   'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
# }





# COMMAND ----------

# MAGIC %md
# MAGIC ### Check on grid overlaps in val and train/cv data sets

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



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

# COMMAND ----------

def parquet_reader(file_system_path:str, path_to_parket:str, cols_to_select:list):
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
    print('Table Count: {}'.format(temp_df_filtered.count()))
    display(temp_df_filtered)

    return temp_df_filtered


# COMMAND ----------

# Set input params
storage_account_name:str = 'dis2datalake'
blob_container_name: str = 'airquality-predictions'

file_system_path = header(storage_account_name, blob_container_name)


# path_to_parket:str = '/ML_Input/' + example_pollutant 
# cols_to_select:list = 

# # # Execute functions
# parquet_reader(file_system_path, path_to_parket, cols_to_select)

# COMMAND ----------

example_pollutant = 'example-01_PM10_'
predval_start_year = 
predval_end_year = '- YEARRRRR /'
dateOfInput

# COMMAND ----------

# Find duplicates to prevent cheating at ML
def find_duplicates(df1:pd.DataFrame, df2:pd.DataFrame, cols_to_compare:list):
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
  duplicated_rows_df = df1[cols_to_compare].merge(df2[cols_to_compare], how='inner', indicator=False)
  
  return duplicated_rows_df


cols_to_compare = ['GridNum1km','Year']
find_duplicates(dftraining, dfvalidation, cols_to_compare)


# COMMAND ----------



# COMMAND ----------

# Hacer una funcion para validar que no hay duplicados entre el training_df y validation_df








# Check on potential overlaps validation vs training data: PASSED (no overlaps)

# # Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutaÇnt, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutant = 'O3_SOMO10' #,'PM10','PM25','O3_SOMO35','NO2'
dateOfInput = '20220826'

example = 'example'
  
if pollutant == 'NO2': 
    example = 'example-03'
    refcol = 'no2_avg'
elif  pollutant == 'PM10': 
    example = 'example-01'
    refcol = 'pm10_avg'
elif  pollutant == 'PM25': 
    example = 'example-02'  
    refcol = 'pm25_avg'
elif  pollutant == 'O3_SOMO35': 
    example = 'example-04'  
    refcol = 'o3_somo35'
elif  pollutant == 'O3_SOMO10': 
    example = 'example-05'  
    refcol = 'o3_somo10'  
    
trainfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'training_input_' + trainset + '_' + pollutant + '_' + train_start_year + '-' + train_end_year + '.parquet'
valfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'validation_input_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '.parquet'

# Reading ML input files into spark data frames

dftraining = spark.read.parquet(trainfile).select('GridNum1km','Year').withColumnRenamed("GridNum1km","GridNum1km_2").withColumnRenamed("Year","Year_2")
dfvalidation = spark.read.parquet(valfile).select('GridNum1km','Year')

df = dfvalidation.join(dftraining,(dfvalidation.GridNum1km == dftraining.GridNum1km_2) & (dfvalidation.Year == dftraining.Year_2),how="inner")
df.display()

# COMMAND ----------

display(dftraining)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation and VIF for features selection

# COMMAND ----------

# AQ ML data loading for correlation and VIF analysis
  
# # Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2015'
train_end_year = '2019'
predval_start_year = '2020'
predval_end_year = '2020'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutant = 'PM25' #,'PM10','PM25','O3_SOMO35','NO2'
dateOfInput = '20220720'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0









# Esta funcion debería de recibir como input el df sin duplicados en lugar de leer el archivo de nuevo o validar aquí los duplicados
def loadAQMLDataForVIF(path, train_start_year, train_end_year, predval_start_year, predval_end_year, trainset, pollutant, dateOfInput, frac=1.0):

  example = 'example'
  
  if pollutant == 'NO2': 
    example = 'example-03'
    refcol = 'no2_avg'
  elif  pollutant == 'PM10': 
    example = 'example-01'
    refcol = 'pm10_avg'
  elif  pollutant == 'PM25': 
    example = 'example-02'  
    refcol = 'pm25_avg'
  elif  pollutant == 'O3_SOMO35': 
    example = 'example-04'  
    refcol = 'o3_somo35'

  # Compiling path to files
  trainfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'training_input_' + trainset + '_' + pollutant + '_' + train_start_year + '-' + train_end_year + '.parquet'

  # Reading ML input files into spark data frames
  dftraining = spark.read.parquet(trainfile)
    
  return dftraining

dftraining = loadAQMLDataForVIF(path, train_start_year,train_end_year, predval_start_year, predval_end_year, trainset,pollutant, dateOfInput)
print('Pollutant: ', pollutant)
display(dftraining)

# COMMAND ----------

display(dftraining)

# COMMAND ----------



# COMMAND ----------

df = dftraining[[col for col in dftraining.columns if col not in ['GridNum1km','Year','AreaHa']]]
Xvif = df[[col for col in df.columns if col != 'eRep_PM25']]

def calculate_VIF_correlation(df)



# COMMAND ----------

# Correlation & VIF analysis to select variables in case of high variance model

# Getting the input into pandas df
df = dftraining.drop('GridNum1km','Year','AreaHa').toPandas()

# Getting features list for analysis
features = dftraining.drop('eRep_PM25','GridNum1km','Year','AreaHa').toPandas()
Xvif = dmatrix(features, df, return_type='dataframe')

# Correlation between each of the feature and target
corr = df.corrwith(df['eRep_PM25'])

# Generating spark df with features and corresponding correlations
dfcorr = corr.to_frame()
dfcorr = dfcorr.drop('eRep_PM25')
dfcorr['variable'] = Xvif.columns
dfcorr.columns =['Corr', 'Feature']
sparkcorr=spark.createDataFrame(dfcorr) 

# Calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])]
vif['variable'] = Xvif.columns

# Generating spark df with features and corresponding vifs
sparkvif=spark.createDataFrame(vif)

# Joining correlation and vif and display
corrvif = sparkcorr.join(sparkvif,sparkcorr.Feature == sparkvif.variable,how="inner").drop('variable').select('Feature','Corr','VIF')
corrvif.display()

# COMMAND ----------

display(corrvif)



# COMMAND ----------

#O3_SOMO35 selection

corrvif.filter(col("Feature").contains("climate")).display()

# 'avg_smallwoody_mean_sur',
# 'avg_imp2015_mean_sur',
# 'p50_hrlgrass',
# 'avg_fga2015_mean_sur',
# 'max_eudem'
# 'sum_clc18_112',
# 'sum_clc18_141_mean_sur',
# 'sum_clc18_121_mean_sur',
# 'sum_clc18_243_mean_sur',
# 'sum_clc18_323',
# 'sum_elevbreak_Mountains',
# 'sum_elevbreak_Inlands',
# 'sum_elevbreak_Low_coasts',
# 'sum_urbdegree_11_mean_sur',
# 'sum_urbdegree_30',
# 'sum_envzones_MDM',
# 'sum_envzones_MDN',
# 'sum_envzones_MDS',
# 'sum_envzones_ATC',
# 'sum_envzones_LUS',
# 'carbonseqcapacity_mean_sur',
# 'windspeed_mean_sur',
# 'weight_urb',
# 'pop2018',
# 'droughtimp_mean_sur',
# 'weight_tr_mean_sur',
# 'ecoclimaregions_1',
# 'biogeoregions_9',
# 'biogeoregions_4',
# 'biogeoregions_6',
# 'biogeoregions_1',
# 'climate_HU',
# 'climate_TX',
# 'climate_TG',
# 'cams_O3'


# COMMAND ----------

# MAGIC %md
# MAGIC ### Check of correlation & VIF analysis for selected features

# COMMAND ----------

# Set of features selected for each pollutant based on correlation to target and features multicolinearity (VIF) analysis (see: https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/2846267096674537/command/2490192172322150)

no2cols = ['eRep_NO2','GridNum1km','Year','AreaHa',
           'avg_smallwoody_mean_sur',
'avg_imp2015_mean_sur',
'avg_hrlgrass',
'avg_fga2015_mean_sur',
'avg_eudem',
'sum_clc18_111_mean_sur',
'sum_clc18_121_mean_sur',
'sum_clc18_141_mean_sur',
'sum_clc18_122_mean_sur',
'sum_clc18_211',
'sum_clc18_311_mean_sur',
'sum_clc18_312_mean_sur',
'sum_clc18_313_mean_sur',
'sum_elevbreak_Inlands',
'sum_elevbreak_Mountains',
'sum_urbdegree_30',
'sum_urbdegree_11_mean_sur',
'sum_urbdegree_12_mean_sur',
'sum_urbdegree_13_mean_sur',
'sum_envzones_LUS',
'sum_envzones_ATC',
'carbonseqcapacity_mean_sur',
'weight_urb',
'pop2018',
'windspeed_mean_sur',
'droughtimp_mean_sur',
'weight_tr_mean_sur',
'weight_tr',
'ecoclimaregions_28',
'biogeoregions_6',
'cams_NO2']

# PM2.5 feature selection # NOTE: XGB seems still to overfit although cv error is below ref
pm25cols = ['eRep_PM25','GridNum1km','Year','AreaHa',
           'max_smallwoody',
'max_imp2015', #
'max_fga2015', #
'std_eudem',
'sum_clc18_112',
'sum_clc18_211_mean_sur',
'sum_clc18_231',
'sum_clc18_312_mean_sur',
'sum_clc18_523_mean_sur',
'sum_elevbreak_Low_coasts',
'sum_elevbreak_Mountains',
'sum_elevbreak_Uplands',
'sum_elevbreak_Inlands',
'sum_urbdegree_11_mean_sur',
'sum_urbdegree_21_mean_sur',
'sum_envzones_BOR',
'sum_envzones_CON',
'carbonseqcapacity_mean_sur',
'windspeed_mean_sur',
'weight_urb', #
'pop2018',
'weight_tr_mean_var_sur',
'ecoclimaregions_5',
'ecoclimaregions_7',
'biogeoregions_7',
'biogeoregions_4',
'climate_RR',
'climate_PP', #
'climate_TX', #
'cams_PM25']

# PM10 feature selection
pm10cols = ['eRep_PM10','GridNum1km','Year','AreaHa',
           'avg_smallwoody_mean_sur',
'max_imp2015',
'max_fga2015',
'avg_eudem',
'std_eudem',
'sum_clc18_112',
'sum_clc18_231_mean_sur',
'sum_clc18_312_mean_sur',
'sum_elevbreak_Mountains',
'sum_elevbreak_Inlands',
'sum_urbdegree_11_mean_sur',
'sum_urbdegree_30',
'sum_envzones_BOR',
'sum_envzones_PAN',
'carbonseqcapacity_mean_sur',
'windspeed_mean_sur',
'weight_urb',
'pop2018',
'droughtimp',
'ecoclimaregions_7',
'ecoclimaregions_1',
'biogeoregions_7',
'biogeoregions_4',
'climate_RR',
'climate_HU',
'climate_TG',
'climate_TX',
'cams_PM10']

# O3_SOMO35 feature selection
o3somo35cols = ['eRep_O3_SOMO35','GridNum1km','Year','AreaHa',
'avg_smallwoody_mean_sur',
'avg_imp2015_mean_sur',
'p50_hrlgrass',
'avg_fga2015_mean_sur',
'max_eudem',
'sum_clc18_112',
'sum_clc18_141_mean_sur',
'sum_clc18_121_mean_sur',
'sum_clc18_243_mean_sur',
'sum_clc18_323',
'sum_elevbreak_Mountains',
'sum_elevbreak_Inlands',
'sum_elevbreak_Low_coasts',
'sum_urbdegree_11_mean_sur',
'sum_urbdegree_30',
# 'sum_envzones_MDM',
# 'sum_envzones_MDN',
# 'sum_envzones_MDS',
'sum_envzones_ATC',
'sum_envzones_LUS',
'carbonseqcapacity_mean_sur',
'windspeed_mean_sur',
'weight_urb',
'pop2018',
'droughtimp_mean_sur',
'weight_tr_mean_sur',
'ecoclimaregions_1',
'biogeoregions_9',
'biogeoregions_4',
'biogeoregions_6',
'biogeoregions_1',
'climate_HU',
'climate_TX',
'climate_TG',
'cams_O3']

# O3_SOMO10 feature selection
o3somo10cols = ['eRep_O3_SOMO10','GridNum1km','Year','AreaHa',
'avg_smallwoody_mean_sur',
'avg_imp2015_mean_sur',
'p50_hrlgrass',
'avg_fga2015_mean_sur',
'max_eudem',
'sum_clc18_112',
'sum_clc18_141_mean_sur',
'sum_clc18_121_mean_sur',
'sum_clc18_243_mean_sur',
'sum_clc18_323',
'sum_elevbreak_Mountains',
'sum_elevbreak_Inlands',
'sum_elevbreak_Low_coasts',
'sum_urbdegree_11_mean_sur',
'sum_urbdegree_30',
# 'sum_envzones_MDM',
# 'sum_envzones_MDN',
# 'sum_envzones_MDS',
'sum_envzones_ATC',
'sum_envzones_LUS',
'carbonseqcapacity_mean_sur',
'windspeed_mean_sur',
'weight_urb',
'pop2018',
'droughtimp_mean_sur',
'weight_tr_mean_sur',
'ecoclimaregions_1',
'biogeoregions_9',
'biogeoregions_4',
'biogeoregions_6',
'biogeoregions_1',
'climate_HU',
'climate_TX',
'climate_TG',
'cams_O3']



# COMMAND ----------

dftraining2 = dftraining

# COMMAND ----------

dftraining = dftraining2

# COMMAND ----------

# Check of correlation & VIF analysis for selected variables
# PM2.5


dftraining = loadAQMLDataForVIF(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput).select(pm25cols)

# Getting the input into pandas df
df = dftraining.drop('GridNum1km','Year','AreaHa').toPandas()

# Getting features list for analysis
features = dftraining.drop('eRep_PM25','GridNum1km','Year','AreaHa').toPandas()
Xvif = dmatrix(features, df, return_type='dataframe')

# Correlation between each of the feature and target
corr = df.corrwith(df['eRep_PM25'])

# Generating spark df with features and corresponding correlations
dfcorr = corr.to_frame()
dfcorr = dfcorr.drop('eRep_PM25')
dfcorr['variable'] = Xvif.columns
dfcorr.columns =['Corr', 'Feature']
sparkcorr=spark.createDataFrame(dfcorr) 

# Calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])]
vif['variable'] = Xvif.columns

# Generating spark df with features and corresponding vifs
sparkvif=spark.createDataFrame(vif)

# Joining correlation and vif and display
corrvif = sparkcorr.join(sparkvif,sparkcorr.Feature == sparkvif.variable,how="inner").drop('variable').select('Feature','Corr','VIF')
corrvif.display()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


