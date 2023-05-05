# Databricks notebook source
"""
================================================================================
Pipeline to build input Datasets for ML Predictions applying additional  
filters on years and pollutants.

NOTE:
  Original work developed by Mikel Gonzalez Gainza (mgonzalez@tracasa.es)
  
Summary of data sets which need to be included:

  1. Static data, time (year) and pollutant independent data part.
  
     - Administrative boundaries, 
       to limit area for ML to the European area of interest, see: 
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847866
       
     - Static data set is located at "dis2datalake/airquality-predictions/StaticData/FinalStaticDataNoGS.parquet",
       but only a subset of this data will be taken for ML training, see VIF results in (#139610) 
       and selection at:
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847873
           
     The static data are only joined based on GridNum identifier (here: 1km).
     
  2. Climatic data, time (year) dependent, but pollutant independent data part.
  
     Hourly dataset, waiting for ETC input
     
  3. Air quality data, both time (year) and pollutant dependent part.
  
     Both time (start and end year) and pollutant parameters should affect selection on these data sets. 
     Since the group of available pollutants-statistics is a bit different for each data set, 
     and it may be interesting also to combine different pollutants for features and targets (e.g. using CAMS PM2.5 for predicting BaP) 
     Artur suggests to have separate selection lists:

     - hourly AQ values from aQ e-Reporting, in part 1 both UTD and validated can be used,         
     
     - CAMS data, which belongs to features, Artur has been combining the data together in here: 
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847889, 
       this example can be used for info on file locations and variables to include;
       
       pollutant-statistics list:
         + NO2_avg (cams_NO2), 
         + PM10_avg (cams_PM10), 
         + PM25_avg (cams_PM25), 
         + O3_avg (cams_O3), 
         + O3_SOMO35 (cams_SOMO35), 
         + SO2_avg (cams_SO2)        
     
     The AQ data are joined based on GridNum identifier (here: 1km) and year.
     
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/155728
Author   : mgonzalez@tracasa.es

================================================================================
"""

# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}


# COMMAND ----------

# MAGIC %md #User settings to customize pipeline to be run

# COMMAND ----------

"""
List of attributes to fetch from static data.
This is 2nd version of static data set located at "dis2datalake/airquality-predictions/StaticData/JoinedGS_*.parquet", 
but only a subset of this data will be taken for ML training.
"""
DEFAULT_TRAINING_START_YEAR = 2016
DEFAULT_TRAINING_END_YEAR = 2019
DEFAULT_PREDICT_START_YEAR = 2020
DEFAULT_PREDICT_END_YEAR = 2020

DEFAULT_FEATURE_SUBSETS = ["GridNum1km","windspeed","weight_tr","weight_tr_mean_var_sur","weight_urb","pop2018","pop2018_mean_var_sur","min_imp2015","var_imp2015","avg_imp2015_mean_diff_sur","min_fga2015","max_fga2015",
    "avg_fga2015_mean_sur","sum_urbdegree_12","sum_urbdegree_13","sum_urbdegree_21","sum_urbdegree_22","sum_urbdegree_30","sum_urbdegree_13_mean_sur","sum_urbdegree_21_mean_sur","sum_urbdegree_22_mean_sur",
    "sum_urbdegree_12_mean_sur","var_eudem","avg_eudem_mean_diff_sur","avg_eudem_mean_sur","sum_clc18_211","sum_clc18_312","sum_clc18_311","sum_clc18_231","sum_clc18_313","sum_clc18_243","sum_clc18_242",
    "sum_clc18_523","sum_clc18_121","sum_clc18_142","sum_clc18_111","sum_clc18_122","sum_clc18_141","sum_clc18_312_mean_sur","sum_clc18_311_mean_sur","sum_clc18_231_mean_sur","sum_clc18_313_mean_sur",
    "sum_clc18_243_mean_sur","sum_clc18_242_mean_sur","sum_clc18_523_mean_sur","sum_clc18_121_mean_sur","sum_clc18_142_mean_sur","sum_clc18_111_mean_sur","sum_clc18_122_mean_sur","sum_clc18_141_mean_sur",
    "droughtimp_mean_sur","soilorgcarbon_mean_sur","min_smallwoody","max_smallwoody","p50_smallwoody","var_smallwoody","avg_smallwoody_mean_sur","min_hrlgrass","max_hrlgrass","var_hrlgrass","p25_hrlgrass",
    "sum_ripzones_20","sum_ripzones_3","sum_ripzones_4","sum_ripzones_8","sum_ripzones_2"]

dbutils.widgets.text(name='TrainingStartYear', defaultValue=str(DEFAULT_TRAINING_START_YEAR), label='Training start year')
dbutils.widgets.text(name='TrainingEndYear', defaultValue=str(DEFAULT_TRAINING_END_YEAR), label='Training end year')
dbutils.widgets.text(name='PredictStartYear', defaultValue=str(DEFAULT_PREDICT_START_YEAR), label='Predict start year')
dbutils.widgets.text(name='PredictEndYear', defaultValue=str(DEFAULT_PREDICT_END_YEAR), label='Predict end year')
dbutils.widgets.multiselect('FeatureSubsets', "GridNum1km", DEFAULT_FEATURE_SUBSETS, 'Feature subsets')

