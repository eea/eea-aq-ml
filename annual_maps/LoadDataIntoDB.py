# Databricks notebook source
"""
================================================================================
Pipeline to build input Datasets for ML Predictions applying additional  
filters on years and pollutants.

NOTE:
  Original work developed by Artur Bernard Gsella (Artur.Gsella@eea.europa.eu)
  at:
  https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862
  
  Review by David Rivero at:
  https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/4435339981114382
  
Summary of data sets which need to be included:

  1. Static data, time (year) and pollutant independent data part.
  
     - Administrative boundaries, 
       to limit area for ML to the European area of interest, see: 
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847866
       
     - Static data set is located at "dis2datalake/airquality-predictions/StaticData/FinalStaticDataNoGS.parquet",
       but only a subset of this data will be taken for ML training, see VIF results in (#139610) 
       and selection at:
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847873
       
     - There will be another part of static data set available, after the grid-shifting task (#139610)
       and multicollinearity analysis is completed, the location of source table and selection of 
       data subset will be given later in (#141262), to be joined with the rest of the input.
       
     The static data are only joined based on GridNum identifier (here: 1km).
     
  2. Climatic data, time (year) dependent, but pollutant independent data part.
  
     Artur has been combining this data here: 
     https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847888, 
     from the VIF analysis it seems that we should be using climate_RR and climate_QQ, 
     so no selection on these variables is needed; however, the time parameters (start and end year) 
     should affect selection of this data set.
     
     The climatic data are joined based on GridNum identifier (here: 1km) and year.
     
  3. Air quality data, both time (year) and pollutant dependent part.
  
     Both time (start and end year) and pollutant parameters should affect selection on these data sets. 
     Since the group of available pollutants-statistics is a bit different for each data set, 
     and it may be interesting also to combine different pollutants for features and targets (e.g. using CAMS PM2.5 for predicting BaP) 
     Artur suggests to have separate selection lists:
     
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
         
     - E1b data, which will serve as target, Artur has been combining the data together in here: 
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/181335459309456, 
       this example can be used for info on file locations and variables to include;
       
       pollutant-statistics list:
         + NO2_avg (e1b_NO2), 
         + PM10_avg (e1b_PM10), 
         + PM25_avg (e1b_PM25), 
         + BaP_avg (e1b_BaP), 
         + O3_AOT40c (e1b_O3)
         
     - AQ measurement data should be joined from "dis2datalake/airquality_predictions/AQeReporting/AirQualityStatistics.parquet"
       using query as given in:
       https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1953539243064437,
       see use of "AQ_MEASUREMENT_DATA_QUERY" variable in this Notebook.
       
       This is selection of AQ statistics originating from the AQ e-Reporting SQL DB table.
       The attributes from this table are the 'target' in ML training.
       
       pollutant-statistics list:
         + NO2_avg (no2_avg_eRep), 
         + PM10_avg (pm10_avg_eRep), 
         + PM25_avg (pm25_avg_eRep), 
         + O3_avg (o3_avg_eRep), 
         + O3_SOMO35 (o3_somo35_eRep), 
         + SO2_avg (so2_avg_eRep),
         + BaP_avg (bap_avg_eRep),
         + O3_AOT40c (o3_aot40c_eRep)
     
     The AQ data are joined based on GridNum identifier (here: 1km) and year.
     
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/141261
Author   : ahuarte@tracasa.es

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

DEFAULT_TRAINING_START_YEAR = 2016
DEFAULT_TRAINING_END_YEAR = 2019
DEFAULT_PREDICT_START_YEAR = 2020
DEFAULT_PREDICT_END_YEAR = 2020

# Available list of Pollutants:
# - CAMS: NO2, PM10, PM25, O3, O3_SOMO35, SO2.
# - E1b: NO2, PM10, PM25, BaP, O3_AOT40c.
# - AQ Measurements: NO2, PM10, PM25, O3, O3_SOMO35, O3_AOT40c, BaP, SO2.

DEFAULT_ADMINISTRATIVE_FILTER = ''
DEFAULT_CAMS_POLLUTANT_FILTER = ['NO2', 'PM10', 'PM25', 'O3', 'O3_SOMO35', 'SO2']
DEFAULT_CITIES_FILTER = ''
DEFAULT_E1b_POLLUTANT_FILTER = ['NO2', 'PM10', 'PM25', 'BaP', 'O3_AOT40c', 'SO2']
DEFAULT_eRep_POLLUTANT_FILTER = ['NO2', 'PM10', 'PM25',  'BaP','O3', 'O3_SOMO35', 'O3_AOT40c', 'SO2']

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

dbutils.widgets.text(name='AdministrativeFilter', defaultValue=str(DEFAULT_ADMINISTRATIVE_FILTER), label='Administrative Filter')
dbutils.widgets.text(name='CitiesFilter', defaultValue=str(DEFAULT_CITIES_FILTER), label='Cities Filter')
dbutils.widgets.multiselect('CAMSPollutant', "NO2", DEFAULT_CAMS_POLLUTANT_FILTER, label='CAMS Pollutant')
dbutils.widgets.multiselect('E1bPollutant', "NO2", DEFAULT_E1b_POLLUTANT_FILTER, label='E1b Pollutant')
dbutils.widgets.multiselect('eRepPollutant', "NO2", DEFAULT_eRep_POLLUTANT_FILTER, label='ERep Pollutant')

# COMMAND ----------

# DBTITLE 1,User settings to customize the final Pipeline to run

"""
List of attributes to fetch from static data.
This is 2nd version of static data set located at "dis2datalake/airquality-predictions/StaticData/JoinedGS_*.parquet", 
but only a subset of this data will be taken for ML training.
"""
TrainingStartYear = dbutils.widgets.get('TrainingStartYear')
TrainingEndYear = dbutils.widgets.get('TrainingEndYear')
PredictStartYear = dbutils.widgets.get('PredictStartYear')
PredictEndYear = dbutils.widgets.get('PredictEndYear')
FeatureSubsets = dbutils.widgets.get('FeatureSubsets')

AdministrativeFilter = dbutils.widgets.get('AdministrativeFilter')
CitiesFilter = dbutils.widgets.get('CitiesFilter')
CAMSPollutant = dbutils.widgets.get('CAMSPollutant').split(",")
E1bPollutant = dbutils.widgets.get('E1bPollutant').split(",")
eRepPollutant = dbutils.widgets.get('eRepPollutant').split(",")

# Subset of fields of static data (VERSION-1), 
# see VIF results in:
# https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1664542031207191/command/1664542031207204
STATIC_FIELDS_v1 = FeatureSubsets

# User settings to apply later when the Pipeline is executed.
custom_settings = {
  # Apply MinMaxScaler operation to input CAMS Dataframe (Choose between "TRUE" or "FALSE").
  "CAMS_APPLY_MINMAX_SCALER": "TRUE",
  
  # Apply MinMaxScaler operation to input CLIMATIC Dataframe (Choose between "TRUE" or "FALSE").
  "CLIM_APPLY_MINMAX_SCALER": "TRUE",
  
  # Parquet filename used as input STATIC Dataframe (Choose between "JoinedGS_and_NOGS_AGs_minmax.parquet" which is minmax scaled or "JoinedGS_and_NOGS_AGs.parquet").
  "STATIC_DATA_INPUT_FILE": "JoinedGS_and_NOGS_AGs_final.parquet",
  # Fields to fetch from STATIC data (Choose between STATIC_FIELDS_v0, STATIC_FIELDS_v1, ...).
  "STATIC_DATA_FIELDS": ", ".join(['"{}"'.format(f) for f in STATIC_FIELDS_v1])
}
print(CAMSPollutant)


# COMMAND ----------

# DBTITLE 1,Settings of Input data

# =======================================================================================
# Define several examples of input parameters
# =======================================================================================

# Available list of Pollutants:
# - CAMS: NO2, PM10, PM25, O3, O3_SOMO35, SO2.
# - E1b: NO2, PM10, PM25, BaP, O3_AOT40c.
# - AQ Measurements: NO2, PM10, PM25, O3, O3_SOMO35, O3_AOT40c, BaP, SO2.

settings = []
i = 0
# Set Settings for the Pipeline
for pollu in CAMSPollutant:  
  p = \
  {
    "AdministrativeFilter": AdministrativeFilter,
    "CitiesFilter": CitiesFilter,

    "TrainingStartYear": TrainingStartYear,
    "TrainingEndYear": TrainingEndYear,
    "PredictStartYear": PredictStartYear,
    "PredictEndYear": PredictEndYear,

    "CAMSPollutant": pollu,
    "E1bPollutant": E1bPollutant[i],
    "eRepPollutant": eRepPollutant[i]    
  }
  settings.append(p)
  
print(settings)

# COMMAND ----------


#
# Query to build the AQ Measurement data that provides the 'target' in ML training.
# See:
# https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1953539243064437
#


AQ_MEASUREMENT_DATA_QUERY = """
--
-- This is selection of AQ statistics originating from the AQ e-Reporting SQL DB table.
-- The attributes from this table are the 'target' in ML training.
--
-- WARNING: 
--   The table 'aq_stats_20211029' defined in the original SQL statement provided 
--   by Artur has beed replaced with the special tag '$THIS' in order to indicate 
--   to the Pipeline that current Dataframe is the source of Records.
--
with aqstats as 
(
  select 
      * 
  from 
      $THIS
  where
      AirPollutant in ('NO2','PM10','PM2.5','O3','SO2','BaP in PM10')
      and DataCovFilterYN = 'Yes'
      and OneSPOYN = 'Yes'
      and potentialOutlier in ('No','Unknown')
      and DataAggregationProcessId in ('P1Y','SOMO35','SOMO10','AOT40c')
      and YearOfStatistics > 2013
)

select
    gridyears.GridNum1km
   ,gridyears.YearOfStatistics
   ,no2_avg_eRep
   ,pm10_avg_eRep
   ,pm25_avg_eRep
   ,o3_avg_eRep
   ,o3_somo35_eRep
   ,o3_somo10_eRep
   ,o3_aot40c_eRep
   ,bap_avg_eRep
   ,so2_avg_eRep
from
(
    select distinct 
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics
    from 
        aqstats
)
as gridyears

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as no2_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'NO2' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as no2 on no2.GridNum1km = gridyears.GridNum1km and no2.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as pm10_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'PM10' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as pm10 on pm10.GridNum1km = gridyears.GridNum1km and pm10.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as pm25_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'PM2.5' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as pm25 on pm25.GridNum1km = gridyears.GridNum1km and pm25.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as o3_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'O3' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as o3 on o3.GridNum1km = gridyears.GridNum1km and o3.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct 
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as o3_somo35_eRep
    from 
        aqstats
    where
        AirPollutant = 'O3' and DataAggregationProcessId in ('SOMO35')
    group by
        GridNum1km, YearOfStatistics
)
as somo35 on somo35.GridNum1km = gridyears.GridNum1km and somo35.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct 
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as o3_somo10_eRep
    from 
        aqstats
    where
        AirPollutant = 'O3' and DataAggregationProcessId in ('SOMO10')
    group by
        GridNum1km, YearOfStatistics
)
as somo10 on somo10.GridNum1km = gridyears.GridNum1km and somo10.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as o3_aot40c_eRep
    from 
        aqstats
    where
        AirPollutant = 'O3' and DataAggregationProcessId in ('AOT40c')
    group by
        GridNum1km, YearOfStatistics
)
as aot40 on aot40.GridNum1km = gridyears.GridNum1km and aot40.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as bap_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'BaP in PM10' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as bap on bap.GridNum1km = gridyears.GridNum1km and bap.YearOfStatistics = gridyears.YearOfStatistics

left join
(
    select distinct
        cast(GridNum1km as bigint) as GridNum1km, YearOfStatistics, avg(AirPollutionLevel) as so2_avg_eRep
    from 
        aqstats
    where
        AirPollutant = 'SO2' and DataAggregationProcessId in ('P1Y')
    group by
        GridNum1km, YearOfStatistics
)
as so2 on so2.GridNum1km = gridyears.GridNum1km and so2.YearOfStatistics = gridyears.YearOfStatistics


where
    gridyears.GridNum1km is not null
order by 
    gridyears.GridNum1km, gridyears.YearOfStatistics

"""


# COMMAND ----------

# DBTITLE 1,Implementing & testing Pipeline declaration for Inputs

# Pipeline declaration with the Dataflow to perform (As STRING).
inputs_pipeline_as_text = """
{
  "Pipeline": [
    # ============================================================================================
    # 1. Static, time (year) and pollutant independent data part.
    # ============================================================================================
    #
    # --------------------------------------------------------------------------------------------
    #  - Administrative boundaries, 
    #    to limit area for ML to the European area of interest, see: 
    #    https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847866
    # --------------------------------------------------------------------------------------------
    {
      "Type": "Branch",
      "Label": "Performing branch of 'Administrative data'...",
      "IgnoreModule": false,
      
      "Pipeline": [
        # This is 100m grid with administrative boundaries:
        # GridNum id on 1km is added
        # adminbound is the id linking with administrative lookup table
        # uacities is linking with cities lookup table
        {
          "Type": "Dataset",
          
          "SQL": "SELECT DISTINCT 
                    GridNum10km, GridNum1km, uacities, adminbound, AreaHa
                  FROM 
                    ml_input_from_jedi.aq_adminman_1000",
                    
          "Pipeline": [
            {
              "Type": "DerivedColumn",
              "Columns": {
                "x": "F.EEA_gridid2laea_x('GridNum1km') + F.lit(500)",
                "y": "F.EEA_gridid2laea_y('GridNum1km') - F.lit(500)"
              }
            },
            {
              "Type": "Cache",
              "Name": "AQ_ADMINMAN_1000"
            }
          ]
        },
        # This is cities lookup table:
        # id links to the grid with administrative boundaries
        # The selections possible for user should be based on list of cities
        {
          "Type": "Dataset",
          "Name": "UACITIES_LOOKUP",
                    
          "SQL": "SELECT DISTINCT 
                    OBJECTID as id, URAU_CODE as CityCode, URAU_NAME as City
                   ,case 
                      when CNTR_ID = 'EL' then 'GR' 
                      when CNTR_ID = 'UK' then 'GB' 
                      else CNTR_ID
                    end as cntry_id
                  FROM
                    hra_input_from_sqldbs.uacities_lookup"
        },
        # This is the administrative areas lookup table, it will be used for filtering on areas for modelling:
        # ADM_ID is linking with the grid with administrative boundaries
        # The selections possible for user should be based on lists: Country ISO code, NUTS3 name, Group of countries (EEA39, EU27_noUK, EU10, EU15, EU27, EU28)
        # Filtering out seas and no country areas should be always kept in the query
        {
          "Type": "Dataset",
          "Name": "AQ_ADMIN_LOOKUP",
          
          "SQL": "SELECT DISTINCT
                    ADM_ID
                   ,case when ICC = 'GI' then 'Gibraltar' else ADM_country end as Country
                   ,case
                      when ADM_country = 'Greece' then ICC
                      when ADM_country = 'Kosovo' then LEVEL0_code
                      when ADM_country = 'United Kingdom' and ICC = 'GB' then ICC
                      when ADM_country = 'United Kingdom' and ICC = 'ND' then 'GB'
                      else ICC
                    end as Country_ISO -- this is to adapt the values of the attribute for joining with lookup tables
                 --,ICC
                   ,LEVEL3_name -- these are NUTS3 levels names
                   ,LEVEL2_name -- NUTS2
                   ,LEVEL1_name -- NUTS1
                   ,LEVEL0_name -- NUTS0
                 --,TAA
                 --,TAA_DESN
                   ,LEVEL3_code -- these are NUTS3 levels codes
                   ,LEVEL2_code -- NUTS2
                   ,LEVEL1_code -- NUTS1
                   ,LEVEL0_code -- NUTS0
                   ,EEA39
                 --,EEA33
                 --,EEA32
                 --,CC
                   ,EU10
                   ,EU15
                   ,EU27
                   ,EU28
                 --,EFTA4
                 --,OVERSEAS
                   ,EU27_noUK
                   
                 FROM 
                   hra_input_from_sqldbs.aq_admin_lookup 
                 WHERE 
                   LEVEL0_code <> 'ZZ' and LEVEL0_code not like '%SEA'"
        },
        # Apply lookup filtering based on alphanumeric attributes (Administrative boundaries, Cities, Spatial filters).
        {
          "Type": "Branch",
          
          "Pipeline": [
            # Option 1: Activate full European AOI as default lookup filter.
            {
              "Type": "GetObject",
              "Name": "FULL_LOOKUP_INPUT",
              
              "ObjectName": "AQ_ADMINMAN_1000"
            },
            # Option 2: Is "AdministrativeFilter" tag defined? then filtering with ADMIN_LOOKUP_INPUT as join of AQ_ADMIN_LOOKUP and AQ_ADMINMAN_1000 Datasets.
            {
              "Type": "Join",
              "Name": "ADMIN_LOOKUP_INPUT",
              
              "ApplyModule": "$AdministrativeFilter",
              
              "Left": "AQ_ADMINMAN_1000",
              "LeftKey": "adminbound",
              "Right": "AQ_ADMIN_LOOKUP",
              "RightKey": "ADM_ID",
              "JoinType": "inner",
              
              "Pipeline": [
                {
                  "Type": "Filter",
                  "Expression": "$AdministrativeFilter"
                }
              ]
            },
            # Option 3: Is "CitiesFilter" tag defined? then filtering with CITIES_LOOKUP_INPUT as join of UACITIES_LOOKUP and AQ_ADMINMAN_1000 Datasets.
            {
              "Type": "Join",
              "Name": "CITIES_LOOKUP_INPUT",
              
              "ApplyModule": "$CitiesFilter",
              
              "Left": "AQ_ADMINMAN_1000",
              "LeftKey": "uacities",
              "Right": "UACITIES_LOOKUP",
              "RightKey": "id",
              "JoinType": "inner",
              
              "Pipeline": [
                {
                  "Type": "Filter",
                  "Expression": "$CitiesFilter"
                }
              ]
            },
            # Final steps of alphanumeric/spatial filtering.
            {
              "Type": "Branch",
              "Name": "LOOKUP_INPUT_WITH_ALL_ATTRIBUTES"
            },
            {
              "Type": "Select",
              "Columns": [ "GridNum1km", "x", "y" ]
            },
            {
              "Type": "Distinct",
              "Name": "LOOKUP_INPUT"
            }
          ]
        }
      ]
    },
    # --------------------------------------------------------------------------------------------
    #  - This is 2nd version of static data set located at "dis2datalake/airquality-predictions/StaticData/JoinedGS_*.parquet",
    #    but only a subset of this data will be taken for ML training.
    # --------------------------------------------------------------------------------------------
    {
      "Type": "Dataset",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-predictions",
      "File": "StaticData/$STATIC_DATA_INPUT_FILE",
      
      "Pipeline": [
        {
          "Type": "Filter",
          "Expression": "(weight_tr >= 0)"
        },
        {
          "Type": "SQL",
          "Name": "STATIC_INPUT",
          "SQL": "SELECT * FROM $THIS"
        }
      ]
    },
    # --------------------------------------------------------------------------------------------
    # TODO:
    #  - There will be another part of static data set available, after the grid-shifting task (#139610)
    #    and multicollinearity analysis is completed, the location of source table and selection of 
    #    data subset will be given later in (#141262), to be joined with the rest of the input.
    # --------------------------------------------------------------------------------------------
    # ...
    
    # ============================================================================================
    # 2. Climatic data, time (year) dependent, but pollutant independent data part.
    # ============================================================================================
    # Artur have been combining this data here: 
    # https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847888, 
    # from the VIF analysis it seems that we should be using climate_RR and climate_QQ, 
    # so no selection on these variables is needed; however, the time parameters (start and end year) 
    # should affect selection of this data set.
    {
      "Type": "Dataset",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-climate",
      "File": "Ensemble/ALL/*/CLIMATE_ALL_avg+gapfilling_*-XX-XX.parquet",
          
      "Pipeline": [
        {
          "Type": "Condition",
          "Expression": "'$CLIM_APPLY_MINMAX_SCALER' == 'TRUE'",
          
          "True": [
            {
              "Type": "Preprocessing",
              "Operation": "FastMinMaxScaler",
              "Columns": [ "climate_RR", "climate_QQ", "climate_TG", "climate_TN", "climate_TX", "climate_PP", "climate_HU" ]
            }
          ],
          "False": [
          ]
        },
        {
          "Type": "Select",
          "Name": "CLIMATE_INPUT",
          
          "Columns": [ "GridNum1km", "Year", "climate_RR", "climate_QQ", "climate_TG", "climate_TN", "climate_TX", "climate_PP", "climate_HU" ]
        }
      ]
    },
    # ============================================================================================
    # 3. Air quality data, both time (year) and pollutant dependent part.
    # ============================================================================================
    # Both time (start and end year) and pollutant parameters should affect selection on these data sets. 
    # Since the group of available pollutants-statistics is a bit different for each data set, ca
    # and it may be interesting also to combine different pollutants for features and targets (e.g. using CAMS PM2.5 for predicting BaP) 
    # Artur suggests to have separate selection lists:
    #
    # --------------------------------------------------------------------------------------------
    #  - CAMS data, which belongs to features, Artur has been combining the data together in here: 
    #    https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1002874667847889, 
    #    this example can be used for info on file locations and variables to include;
    #    
    #    pollutant-statistics list:
    #      + NO2_avg (cams_NO2), 
    #      + PM10_avg (cams_PM10), 
    #      + PM25_avg (cams_PM25), 
    #      + O3_avg (cams_O3), 
    #      + O3_SOMO35 (cams_SOMO35), 
    #      + SO2_avg (cams_SO2)
    # --------------------------------------------------------------------------------------------
    {
      "Type": "Dataset",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-cams",
      "File": "Ensemble/ALL/*/CAMS_ALL_avg_*-XX-XX.parquet",
      
      "Pipeline": [
        {
          "Type": "Condition",
          "Expression": "'$CAMS_APPLY_MINMAX_SCALER' == 'TRUE'",
          
          "True": [
            {
              "Type": "Preprocessing",
              "Operation": "FastMinMaxScaler",
              "Columns": [ "cams_$CAMSPollutant" ]
            }
          ],
          "False": [          
          ]
        },
        {
          "Type": "Select",
          "Name": "CAMS_INPUT",
          
          "Columns": [ "GridNum1km", "Year", "cams_$CAMSPollutant" ]
        }
      ]
    },
    # --------------------------------------------------------------------------------------------
    #  - E1b data, which will serve as target, Artur has been combining the data together in here: 
    #    https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/181335459309456,
    #    this example can be used for info on file locations and variables to include;
    #    
    #    pollutant-statistics list:
    #      + NO2_avg (e1b_NO2), 
    #      + PM10_avg (e1b_PM10), 
    #      + PM25_avg (e1b_PM25), 
    #      + BaP_avg (e1b_BaP), 
    #      + O3_AOT40c (e1b_O3)
    # --------------------------------------------------------------------------------------------
    {
      "Type": "Dataset",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-e1b",
      "File": "Ensemble/ALL/*/E1b_ALL_avg_*-XX-XX.parquet",
      
      "Pipeline": [
        {
          "Type": "Select",
          "Name": "E1B_INPUT",
          
          "Columns": [ "~x", "~y" ]
        }
      ]
    },
    # --------------------------------------------------------------------------------------------
    #  - AQ measurement data should be joined from "dis2datalake/airquality_predictions/AQeReporting/AirQualityStatistics.parquet"
    #    using query as given in:
    #    https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/1002874667847862/command/1953539243064437;
    #
    #    This is selection of AQ statistics originating from the AQ e-Reporting SQL DB table.
    #    The attributes from this table are the 'target' in ML training.
    #
    #    pollutant-statistics list:
    #      + NO2_avg (no2_avg_eRep), 
    #      + PM10_avg (pm10_avg_eRep), 
    #      + PM25_avg (pm25_avg_eRep), 
    #      + O3_avg (o3_avg_eRep), 
    #      + O3_SOMO35 (o3_somo35_eRep), 
    #      + O3_SOMO10 (o3_somo10_eRep), 
    #      + SO2_avg (so2_avg_eRep),
    #      + BaP_avg (bap_avg_eRep),
    #      + O3_AOT40c (o3_aot40c_eRep)
    # --------------------------------------------------------------------------------------------
    {
      "Type": "Dataset",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-predictions",
      "File": "AQeReporting/AirQualityStatistics.parquet",
      
      "Pipeline": [
        {
          "Type": "SQL",
          "SQL": "$AQ_MEASUREMENT_DATA_QUERY"
        },
        {
          "Type": "Select",
          "Columns": [ 
            "GridNum1km",
            "YearOfStatistics AS Year", 
            "no2_avg_eRep AS eRep_NO2", 
            "o3_avg_eRep AS eRep_O3", 
            "o3_somo35_eRep AS eRep_O3_SOMO35", 
            "o3_somo10_eRep AS eRep_O3_SOMO10", 
            "o3_aot40c_eRep AS eRep_O3_AOT40c", 
            "pm10_avg_eRep AS eRep_PM10", 
            "pm25_avg_eRep AS eRep_PM25", 
            "bap_avg_eRep AS eRep_BaP",
            "so2_avg_eRep AS eRep_SO2"
          ]
        },
        {
          "Type": "Cache",
          "Name": "EREP_INPUT"
        }
      ]
    },
    # ============================================================================================
    # 4. JOIN of internal tables to compose the outputs.
    # ============================================================================================
    # Joining "EEA-AOI or Admin or Cities" Lookup table and static input table.
    {
      "Type": "Join",
      "Name": "LOOKUP_STATIC_JOIN",
      
      "Left": "LOOKUP_INPUT",
      "LeftKey": "GridNum1km",
      "Right": "STATIC_INPUT",
      "RightKey": "GridNum1km",
      "JoinType": "inner"
    },
    # Joining CLIMATE and CAMS tables.
    {
      "Type": "Join",
      "Name": "CLIMATE_CAMS_JOIN",

      "Left": "CLIMATE_INPUT",
      "LeftKey": [ "GridNum1km", "Year" ],
      "Right": "CAMS_INPUT",
      "RightKey": [ "GridNum1km", "Year" ],
      "JoinType": "inner"
    },
    # Joining LOOKUP, STATIC, CLIMATE and CAMS tables.
    {
      "Type": "Join",
      
      "Left": "LOOKUP_STATIC_JOIN",
      "LeftKey": "GridNum1km",
      "Right": "CLIMATE_CAMS_JOIN",
      "RightKey": "GridNum1km",
      "JoinType": "inner" 
    },
    {
      "Type": "Filter",
      "Expression": "(Year >= $TrainingStartYear AND Year <= $TrainingEndYear) OR (Year >= $PredictStartYear AND Year <= $PredictEndYear)"
    },
    {
      "Type": "Cache",
      "Name": "ML_MAIN_INPUT"
    }
    # ============================================================================================
    # 5. Outputs.
    # ============================================================================================    
    # ...
  ]
}
""".replace("$AQ_MEASUREMENT_DATA_QUERY", AQ_MEASUREMENT_DATA_QUERY)
    
# print(inputs_pipeline_as_text)


# COMMAND ----------

# DBTITLE 1,Implementing & testing Pipeline declaration for Outputs
# Pipeline declaration with the Dataflow to perform (As STRING).
output_pipeline_as_text = """
{
  "Pipeline": [
    # ============================================================================================
    # 5. Outputs.
    # ============================================================================================
    # Building ML Predictions Dataset (When is specified a valid Year period).
    {
      "Type": "GetObject",
      "ObjectName": "ML_MAIN_INPUT",
      
      "ApplyModule": "$EVAL($PredictStartYear > 0 and $PredictEndYear >= $PredictStartYear)",
      
      "Pipeline": [
        {
          "Type": "Filter",
          "Name": "ML_PREDICTION_DATASET",
          
          "Expression": "(Year >= $PredictStartYear AND Year <= $PredictEndYear)"
        },
        {
          "Type": "Select",
          "Name": "ML_PREDICTION_TABLE",
          
          "Columns": [ "~x", "~y"]
        }
      ]
    },
    # Building ML Training/Validation Datasets (When is specified a valid Year period and 'target' Pollutant).
    {
      "Type": "Branch",
      
      "ApplyModule": "$EVAL($TrainingStartYear > 0 and $TrainingEndYear >= $TrainingStartYear and (len('$E1bPollutant') > 0 or len('$eRepPollutant') > 0))",
      
      "Pipeline": [
        # Building ML E1B Training data.
        {
          "Type": "GetObject",
          "ObjectName": "E1B_INPUT",
          
          "ApplyModule": "$EVAL(len('$E1bPollutant') > 0)",
          
          "Pipeline": [
            {
              "Type": "Select",
              "Name": "E1B_PREP_INPUT",
              
              "Columns": [ "GridNum1km", "Year", "e1b_$E1bPollutant" ]
            },
            {
              "Type": "Join",
              
              "Left": "ML_MAIN_INPUT",
              "LeftKey": [ "GridNum1km", "Year" ],
              "Right": "E1B_PREP_INPUT",
              "RightKey": [ "GridNum1km", "Year" ],
              "JoinType": "inner"
            },
            {
              "Type": "Filter",
              "Name": "ML_E1B_TRAINING_DATASET",
              
              "Expression": "(Year >= $TrainingStartYear AND Year <= $TrainingEndYear AND e1b_$E1bPollutant IS NOT NULL)"
            },
            {
              "Type": "Select",
              "Name": "ML_E1B_TRAINING_TABLE",
              
              "Columns": [ "~x", "~y"]
            }
          ]
        },
        # Building ML EREP (AQ Measurements / eReporting) Training/Validation data.
        {
          "Type": "GetObject",
          "ObjectName": "EREP_INPUT",
          
          "ApplyModule": "$EVAL(len('$eRepPollutant') > 0)",
          
          "Pipeline": [
            {
              "Type": "Select",
              "Name": "EREP_PREP_INPUT",
              
              "Columns": [ "GridNum1km", "Year", "eRep_$eRepPollutant" ]
            },
            {
              "Type": "Join",
              
              "Left": "ML_MAIN_INPUT",
              "LeftKey": [ "GridNum1km", "Year" ],
              "Right": "EREP_PREP_INPUT",
              "RightKey": [ "GridNum1km", "Year" ],
              "JoinType": "inner"
            },
            {
              "Type": "Filter",
              "Expression": "(eRep_$eRepPollutant IS NOT NULL)"
            },
            {
              "Type": "Cache",
              "Name": "EREP_PREP_TABLE_0"
            },
            # Splitting Rows for training and prediction/validation periods.
            {
              "Type": "Branch",
              
              "Pipeline": [
                {
                  "Type": "GetObject",
                  "ObjectName": "EREP_PREP_TABLE_0"
                },
                {
                  # Separate all eRep records from non-overlapping years on the side of training period.
                  "Type": "Filter",
                  "Name": "EREP_PREP_TABLE_TRAINING_1",
                  
                  "Expression": "($TrainingStartYear < $PredictStartYear AND Year >= $TrainingStartYear AND Year < $PredictStartYear) 
                                  OR
                                 ($TrainingEndYear > $PredictEndYear AND Year > $PredictEndYear AND Year <= $TrainingEndYear)
                                  OR 
                                (($TrainingStartYear > $PredictEndYear OR $TrainingEndYear < $PredictStartYear) AND Year >= $TrainingStartYear AND Year <= $TrainingEndYear)"
                },
                {
                  "Type": "GetObject",
                  "ObjectName": "EREP_PREP_TABLE_0"
                },
                {
                  # Separate all eRep records from non-overlapping years on the side of prediction/validation period.
                  "Type": "Filter",
                  "Name": "EREP_PREP_TABLE_VALIDATION_1",
                  
                  "Expression": "($PredictStartYear < $TrainingStartYear AND Year >= $PredictStartYear AND Year < $TrainingStartYear)
                                  OR
                                 ($PredictEndYear > $TrainingEndYear AND Year > $TrainingEndYear AND Year <= $PredictEndYear)
                                  OR
                                (($PredictStartYear > $TrainingEndYear OR $PredictEndYear < $TrainingStartYear) AND Year >= $PredictStartYear AND Year <= $PredictEndYear)"
                },
                {
                  "Type": "GetObject",
                  "ObjectName": "EREP_PREP_TABLE_0"
                },
                {
                  # Separate all eRep records from overlapping years and get --- 95% --- random for training and --- 5% --- for validation.
                  "Type": "Filter",
                  "Name": "EREP_PREP_OVERLAPPING_TABLE",
                  
                  "Expression": "($PredictStartYear <= $TrainingStartYear AND $PredictEndYear >= $TrainingEndYear AND Year >= $TrainingStartYear AND Year <= $TrainingEndYear)
                                  OR
                                 ($PredictStartYear >= $TrainingStartYear AND $PredictEndYear <= $TrainingEndYear AND Year >= $PredictStartYear AND Year <= $PredictEndYear)
                                  OR
                                 ($PredictStartYear >= $TrainingStartYear AND $PredictEndYear >= $TrainingEndYear AND Year >= $PredictStartYear AND Year <= $TrainingEndYear)
                                  OR 
                                 ($PredictStartYear <= $TrainingStartYear AND $PredictEndYear <= $TrainingEndYear AND Year >= $TrainingStartYear AND Year <= $PredictEndYear)"
                },
                {
                  "Type": "RandomSplit",
                  "Seed": 42,
                  "ObjectNames": [ "EREP_PREP_TABLE_TRAINING_2", "EREP_PREP_TABLE_VALIDATION_2" ],
                  "Weights": [ 0.95, 0.05 ]
                }
              ]
            },
            # Final steps.
            {
              "Type": "Union",
              "Name": "ML_EREP_TRAINING_DATASET",
              
              "Left": "EREP_PREP_TABLE_TRAINING_1",
              "Right": "EREP_PREP_TABLE_TRAINING_2"
            },
            {
              "Type": "Select",
              "Name": "ML_EREP_TRAINING_TABLE",
              
              "Columns": [ "~x", "~y"]
            },
            {
              "Type": "Union",
              "Name": "ML_VALIDATION_DATASET",
              
              "Left": "EREP_PREP_TABLE_VALIDATION_1",
              "Right": "EREP_PREP_TABLE_VALIDATION_2"
            },            
            {
              "Type": "Select",
              "Name": "ML_VALIDATION_TABLE",
              
              "Columns": [ "~x", "~y"]
            }
          ]
        }
      ]
    }
    # ============================================================================================
    # 6. Serialize Outputs.
    # ============================================================================================    
    # ...
  ]
}
"""


# COMMAND ----------

# DBTITLE 1,Implementing & testing Pipeline declaration to save Outputs (Parquet files)

# Pipeline declaration with the Dataflow to perform (As STRING).
serialize_pipeline_as_text = """
{
  "Pipeline": [
    # ============================================================================================
    # 6. Serialize Outputs.
    # ============================================================================================    
    {
      "Type": "GetObject",
      "ObjectName": "ML_PREDICTION_TABLE",
      
      "ApplyModule": "$EVAL($PredictStartYear > 0 and $PredictEndYear >= $PredictStartYear)",
      
      "Pipeline": [
        {
          "Type": "Output",
          
          "StorageAccount": "dis2datalake",
          "Container": "airquality-predictions",
          "File": "$OutputFolder/$CurrentItem_$eRepPollutant_$PredictStartYear-$PredictEndYear/$CurrentDate/prediction_input_$eRepPollutant_$PredictStartYear-$PredictEndYear.parquet",
          
          "OutputEngine": "Spark"
        }
      ]
    },
    {
      "Type": "Branch",
      "ApplyModule": "$EVAL($TrainingStartYear > 0 and $TrainingEndYear >= $TrainingStartYear)",
      
      "Pipeline": [
        {
          "Type": "GetObject",
          "ObjectName": "ML_E1B_TRAINING_TABLE",
          
          "ApplyModule": "$EVAL(len('$E1bPollutant') > 0)",
          
          "Pipeline": [
            {
              "Type": "Output",
              
              "StorageAccount": "dis2datalake",
              "Container": "airquality-predictions",
              "File": "$OutputFolder/$CurrentItem_$eRepPollutant_$PredictStartYear-$PredictEndYear/$CurrentDate/training_input_e1b_$E1bPollutant_$TrainingStartYear-$TrainingEndYear.parquet",
              
              "OutputEngine": "Spark"
            }
          ]
        },
        {
          "Type": "GetObject",
          "ObjectName": "ML_EREP_TRAINING_TABLE",
          
          "Pipeline": [
            {
              "Type": "Output",
              
              "StorageAccount": "dis2datalake",
              "Container": "airquality-predictions",
              "File": "$OutputFolder/$CurrentItem_$eRepPollutant_$PredictStartYear-$PredictEndYear/$CurrentDate/training_input_eRep_$eRepPollutant_$TrainingStartYear-$TrainingEndYear.parquet",
              
              "OutputEngine": "Spark"
            }
          ]
        },
        {
          "Type": "GetObject",
          "ObjectName": "ML_VALIDATION_TABLE",
          
          "Pipeline": [
            {
              "Type": "Output",
              
              "StorageAccount": "dis2datalake",
              "Container": "airquality-predictions",
              "File": "$OutputFolder/$CurrentItem_$eRepPollutant_$PredictStartYear-$PredictEndYear/$CurrentDate/validation_input_$eRepPollutant_$PredictStartYear-$PredictEndYear.parquet",
              
              "OutputEngine": "Spark"
            }
          ]
        }
      ]
    }
  ]
}
"""

spark.conf.set("spark.databricks.io.cache.enabled", "true")
serialize_settings = \
{
  "OutputFolder": "ML_Input"
}

# Process Pipeline of each example saving Output files.
index = 0
for settings_ob in settings:
    example_id = index + 1
    #
    # ########################### WARNING: Do we want to ignore the procesing of some Pipelines? please, edit this code...
    """
    if example_id != 4: 
        index = index + 1
        continue
    """
    # ########################### WARNING
    #
    temp_context_args = context_args.copy()
    
    print('INFO: Run full Pipeline of Example {0:02d}...'.format(example_id))
    
    # Set user settings of Pipeline part about inputs (MinMaxScaler operations, what input static data to use...).
    temp_inputs_pipeline_as_text = inputs_pipeline_as_text
    for k,v in custom_settings.items():
        temp_inputs_pipeline_as_text = temp_inputs_pipeline_as_text.replace("$"+k, v)
    
    # Prepare settings.
    temp_settings = serialize_settings.copy()
    temp_settings['CurrentItem'] = 'example-{0:02d}'.format(example_id)
    temp_settings['CurrentDate'] = datetime.datetime.now().strftime('%Y%m%d')
    for k,v in settings_ob.items(): temp_settings[k] = v
    
    temp_pipeline = DataPipeline.concat_pipelines(temp_inputs_pipeline_as_text, output_pipeline_as_text)
    temp_pipeline = DataPipeline.concat_pipelines(temp_pipeline, serialize_pipeline_as_text)
    temp_pipeline = DataPipeline.prepare_pipeline(temp_pipeline, temp_settings)
    
    # Run the Pipeline.
    pipeline_ob = GeoDataPipeline()
    temp_df = pipeline_ob.run_from_string(temp_pipeline, factories={}, context_args=temp_context_args)
    pipeline_ob = None
    
    # Next example.
    index = index + 1
    
print('Ok!')


# COMMAND ----------

temp_settings

# COMMAND ----------

import glob
import os

spark.conf.set("spark.databricks.io.cache.enabled", "true")
serialize_settings = \
{
  "OutputFolder": "ML_Input"
}

# Load Datasets just serialized...
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


