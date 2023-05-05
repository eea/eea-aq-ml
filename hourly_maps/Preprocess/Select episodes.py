# Databricks notebook source
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType
from pyspark.sql.functions import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aq_e1a_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-e1a', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)

aq_predictions_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-predictions', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)

# Esquema de datos
schema = StructType([
    StructField('ts', IntegerType(), True),
    StructField('fk_samplingpoint', IntegerType(), True),
    StructField('fk_property', IntegerType(), True),
    StructField('time_diff', IntegerType(), True),
    StructField('datetime_begin', TimestampType(), True),
    StructField('datetime_end', TimestampType(), True),
    StructField('value_numeric', FloatType(), True),
    StructField('fk_validity', IntegerType(), True),
    StructField('fk_verification', IntegerType(), True),
    StructField('resulttime', TimestampType(), True),
    StructField('fk_unit', IntegerType(), True),
    StructField('fk_aggregationtype', IntegerType(), True),
    StructField('fk_trace', IntegerType(), True),
    StructField('datacapture', FloatType(), True),
    StructField('fk_aggregationtypeOriginal', IntegerType(), True)
])

# DataFrame de estaciones
stations = spark.read.option('header', 'true').option('delimiter', ',').csv('/mnt/dis2datalake_airquality-predictions/StaticData/Station_ML.csv') \
    .withColumn("pk_spo", col("pk_spo").cast("int")) \
    .withColumn("pk_polu", col("pk_polu").cast("int"))
stations = stations.drop('spo', 'spo_norm')
#display(stations)
# Parquet statistics    
statistics = spark.read.parquet(aq_predictions_path+'/AQeReporting/AirQualityStatistics.parquet').select("AirQualityStationEoICode", "AirPollutantGroup", "ReportingYear", "GridNum1km")
staticdata = spark.read.parquet(aq_predictions_path+'/StaticData/JoinedGS_and_NOGS_AGs_final.parquet/').select("GridNum1km", 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

# COMMAND ----------

# DBTITLE 1,2019 Episodes
# MAGIC %md
# MAGIC - Jan 19-25: PM10/PM2.5 episode in southern PL as well as CZ and SK (also elevated NO2 levels), moving west to Germany on Jan 23-24; not described in Copernicus reports but visible on EEA viewers,
# MAGIC - Feb 17-28: PM10/PM2.5 episode in PL at the beginning of the period and in northern and western Europe (UK, NL, Scandinavia, north of FR) at the end of the period, also - elevated NO2 levels with similar pattern as PMs; not described in Copernicus reports but visible on EEA viewers,
# MAGIC - Jun 24 - Jul 7: huge O3 episode stretching across most of Europe, described in Copernicus reports and clear in EEA episode viewer,
# MAGIC - Jul 23-27: clear O3 episode mostly in central western Europe, described in Copernicus reports and clear in EEA episode viewer,

# COMMAND ----------

countries = ['PL', 'CZ', 'SK', 'DE']
 
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2019/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')
    

    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-01-19' and '2019-01-25' and pk_polu in (192)")      
    target = aq_predictions_path + '/ML_Input/episodes/201901_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-01-19' and '2019-01-25' and pk_polu in (246)")      
    target = aq_predictions_path + '/ML_Input/episodes/201901_PM25.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-01-19' and '2019-01-25' and pk_polu in (423)")      
    target = aq_predictions_path + '/ML_Input/episodes/201901_NO2.parquet'    
    
    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

countries = ['PL', 'GB', 'NL', 'FR', 'SE', 'NO', 'FI']
 
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2019/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')
    
    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-02-17' and '2019-02-28' and pk_polu in (192)")      
    target = aq_predictions_path + '/ML_Input/episodes/201902_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-02-17' and '2019-02-28' and pk_polu in (246)")      
    target = aq_predictions_path + '/ML_Input/episodes/201902_PM25.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-02-17' and '2019-02-28' and pk_polu in (423)")      
    target = aq_predictions_path + '/ML_Input/episodes/201902_NO2.parquet'    
    
    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

countries = ['AL', 'AD', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LT', 'LU', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'MK', 'TR', 'GB']
#countries = ['GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LT', 'LU', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'MK', 'TR', 'GB', 'GE', 'UA']
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2019/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')
    
    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-06-24' and '2019-07-07' and pk_polu = 352")      
    target = aq_predictions_path + '/ML_Input/episodes/201906_O3.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

#countries = ["AT", "BE", "CH", "DE", "DK", "FI", "FR", "GB", "IE", "IS", "IT", "LU", "NL", "NO", "PT", "SE", "ES"]
countries = ["LU", "NL", "NO", "PT", "SE", "ES"]
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2019/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')
    
    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2019-07-23' and '2019-07-27' and pk_polu = 352")      
    target = aq_predictions_path + '/ML_Input/episodes/201907_O3.parquet'    

    df_ts.write.mode('append').parquet(target)

# COMMAND ----------

# DBTITLE 1,2020 Episodes
# MAGIC %md
# MAGIC - Jan 7-16: PM10 episode in eastern and south-eastern Europe (also some days with elevated NO2), described in Copernicus reports and clear in EEA episode viewer,
# MAGIC - Jan 22-27: PM10 episode (also reflected a bit in PM2.5 values) passing from west to east, affecting mostly ES, south FR and IT, later passing through Balkans and also appearing in PL in the end of period, not noticed in Copernicus reports but visible on EEA viewers,
# MAGIC - Mar 27-30: serious PM10 episode stretching from AT, through Balkans to BG, affecting also IT, not noticed in Copernicus reports but clearly visible on EEA viewers,
# MAGIC - Aug 6-13: significant O3 episode stretching over north-west Europe (BE, NL, south UK, north DE, north FR), not noticed in Copernicus reports but clearly visible on EEA episode viewer,
# MAGIC - Oct 1-4: PM10 episode over Scandinavia and Baltic countries, described in Copernicus reports and clear in EEA episode viewer,

# COMMAND ----------

#countries = ["AL", "BA", "BG", "HR", "CY", "CZ", "EE", "GR", "HU", "XK", "LV", "LT", "MK", "PL", "RO", "RS", "SK", "SI", "TR", "UA"]
countries = ["PL", "RO", "RS", "SK", "SI", "TR", "UA"]
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2020/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between  '2020-01-07' and '2020-01-16' and pk_polu = 192")      
    target = aq_predictions_path + '/ML_Input/episodes/202001_PM10_south_eastern_Europe.parquet'    

    
    df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between  '2020-01-07' and '2020-01-16' and pk_polu = 423")      
    target = aq_predictions_path + '/ML_Input/episodes/202001_NO2_south_eastern_Europe.parquet'    

    df_ts.write.mode('append').parquet(target)

# COMMAND ----------

countries = ['ES', 'FR', 'IT', 'PL', 'AL', 'BA', 'BG', 'HR', 'GR', 'XK', 'ME', 'MK', 'RO', 'RS', 'SI']
 
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2020/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2020-01-22' and '2020-01-27' and pk_polu = 192")      
    target = aq_predictions_path + '/ML_Input/episodes/202001_PM10_Balkans_west_Europe.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2020-01-22' and '2020-01-27' and pk_polu = 246")      
    target = aq_predictions_path + '/ML_Input/episodes/202001_PM25_Balkans_west_Europe.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

countries = ["AT", "SI", "HR", "BA", "RS", "XK", "ME", "AL", "MK", "GR", "IT", "BG"]
 
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2020/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2020-03-27' and '2020-03-30' and pk_polu = 192")  
    target = aq_predictions_path + '/ML_Input/episodes/202003_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)


# COMMAND ----------

countries = ["BE", "NL", "GB", "DK", "FR"]
 
for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2020/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2020-08-06' and '2020-08-13' and pk_polu = 352")  
    target = aq_predictions_path + '/ML_Input/episodes/202008_O3.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

countries = ["FI", "SE", "NO", "DK", "IS", "EE", "LV", "LT"]

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2020/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2020-08-06' and '2020-08-13' and pk_polu = 192")  
    target = aq_predictions_path + '/ML_Input/episodes/202010_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

# DBTITLE 1,2021 episodes
# MAGIC %md
# MAGIC - Feb 22-28: Saharan dust crossing Europe, described in Copernicus reports and clearly visible in EEA viewers (some elevated levels of PM10 measured in different parts of Europe until about Mar 7), also elevated NO2 levels, especially at the beginning of the period, Baltic countries, PL and RO,
# MAGIC - Jun 14-21: significant O3 episode in central and northern Europe, described in Copernicus reports and clear in EEA episode viewer,

# COMMAND ----------

countries = ['AL', 'AD', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LT', 'LU', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'MK', 'TR', 'GB', 'GE', 'UA']

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2021/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2021-02-22' and '2021-02-28' and pk_polu = 192")  
    target = aq_predictions_path + '/ML_Input/episodes/202102_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2021-02-22' and '2021-02-28' and pk_polu = 423")  
    target = aq_predictions_path + '/ML_Input/episodes/202102_NO2.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)


# COMMAND ----------

countries = ["AT", "BE", "CH", "DE", "DK", "FI", "FR", "GB", "IE", "IS", "LU", "NL", "NO", "SE"]

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2021/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2021-06-14' and '2021-06-21' and pk_polu = 352")  
    target = aq_predictions_path + '/ML_Input/episodes/202106_O3.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)


# COMMAND ----------

# DBTITLE 1,2022 episodes
# MAGIC %md
# MAGIC - Mar 15-26: big Saharan dust event, described in Copernicus reports and visible on EEA viewers, the event was mostly visible in Iberian peninsula on Mar 15-16, but I would include time until Mar 26, as there are many occurrences of high PM10 levels across all Europe following the initial intrusion; also - elevated NO2 levels, at the beginning of the period in FI, and then in PL and DE,
# MAGIC - Jul 17-25: big O3 episode crossing Europe from west to east, described in Copernicus reports and clear in EEA episode viewer,
# MAGIC - Aug 9-15: significant O3 episode in western Europe (mostly north ES, FR, BE, NL, south UK), not noticed in Copernicus reports but clearly visible on EEA episode viewer.

# COMMAND ----------

countries = ["PT", "ES", "AD", "FI", "PL", "DE"]

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2022/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2022-03-15' and '2022-03-26' and pk_polu = 352")  
    target = aq_predictions_path + '/ML_Input/episodes/202203_NO2.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)


    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2022-03-15' and '2022-03-26' and pk_polu = 192")  
    target = aq_predictions_path + '/ML_Input/episodes/202203_PM10.parquet'    

    if i == 0:
      df_ts.write.parquet(target)
    else:
      df_ts.write.mode('append').parquet(target)

# COMMAND ----------

#countries = ["PT", "ES", "FR", "BE", "NL", "LU", "DE", "AT", "CH", "IT", "SI", "HR", "BA", "ME", 
countries = ["AL", "MK", "GR", "XK", "BG", "RO", "HU", "SK", "CZ", "PL", "LT", "LV", "EE", "FI", "SE", "NO", "IS"]

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2022/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
        (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

    df_join3.createOrReplaceTempView('table')

    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2022-07-17' and '2022-07-25' and pk_polu = 352")  
    target = aq_predictions_path + '/ML_Input/episodes/202207_O3.parquet'    

    
    df_ts.write.mode('append').parquet(target)


# COMMAND ----------

countries = ["PT", "ES", "FR", "BE", "NL", "LU", "IE", "GB", "AD"]

for i, country in enumerate(countries):
    file_location = f'wasbs://airquality-e@aqblobs.blob.core.windows.net/{country}/2022/*/*.csv.gz'
    temp_df = spark.read.option("header", "true").schema(schema).csv(file_location)
    temp_df = temp_df.withColumn("weekday", dayofweek("datetime_end"))
    #display(temp_df)

    df_join = temp_df.join(stations, (temp_df["fk_samplingpoint"] == stations["pk_spo"]) & (temp_df["fk_property"] == stations["pk_polu"]))
    
    df_join2 = df_join.join(statistics, 
        (df_join["eucode"] == statistics["AirQualityStationEoICode"]) & 
        (df_join["polu"] == statistics["AirPollutantGroup"]) & 
        (year(df_join["datetime_end"]) == statistics["ReportingYear"])
    )
    df_join2 = df_join2.drop("AirPollutantGroup", 'ts')

    df_join3 = df_join2.alias("a").join(staticdata.alias("b"), 
            (df_join2["GridNum1km"] == staticdata["GridNum1km"])).select('a.*', 'AreaHa', 'avg_smallwoody_mean_sur', 'avg_imp2015_mean_sur', 'avg_hrlgrass', 'avg_eudem', 'sum_clc18_111_mean_sur', 'sum_clc18_121_mean_sur', 'sum_clc18_141_mean_sur', 'sum_clc18_122_mean_sur', 'sum_clc18_211', 'sum_clc18_311_mean_sur', 'sum_clc18_312_mean_sur', 'sum_clc18_313_mean_sur', 'sum_elevbreak_Inlands', 'sum_elevbreak_Mountains', 'sum_urbdegree_30', 'sum_urbdegree_11_mean_sur', 'sum_urbdegree_12_mean_sur', 'sum_urbdegree_13_mean_sur', 'sum_envzones_LUS', 'sum_envzones_ATC', 'carbonseqcapacity_mean_sur', 'weight_urb', 'pop2018', 'windspeed_mean_sur', 'droughtimp_mean_sur', 'weight_tr_mean_sur', 'weight_tr', 'ecoclimaregions_28', 'biogeoregions_6')

        df_join3.createOrReplaceTempView('table')

    # Selección de los datos anteriores al 1 de enero de 2022 y escritura en el directorio especificado
    df_ts = spark.sql("SELECT * FROM table WHERE datetime_begin between '2022-08-09' and '2022-08-15' and pk_polu = 352")  

    target = aq_predictions_path + '/ML_Input/episodes/202208_O3.parquet'
    
    if i == 0:
      pivoted_df.write.parquet(target)
    else:
      pivoted_df.write.mode('append').parquet(target)

# COMMAND ----------

staticdata = spark.read.parquet(aq_predictions_path+'/StaticData/JoinedGS_and_NOGS_AGs_final.parquet/')

# COMMAND ----------

temp_df = spark.read.parquet('/mnt/dis2datalake_airquality-predictions/ML_Input/episodes/202208_O3.parquet')
df_join = temp_df.alias("a").join(staticdata.alias("b"), (temp_df["GridNum1km"] == staticdata["GridNum1km"])).select('time_diff','datetime_begin','datetime_end','fk_validity','fk_verification','fk_aggregationtype','fk_trace','weekday','eucode','AirQualityStationType','AirQualityStationArea','Altitude','CityPopulation','X','Y','a.GridNum1km')

target = aq_predictions_path + '/ML_Input/episodes/202208_O3.parquet'
df_join.show()
df_join.write.parquet(target)
