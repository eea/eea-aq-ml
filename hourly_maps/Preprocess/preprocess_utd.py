# Databricks notebook source
"""
================================================================================
Preprocess UTD data already in azure to include coordinate data, assign a Gridnum and store it as parquet.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157037
Author   : mgonzalez@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Mount the 'AQ-E' Azure Blob Storage Container as a File System.
aq_e_path = fsutils.mount_azure_container(
  storage_account_name = 'aqblobs', 
  container_name = 'airquality-e', 
  sas_key = 'sv=2021-08-06&st=2022-12-15T07%3A52%3A57Z&se=2030-08-23T07%3A08%3A00Z&sr=c&sp=rl&sig=qwYPZ0Kg11gPXLPfpTPfqotaRyrrwVMPF5X9HW5ADDw%3D'
)
# Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
aq_metadata = fsutils.mount_azure_container(
  storage_account_name = 'eeadmz1batchservice01', 
  container_name = 'aq-metadata', 
  sas_key = '?sv=2021-08-06&ss=btqf&srt=sco&st=2022-12-19T06%3A54%3A07Z&se=2030-07-20T05%3A54%3A00Z&sp=rl&sig=jKvbfV7zRREm64BvR8efB2R%2BeqRzsM0BT6NzRk0ElZU%3D'
)
# Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
aq_predictions_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-predictions', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)

context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

import os

import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import *

from datetime import datetime

# COMMAND ----------

# Pollutant settings.
pollutants = {
  'NO2': 'nitrogen_dioxide', 
  'O3': 'ozone',
  'PM10': 'particulate_matter_10um',
  'PM2.5': 'particulate_matter_2.5um',
  'SO2': 'sulphur_dioxide'
}

# COMMAND ----------

# Load list of required files

utd_files = pd.read_csv('/dbfs/'+aq_e_path+'/E2aInventory.csv.gz')
df_utd_files_list = spark.read.csv(aq_e_path+'/E2aInventory.csv.gz', header=True)

# COMMAND ----------

utd_files.display()

# COMMAND ----------

#df_utd_files_list = df_utd_files_list.withColumn('filename', F.regexp_replace('url', 'https://aqblobs.blob.core.windows.net/airquality-e','dbfs:'+aq_e_path))
utd_files['filename'] = utd_files['url'].str.replace('https://aqblobs.blob.core.windows.net/airquality-e','dbfs:'+aq_e_path)
utd_files['preprocess_time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#all_pollutants =  utd_files.drop_duplicates(subset = ["pollutant"])
utd_files = utd_files[utd_files['pollutant'].isin(pollutants.keys())]
utd_files = utd_files[utd_files['year']>2020]

#pollutants_after_drop = utd_files.drop_duplicates(subset = ["pollutant"])
df_utd_files_list = spark.createDataFrame(utd_files)
#all_pollutants.display()
#pollutants_after_drop.display()

# COMMAND ----------

#TODO: Filter files that have not been loaded
#utd_files.filename.values.tolist()
print("Files to load: {}".format(df_utd_files_list.count()))
print(df_utd_files_list.dropDuplicates(["pollutant"]).select("pollutant").collect())

# COMMAND ----------

df_utd_files = spark.read.csv(utd_files.filename.values.tolist(), header=True)\
    .withColumn('filename', F.input_file_name() )


# COMMAND ----------

df_utd_files.display()

# COMMAND ----------

df_utd_files_join = df_utd_files.join(df_utd_files_list, ['filename'])\
                            .select('country','station', 'spo', 'pollutant', 'year', 'month', 'size','url','datetime_begin','datetime_end','value_numeric', 'resulttime')\

df_utd_files_join.display()

# COMMAND ----------

df_station_infos = spark.createDataFrame(pd.read_csv('https://eeadmz1batchservice01.file.core.windows.net/aq-metadata/Station_ML.csv?sv=2021-08-06&st=2022-12-19T07%3A09%3A27Z&se=2030-12-20T07%3A09%3A00Z&sr=f&sp=r&sig=HtGT7n3U77yCR2BpCIuVJv8EjHDxDypHP8dZa9slDHQ%3D'))
df_station_infos.display()
    
  

# COMMAND ----------

final_df = df_utd_files_join.join(df_station_infos, (df_utd_files_join['station'] == df_station_infos['eucode']) & (df_utd_files_join['pollutant'] == df_station_infos['polu']), 'left')

# COMMAND ----------

final_df.display()

# COMMAND ----------

final_df.select('country','station','spo_norm','pollutant','year','month','size','url','datetime_begin','datetime_end','value_numeric','eucode','Lat','Lon')\
        .write.mode('overwrite').parquet(aq_predictions_path+'/ML_Input/HourlyMaps/hourlydata.parquet')

# COMMAND ----------

pipeline_as_text = """
{
  "Pipeline": [
    {
      "Type": "Dataset",
      "StorageAccount": "dis2datalake",
      "Container": "airquality-predictions",
      "Name": "MyTable",
      "File": "ML_Input/HourlyMaps/hourlydata.parquet"
    },
    
    // Create a Geometry from Longitude & Latitude attributes,
    {
      "Type": "SQL",
      "SQL": "SELECT *, st_geomFromText( concat('POINT(',string(Lon),' ',string(Lat),')') ) AS geometry_1 FROM $THIS"
    },
    
    // Transform EPSG:4326 Lon/Lat position to EPSG:3035 XY.
    {
      "Type": "SQL",
      "SQL": "SELECT *, st_transform(geometry_1, 'EPSG:4326', 'EPSG:3035') AS geometry_2 FROM $THIS"
    },
    {
      "Type": "SQL",
      "SQL": "SELECT *, st_x(geometry_2) AS X, st_y(geometry_2) AS Y FROM $THIS"
    },
      
    // Testing results of "ST_Transform" & "CalcGrid" methods.
    {
      "Type": "SQL",
      "SQL": "SELECT *, st_transform(geometry_2, 'EPSG:3035', 'EPSG:4258') AS geometry_3 FROM $THIS"
    },
    {
      "Type": "SQL",
      "SQL": "SELECT *, st_x(geometry_3) AS Longitude_new, st_y(geometry_3) AS Latitude_new FROM $THIS"
    },
    {
      "Type": "DerivedColumn",
      "Name": "MyTable_TEMP",
      "Columns": {
        "GridNum": "F.EEA_calcgridnum(F.col('X'), F.col('Y'))"
      }
    },
    {
      "Type": "SQL",
      "SQL": "SELECT *, 
              (GridNum & CAST(1152921500311879680 AS BIGINT)) AS GridNum10km, 
              (GridNum & CAST(1152921504590069760 AS BIGINT)) AS GridNum1km, 
              (GridNum & CAST(1152921504606781440 AS BIGINT)) AS GridNum100m 
              FROM MyTable_TEMP;"
    },
    {
      "Type": "DerivedColumn",
      "Columns": {
        "X2": "F.EEA_gridid2laea_x(F.col('GridNum'))",
        "Y2": "F.EEA_gridid2laea_y(F.col('GridNum'))"
      }
    }
  ]
}
"""

# Run the Pipeline.
pipeline_ob = GeoDataPipeline()
result_df = pipeline_ob.run_from_string(pipeline_as_text, factories={}, context_args=context_args.copy())
pipeline_ob = None

for pollutant in pollutants:
  result_df.select('country','station','spo_norm','pollutant','year','month','size','url','datetime_begin','datetime_end','value_numeric','Lat','Lon','GridNum','GridNum1km')\
          .filter(result_df.pollutant == pollutant)\
          .write.mode("overwrite").parquet(aq_predictions_path+'/ML_Input/HourlyMaps/UTD_hourlyData_{}.parquet'.format(pollutant), compression='snappy')

# Show results.
# print('Table Count={}'.format(result_df.count()))
#display(result_df)

# COMMAND ----------

utd_files.to_csv('/dbfs'+aq_predictions_path+'/ML_Input/HourlyMaps/LoadRecord.csv.gz', 
           index=False, 
           compression="gzip")

# COMMAND ----------

df = spark.read.parquet(aq_predictions_path+'/ML_Input/HourlyMaps/hourlydata.parquet')
df.display()

# COMMAND ----------


