# Databricks notebook source
"""
================================================================================
Join the CLIMATE Datasets in one unique Azure Blob storage.
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/131081
Author   : ahuarte@tracasa.es

================================================================================
"""

import sys
sys.path.append('/dbfs/FileStore/scripts/eea/databricks')

# Import EEA Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))

# Mount the 'AQ CLIMATE' Azure Blob Storage Container as a File System.
aq_climate_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-climate', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# COMMAND ----------

# DBTITLE 1,Main variables & constants

# Target root folder where saving the new CLIMATE Datasets with new Aggregates and transfomed to EPSG:3035.
AGGR_TARGET_FOLDER = aq_climate_path + '/Ensemble'

# Measure/Variables settings.
measures = {
  'TG': 'Daily mean temperature', 
  'TN': 'Daily minimum temperature', 
  'TX': 'Daily maximum temperature', 
  'RR': 'Daily precipitation sum', 
  'PP': 'Daily averaged sea level pressure', 
  'HU': 'Daily averaged relative humidity', 
  'QQ': 'Daily mean global radiation', 
}


# COMMAND ----------

# DBTITLE 1,Prepare DataFactory environment

import traceback
import json
import logging
import os
import datetime

# Create Widgets for leveraging parameters:
# https://docs.microsoft.com/es-es/azure/databricks/notebooks/widgets
dbutils.widgets.removeAll()
dbutils.widgets.text(name='date', defaultValue='', label='Downloading Date')
downloading_date = dbutils.widgets.get('date')
if not downloading_date: downloading_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_climate_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing aggregates:')
print_message(logging.INFO, ' + Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Variables:')
for key in measures: print_message(logging.INFO, '\t{}: {}'.format(key, measures[key]))


# COMMAND ----------

# DBTITLE 1,Joining all CLIMATE Datasets in an unique one

import pyspark.sql.functions as F

def parse_date_of_date(date: str):
    """
    Returns the (year, month, day, week) of the specified Date (yyyy-MM-dd).
    """
    y, m, d = date[0:4], date[5:7], date[8:10]
    date_ob = datetime.date(int(y), int(m), int(d))
    w = date_ob.strftime('%W')
    return date, y,m,d, w    

def get_output_file_name(root_folder: str, pollutant_key: str, operation: str, criteria: str, date: str):
    """
    Returns the Output file name of the new Aggregate CLIMATE Dataset.
    """
    date, y,m,d, w = parse_date_of_date(date)
    criteria = criteria.upper()
    
    subfolder = '{}/{}/'.format(pollutant_key, y)
    if criteria in ['MONTH','DAY']: subfolder += (m + '/')
    if criteria in ['WEEK']: subfolder += 'Weeks/'
    if criteria == 'YEAR' : file_name = 'CLIMATE_{}_{}_{}-XX-XX'.format(pollutant_key, operation.lower(), y)
    if criteria == 'MONTH': file_name = 'CLIMATE_{}_{}_{}-{}-XX'.format(pollutant_key, operation.lower(), y, m)
    if criteria == 'WEEK' : file_name = 'CLIMATE_{}_{}_{}-wk-{}'.format(pollutant_key, operation.lower(), y, w)
    if criteria == 'DAY'  : file_name = 'CLIMATE_{}_{}_{}-{}-{}'.format(pollutant_key, operation.lower(), y, m, d)    
    return '{}/{}{}.parquet'.format(root_folder, subfolder, file_name)

# Process for each Measure/Variable...
try:
    dataset = None
    
    # ... join current Dataset with previous one.
    for key in measures:
        measure = measures[key]
        
        print_message(logging.INFO, 'Processing Table of Measure/Variable "{}" ({})...'.format(key, downloading_date[0:4]))
        
        file_name = get_output_file_name(AGGR_TARGET_FOLDER, key, 'avg', 'year', downloading_date)
        temp_ob = spark.read.parquet(file_name)
        temo_ob = temp_ob.alias(key)
        
        if dataset is None:
            dataset = temp_ob
        else:
            temp_ob = temp_ob.withColumnRenamed('GridNum1km', 'GridNum1km_')
            temp_ob = temp_ob.withColumnRenamed('id', 'id_')
            temp_ob = temp_ob.withColumnRenamed('x', 'x_')
            temp_ob = temp_ob.withColumnRenamed('y', 'y_')
            temp_ob = temp_ob.withColumnRenamed('Year', 'Year_')
            dataset = dataset.join(temp_ob, dataset['GridNum1km']==temp_ob['GridNum1km_'], how='outer')
            dataset = dataset.withColumn('GridNum1km', F.coalesce('GridNum1km', 'GridNum1km_'))
            dataset = dataset.withColumn('id', F.coalesce('id', 'id_'))
            dataset = dataset.withColumn('x', F.coalesce('x', 'x_'))
            dataset = dataset.withColumn('y', F.coalesce('y', 'y_'))
            dataset = dataset.withColumn('Year', F.coalesce('Year', 'Year_'))
            dataset = dataset.drop('GridNum1km_')
            dataset = dataset.drop('id_')
            dataset = dataset.drop('x_')
            dataset = dataset.drop('y_')
            dataset = dataset.drop('Year_')
    
    # Sort Columns.
    columns = [c for c in dataset.columns if c.startswith('climate_')]
    dataset = dataset.select(['id', 'x', 'y', 'GridNum1km', 'Year'] + columns)
    
    # Write Join Dataset (Using Pandas, we prefer do not partitionate output files).
    file_name = '/dbfs' + get_output_file_name(AGGR_TARGET_FOLDER, 'ALL', 'avg', 'year', downloading_date)
    file_dir = os.path.dirname(file_name)
    os.makedirs(file_dir, exist_ok=True)
    
    print_message(logging.INFO, 'Writing JOIN Dataset "{}"...'.format(file_name))
    temp_pd = dataset.toPandas()
    temp_pd.to_parquet(file_name, compression='snappy', index=False)
    
    # display(dataset)
    # dataset2 = spark.read.parquet(file_name)
    # display(dataset2)
    
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date})
    
print_message(logging.INFO, 'Join successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date})    
notebook_mgr = None

