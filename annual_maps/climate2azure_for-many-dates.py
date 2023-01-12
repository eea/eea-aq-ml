# Databricks notebook source
"""
================================================================================
Workflow of automatic up-to-date of CLIMATE Datasets to Azure Blob Storages.

Arguments:
  + downloading_date: Date of CAMS Dataset to work with.
    This Date is used as reference to download the CLIMATE Datastet from AWS,
    aggregate some statistical data (Mean,...) applying several Date grouping
    (Year, Week, Month, Day) and finally transform them to the EPSG:3035 SRS 
    1x1km GRID.
    
    If the Date is not specified, each subprocess work with "Today".

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
# NOTE:
# We can't use a simple import statement because of there is a bug in Databricks, 
# It seems that 'dbutils.fs' can't be accessed by local python files.

# Mount the 'AQ CLIMATE' Azure Blob Storage Container as a File System.
aq_climate_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-climate', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


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
dbutils.widgets.text(name='date1', defaultValue='', label='Date 1')
start_date = dbutils.widgets.get('date1')
if not start_date: start_date = (datetime.datetime.now() - datetime.timedelta(days=0)).strftime('%Y-%m-%d')
dbutils.widgets.text(name='date2', defaultValue='', label='Date 2')
final_date = dbutils.widgets.get('date2')
if not final_date: final_date = (datetime.datetime.now() - datetime.timedelta(days=0)).strftime('%Y-%m-%d')

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_climate_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing CAMS Dataset:')
print_message(logging.INFO, ' + Start Date: {}'.format(start_date))
print_message(logging.INFO, ' + Final Date: {}'.format(final_date))


# COMMAND ----------

# DBTITLE 1,Run child Notebooks

GROUP_BY_YEAR = True
temp_dicts = dict()

# Timeout in seconds of execution of child Notebooks.
CHILD_TIMEOUT = 60*60*24

print_message(logging.INFO, 'Starting execution of child Notebooks...')

try:
    #or notebook_name in ['climate2aggregates']:
    for notebook_name in ['climate2aggrunique_gf']:
        start_time = time.time()
        print_message(logging.INFO, 'Executing Notebook "{}"...'.format(notebook_name))
        
        # Run current child Notebook.
        start_date_ob = datetime.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
        final_date_ob = datetime.date(int(final_date[0:4]), int(final_date[5:7]), int(final_date[8:10]))
        delta_date_td = final_date_ob - start_date_ob
        date_count = 0
        fail_count = 0
        
        for i in range(delta_date_td.days + 1):
            date = str(start_date_ob + datetime.timedelta(days=i))
            
            if GROUP_BY_YEAR:
                y = date[0:4]
                if temp_dicts.get(y): continue
                temp_dicts[y] = 1
            
            print_message(logging.INFO, '> Date: {} ({})'.format(date, date_count), indent=1)
            
            result = dbutils.notebook.run(notebook_name, CHILD_TIMEOUT, {'date': date})
            result = json.loads(result)
            
            if result['status']!='SUCCESS':
                print_message(logging.ERROR, 'Date: {}, Message={}'.format(date, result['message']), indent=1)
                fail_count += 1
            
            date_count += 1
            
        elapsed_time = time.time() - start_time
        elapsed_time = datetime.datetime.utcfromtimestamp(elapsed_time)
        elapsed_text = str(elapsed_time)
        elapsed_text = elapsed_text[elapsed_text.index(' ') + 1:]
        print_message(logging.INFO, 'Notebook successfully processed! (DateCount={}, FailCount={}), Elapsed=[{}]'.format(date_count, fail_count, elapsed_text), indent=1)
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'start_date': start_date, 'final_date': final_date})  

print_message(logging.INFO, 'Child Notebooks successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'start_date': start_date, 'final_date': final_date})    
notebook_mgr = None

