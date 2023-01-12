# Databricks notebook source
"""
================================================================================
Workflow of automatic up-to-date of CLIMATE Datasets to Azure Blob Storages.

Arguments:
  + downloading_date: Date of CAMS Dataset to work with.
    This Date is used as reference to download the CLIMATE Datasets from AWS,
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

# DBTITLE 1,Main variables & constants

# Skip normal execution for debugging this Notebook.
DEBUGGING_NOTEBOOK = False

# Default delay in Days of "Today" Date (https://surfobs.climate.copernicus.eu/dataaccess/access_eobs_months.php).
DEFAULT_DAYS_DELAY = 1


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
dbutils.widgets.text(name='delay', defaultValue=str(DEFAULT_DAYS_DELAY), label='Delay of Date')
delay_of_date = dbutils.widgets.get('delay')
delay_of_date = int(delay_of_date) if delay_of_date is not None and len(str(delay_of_date)) > 0 else DEFAULT_DAYS_DELAY
dbutils.widgets.text(name='date', defaultValue='', label='Downloading Date')
downloading_date = dbutils.widgets.get('date')
if not downloading_date: downloading_date = (datetime.datetime.now() - datetime.timedelta(days=delay_of_date)).strftime('%Y-%m-%d')

# Preparing logging resources using the NotebookSingletonManager.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/notebookutils.py').read(), 'notebookutils.py', 'exec'))
notebook_mgr = NotebookSingletonManager(logging_path='/dbfs'+aq_climate_path, logging_mode='w')
#
print_message(logging.INFO, '## Starting process of "{}" Notebook...'.format(os.path.basename(notebook_mgr.notebook_path())))
print_message(logging.INFO, 'Processing CLIMATE Dataset:')
print_message(logging.INFO, ' + Downloading Date: {}'.format(downloading_date))
print_message(logging.INFO, ' + Delay of Date: {}'.format(delay_of_date))


# COMMAND ----------

# DBTITLE 1,Run child Notebooks

# Timeout in seconds of execution of child Notebooks.
CHILD_TIMEOUT = 60*60*24

print_message(logging.INFO, 'Starting execution of child Notebooks...')

try:
    for notebook_name in ['climate2azure', 'climate2aggregates', 'climate2aggrunique', 'climate2aggrunique_gf']:
        start_time = time.time()
        print_message(logging.INFO, 'Executing Notebook "{}"...'.format(notebook_name))
        
        # Run current child Notebook.
        if DEBUGGING_NOTEBOOK:
            print_message(logging.DEBUG, 'Skipping execution, we are debugging.', indent=1)
        else:
            result = dbutils.notebook.run(notebook_name, CHILD_TIMEOUT, {'date': downloading_date, 'delay': delay_of_date})
            print_message(logging.INFO, 'Result: {}'.format(result), indent=1)
            result = json.loads(result)
            
            if result.get('logging_file_name'):
                notebook_mgr.digest_logging_file(result.get('logging_file_name'))
            if result['status']!='SUCCESS': 
                raise Exception(result['message'])
                
        elapsed_time = time.time() - start_time
        elapsed_time = datetime.datetime.utcfromtimestamp(elapsed_time)
        elapsed_text = str(elapsed_time)
        elapsed_text = elapsed_text[elapsed_text.index(' ') + 1:]
        print_message(logging.INFO, 'Notebook successfully processed! Elapsed=[{0}]'.format(elapsed_text), indent=1)
        
except Exception as e:
    message = '{}\n{}'.format(str(e), traceback.format_exc())
    notebook_mgr.exit(status='ERROR', message=message, options={'date': downloading_date, 'delay': delay_of_date})  

print_message(logging.INFO, 'Child Notebooks successfully processed!')


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date, 'delay': delay_of_date})    
notebook_mgr = None

