# Databricks notebook source
# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import io
import pyarrow as pa
import pyarrow.parquet as pq

from azure.storage.blob import BlobServiceClient
from azure.storage.fileshare import ShareFileClient

exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

aq_predictions_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-predictions', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)
csv_df = spark.read.option("header","true").csv(aq_predictions_path+'/StaticData/Station_ML.csv')

stations = csv_df.toPandas()
stations['pk_spo'] = stations['pk_spo'].astype(int)

## Conectar con el servicio de almacenamiento de blobs de Azure
connect_str = "DefaultEndpointsProtocol=https;AccountName=aqblobs;AccountKey=k7Bu0gblGDqqj2waqK9hBu2heB4UD5KX5lGA1RpID2GoUatyLQBVWoXlSwIdlTE1x25L3RIB9ZsLRygeOW95cw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

#
## Obtener una lista de los nombres de los archivos CSV en el contenedor de Azure Storage
container_client = blob_service_client.get_container_client("airquality-e")
blob_list = container_client.list_blobs()
csv_files = [blob.name for blob in blob_list if blob.name.endswith(".gz") and ("/2022/" not in blob.name and "/2023/" not in blob.name)]
#csv_files = [blob.name for blob in blob_list if blob.name.endswith(".gz") and ("/2015/" in blob.name and "AD/" in blob.name)]
# Leer los datos de los archivos CSV y concatenarlos en un solo DataFrame
target = '/dbfs{}/ML_Input/E1a.parquet'.format(aq_predictions_path)

for i, csv_file in enumerate(csv_files):
    blob_client = container_client.get_blob_client(csv_file)
    blob_data = blob_client.download_blob().readall()
    dataframe = pd.read_csv(io.BytesIO(blob_data), header=0, compression='gzip', delimiter=",")
    merged_df = dataframe.merge(stations, left_on='fk_samplingpoint', right_on='pk_spo')
    merged_df['value_numeric'] = merged_df['value_numeric'].astype(float)
    merged_df['Lat'] = merged_df['Lat'].astype(str)
    merged_df['Lon'] = merged_df['Lon'].astype(str)
    table = pa.Table.from_pandas(merged_df)
    print(merged_df)
    if i == 0:
      pqwriter = pq.ParquetWriter(target, table.schema)            
    pqwriter.write_table(table)

# close the parquet writer
if pqwriter:
    pqwriter.close()

# COMMAND ----------

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import io
import pyarrow as pa
import pyarrow.parquet as pq

from azure.storage.blob import BlobServiceClient
from azure.storage.fileshare import ShareFileClient

exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

aq_predictions_path = fsutils.mount_azure_container(
  storage_account_name = 'dis2datalake', 
  container_name = 'airquality-predictions', 
  sas_key = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
)


# Mount the 'AQ Predictions ML' Azure Blob Storage Container as a File System.
aq_data = fsutils.mount_azure_container(
  storage_account_name = 'aqblobs', 
  container_name = 'airquality-e', 
  sas_key = 'k7Bu0gblGDqqj2waqK9hBu2heB4UD5KX5lGA1RpID2GoUatyLQBVWoXlSwIdlTE1x25L3RIB9ZsLRygeOW95cw=='
)

csv_df = spark.read.option("header","true").csv(aq_predictions_path+'/StaticData/Station_ML.csv')

stations = csv_df.toPandas()
stations['pk_spo'] = stations['pk_spo'].astype(int)
## Conectar con el servicio de almacenamiento de blobs de Azure
connect_str = "DefaultEndpointsProtocol=https;AccountName=aqblobs;AccountKey=k7Bu0gblGDqqj2waqK9hBu2heB4UD5KX5lGA1RpID2GoUatyLQBVWoXlSwIdlTE1x25L3RIB9ZsLRygeOW95cw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
#
## Obtener una lista de los nombres de los archivos CSV en el contenedor de Azure Storage
container_client = blob_service_client.get_container_client("airquality-e")
# Leer los datos de los archivos CSV y concatenarlos en un solo DataFrame
target = '/dbfs{}/ML_Input/E1a.parquet'.format(aq_predictions_path)
#dataframe = pd.read_csv(io.BytesIO(blob_data), compression='gzip', delimiter=",", error_bad_lines=False)
#merged_df = pd.concat([dataframe, stations], keys=['fk_samplingpoint','pk_spo'], join='inner')

blob_client = container_client.get_blob_client('AD/2015/AD0942A/E_SPO-AD0942A-0001_SO2_2015_01.csv.gz')

blob_data = blob_client.download_blob().readall()
dataframe = pd.read_csv(io.BytesIO(blob_data), header=0, compression='gzip', delimiter=",")


merged_df = dataframe.merge(stations, left_on='fk_samplingpoint', right_on='pk_spo')
table = pa.Table.from_pandas(merged_df)
pqwriter = pq.ParquetWriter(target, table.schema)            
pqwriter.write_table(table)

# close the parquet writer
if pqwriter:
    pqwriter.close()


# COMMAND ----------

## Importar librerías necesarias
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import glob
#import io
#from azure.storage.blob import BlobServiceClient
#
### Conectar con el servicio de almacenamiento de blobs de Azure
#connect_str = "DefaultEndpointsProtocol=https;AccountName=aqblobs;AccountKey=k7Bu0gblGDqqj2waqK9hBu2heB4UD5KX5lGA1RpID2GoUatyLQBVWoXlSwIdlTE1x25L3RIB9ZsLRygeOW95cw==;EndpointSuffix=core.windows.net"
#blob_service_client = BlobServiceClient.from_connection_string(connect_str)
#
### Obtener una lista de los nombres de los archivos CSV en el contenedor de Azure Storage
#container_client = blob_service_client.get_container_client("airquality-e")
#blob_list = container_client.list_blobs()
#csv_files = [blob.name for blob in blob_list if blob.name.endswith(".gz") and "/2022/" not in blob.name and "/2023/" not in blob.name]
#
### Leer los datos de los archivos CSV y guardar en un archivo Parquet
#blob_client = container_client.get_blob_client("Reanalysis/E1a/E1a.parquet")
#df = pd.concat([pd.read_csv(io.BytesIO(container_client.get_blob_client(csv_file).download_blob().readall()), delimiter=",", compression='gzip', error_bad_lines=False) for csv_file in csv_files])
#df.to_parquet(io.BytesIO(), index=False)
#blob_client.upload_blob(data=io.BytesIO().getvalue(), overwrite=True)
#
