# Databricks notebook source
"""
================================================================================
Apply GapFilling algorithm on EEA CLIMATE Datasets.

The source CLIMATE Datasets have missing values in several zone of EEA envelope.
This Notebook fills these gaps using LinearRegression.

SEE:
  https://adb-2318633810729807.7.azuredatabricks.net/?o=2318633810729807#notebook/177339712966379

NOTE:
  Original code developed by Artur Bernard Gsella (Artur.Gsella@eea.europa.eu).
  
================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/131080
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

# DBTITLE 1,Prepare DataFactory environment

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
print_message(logging.INFO, ' + Variables: ALL')


# COMMAND ----------

# DBTITLE 1,Define GapFilling algorithm

#
# Implementations of GapFilling, by Artur Bernard Gsella
#

def gap_filling(input_object, module_args, context_args):
    """
    Fill Gaps of one Attribute of current Dataset.
    """
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    
    method = module_args.get('Method', 'LinearRegression') #-> Artur's GapFilling implementation, by default, more else?
    attribute = module_args.get('Attribute')
    columns = module_args.get('Columns')
    eval_alg = module_args.get('Eval', False)
    eval_weights = module_args.get('EvalWeights', [0.85, 0.15])
    
    # print('Getting Valid/Null Rows...')
    
    # Split Dataframe in Valid/Null Parts.
    regr_df = input_object.filter(input_object[attribute].isNotNull())
    pred_df = input_object.filter(input_object[attribute].isNull())
    # print('not-null DF={}'.format(regr_df.count()))
    # print('yes-null DF={}'.format(pred_df.count()))
    
    # Configure settings of algorithm.
    if method == 'Unknown':
        raise Exception('Unsupported method "{}" of GapFilling.'.format(method))
    else:
        assembler = VectorAssembler(inputCols=columns, outputCol='features')
        lr = LinearRegression(labelCol=attribute)
        pipeline_ob = Pipeline(stages=[assembler, lr])
        eval_ob = RegressionEvaluator(labelCol=attribute, predictionCol='prediction')
        
    # print('Splitting because We have to eval current algorithm...')
    
    if eval_alg:
        train_df, test_df = regr_df.randomSplit(weights=eval_weights, seed=42)
        # print('train DF={}'.format(train_df.count()))
        # print('test DF={}'.format(test_df.count()))
    else:
        train_df, test_df = regr_df, None
    
    # print('Fit...')
    
    model_ob = pipeline_ob.fit(train_df)
    
    # print('Transform...')
    
    if eval_alg:
        eval_df = model_ob.transform(test_df)
        rmse = eval_ob.evaluate(eval_df, {eval_ob.metricName: 'rmse'})
        r2   = eval_ob.evaluate(eval_df, {eval_ob.metricName: 'r2'})
        mae  = eval_ob.evaluate(eval_df, {eval_ob.metricName: 'mae'})
        print_message(logging.INFO, 'Evaluation Metrics for "{}": RMSE={:.4f}, R-squared={:.4f}, MAE={:.4f}.'.format(attribute, rmse, r2, mae))
        module_args['EvalMetrics'] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
    else:
        print_message(logging.INFO, 'Not Evaluating Metrics for "{}".'.format(attribute))
        
    gapfilled_df = model_ob.transform(pred_df)
    
    # print('gapfilled DF={}'.format(gapfilled_df.count()))
    
    gapfilled_df = gapfilled_df.drop(attribute)
    gapfilled_df = gapfilled_df.withColumnRenamed('prediction', attribute)
    gapfilled_df = gapfilled_df.select(regr_df.columns)
    
    # print('Union')
    
    total_df = regr_df.union(gapfilled_df)
    return total_df


# COMMAND ----------

# DBTITLE 1,Define and Run the Pipeline

# Import EEA Dataflow engine.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/datapipeline.py').read(), 'datapipeline.py', 'exec'))
#
# This Pipeline engine supports expressions using whatever of available "Spark SQL built-in" Functions.
# See:
#   https://spark.apache.org/docs/latest/api/sql/index.html
# Example:
#   F.lit(2.5) + F.col('my_value'); F.coalesce('GridNum1km', 'CLIM_GridNum1km'); ...
#

# Import & register 'SQL AQ CalcGrid' functions.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/calcgrid.py').read(), 'calcgrid.py', 'exec'))
#
# You can invoke CalcGridFunctions in expressions using 'F.EEA_xxx()' (g.e. 'F.EEA_calcgridnum(...)') 
# or use them registering the functions as UDF and invoke the returned wrappers.
#
# from pyspark.sql.types import LongType
# calcgridnum_udf   = spark.udf.register('calcgridnum',   CalcGridFunctions.calcgridnum,   LongType())
# gridid2laea_x_udf = spark.udf.register('gridid2laea_x', CalcGridFunctions.gridid2laea_x, LongType())
# gridid2laea_y_udf = spark.udf.register('gridid2laea_y', CalcGridFunctions.gridid2laea_y, LongType())
#


# COMMAND ----------


# Dynamic Parameters.
measure = 'ALL'   #-> Options: [TG, TN, TX, RR, PP, HU, QQ]
year = downloading_date[0:4]
eval_GF = 'false' #-> true/false: Evaluate and print metrics of GapFilling algorithm on variables of CLIMATIC data

# Pipeline declaration with the Dataflow to perform (As STRING).
template_of_pipeline_as_text = """
{
  "Pipeline": [
    # This is provider of XY+EU-DEM data for applying GapFilling on CLIMATIC data.
    {
      "Type": "Dataset",
      
      "SQL": "SELECT DISTINCT GridNum & CAST(1152921504590069760 AS bigint) AS GridNum1km, AVG(eudem) AS eudem1km 
              FROM ml_input_from_jedi.aq_con_100 
              WHERE eudem IS NOT NULL 
              GROUP BY GridNum & CAST(1152921504590069760 AS bigint)",
              
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
          "Name": "EUDEM_1000_FOR_GF"
        }
      ]
    },
    
    # This is 1km grid of CLIMATIC data for one specific Year (+ GapFilling).
    {
      "Type": "Dataset",
      "Name": "CLIMATE_ORIG",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-climate",
      "File": "Ensemble/$Measure/$Year/CLIMATE_$Measure_avg_$Year-XX-XX.parquet",
      
      "Pipeline": [
        {
          "Type": "Select",
          "Name": "CLIMATE",
          "Columns": [ "~id", "~x", "~y" ]
        },
        {
          "Type": "Join",
          "Left": "EUDEM_1000_FOR_GF",
          "LeftKey": "GridNum1km",
          "Right": "CLIMATE",
          "RightKey": "GridNum1km",
          "JoinType": "left"
        },
        {
          "Type": "Cache",
          "Name": "CLIMATE_BEFORE_GF"
        },
        
        # Perform the GapFilling on all CLIMATIC variables.
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_TG",           
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_TN",           
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_TX", 
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_RR",           
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_PP",           
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_HU", 
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        {
          "Type": "GapFilling",
          "Method": "LinearRegression",
          "Attribute": "climate_QQ", 
          "Columns": [ "x", "y", "eudem1km" ], 
          "Eval": $Eval_GF, 
          "EvalWeights": [0.85, 0.15]
        },
        
        {
          "Type": "Select",
          "Name": "CLIMATE",
          "Columns": [ "~eudem1km" ]
        },
        {
          "Type": "Cache",
          "Name": "CLIMATE_AFTER_GF"
        }
      ]
    },
    
    # Write new Dataset to external Parquet file.
    {
      "Type": "Output",
      "Name": "Result",
      "OutputEngine": "Pandas",
      
      "StorageAccount": "dis2datalake",
      "Container": "airquality-climate",
      "File": "Ensemble/$Measure/$Year/CLIMATE_$Measure_avg+gapfilling_$Year-XX-XX.parquet"
    }
  ]
}
"""

pipeline_as_text = template_of_pipeline_as_text \
  .replace('$Year', str(year)) \
  .replace('$Measure', str(measure)) \
  .replace('$Eval_GF', eval_GF)

# Pipeline declaration with the Dataflow to perform.
# print(pipeline_as_text)


# COMMAND ----------


# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# Run the Pipeline!
pipeline_ob = DataPipeline()
result_df = pipeline_ob.run_from_string(pipeline_as_text, factories={'GapFilling': gap_filling}, context_args=context_args)
stack_ob = context_args.get('Stack')
pipeline_ob = None

# Show the Result of the DataFlow!
# print('Table Count: {}'.format(result_df.count()))
# display(result_df)

print_message(logging.INFO, 'Ok!')


# COMMAND ----------

# DBTITLE 1,Show Results in a Map (Disabled, only for testing, we do not want to show anything in OFFLINE mode)

import folium

# Import EEA [Geo]Dataflow engine (It provides the 'dataframe_to_raster' function).
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/geodatapipeline.py').read(), 'geodatapipeline.py', 'exec'))
# Import EEA Map Databricks utils.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/maputils.py').read(), 'maputils.py', 'exec'))

# Show the CLIM attribute of original CLIMATE Dataset in a Map.
stack_ob = context_args.get('Stack')
climate_df = stack_ob.get('CLIMATE_ORIG')

# Show the CLIM attribute of the original CLIMATE Dataset in a Map.
#my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': climate_df, 'attributes': ['climate_QQ']})
#my_map


# COMMAND ----------


stack_ob = context_args.get('Stack')
climate_df = stack_ob.get('CLIMATE_BEFORE_GF')

# Show the CLIM attribute without GapFilling in a Map.
#my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': climate_df, 'attributes': ['climate_QQ']})
#my_map


# COMMAND ----------


stack_ob = context_args.get('Stack')
climate_df = stack_ob.get('CLIMATE_AFTER_GF')

# Show the CLIM attribute after GapFilling in a Map.
#my_map = FoliumUtils.create_folium_map_from_table(map_content_args={'table': climate_df, 'attributes': ['climate_QQ']})
#my_map


# COMMAND ----------

# DBTITLE 1,Finishing Job

# Notify SUCCESS and Exit.
notebook_mgr.exit(status='SUCCESS', message='', options={'date': downloading_date})    
notebook_mgr = None

