# Databricks notebook source


# COMMAND ----------

# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML predictions
# MAGIC >  **NOTE:** The predictions are made based on fit on training data
# MAGIC <br/>

# COMMAND ----------

# DATA LOADING MODIFIED (NO CUT OFF on REF) TO ALLOW MORE TRAINING DATA FOR SOMO10
# Data loading, split into train, cv and joining reference data from ETC maps
# Function to: 
# - read the data for training and validation
# - prepare data sets for training, cv and validation
# - add reference data from ETC interpolated maps to each of the data set for comparison purposes

def loadAQMLDataValTrain(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput):

#   import pyspark.sql.functions as F
#   import pandas as pd
#   import numpy as np 
#   from sklearn import model_selection
  
  example = 'example'
  
  if pollutant == 'NO2': 
    example = 'example-03'
    #refcol = 'no2_avg'
    cols = no2cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'PM10': 
    example = 'example-01'
    #refcol = 'pm10_avg'
    cols = pm10cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'PM25': 
    example = 'example-02'  
    #refcol = 'pm25_avg'
    cols = pm25cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'O3_SOMO35': 
    example = 'example-04'  
    #refcol = 'o3_somo35'
    cols = o3somo35cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'O3_SOMO10': 
    example = 'example-05'  
    #refcol = 'o3_somo10'
    cols = o3somo10cols # used only for feature post-selection based on correlation and VIF
  # Compiling path to files

  trainfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'training_input_' + trainset + '_' + pollutant + '_' + train_start_year + '-' + train_end_year + '.parquet'
  valfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'validation_input_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '.parquet'

  # Reading ML input files into spark data frames

  dftraining = spark.read.parquet(trainfile).select(cols) # used only for feature post-selection based on correlation and VIF
  dfvalidation = spark.read.parquet(valfile).select(cols) # used only for feature post-selection based on correlation and VIF

  # Reading reference data (ETC interpolated maps) into spark data frames
  
#   dfref = spark.read.parquet(aq_predictions_path + '/ETC_maps/aq_grids_year_all_with2020.parquet')
#   dfref_train = dfref.filter((dfref.Year >= train_start_year) & (dfref.Year <= train_end_year)).select(dfref.GridNum1km, dfref.Year, dfref[refcol])
#   dfref_val = dfref.filter((dfref.Year >= predval_start_year) & (dfref.Year <= predval_end_year)).select(dfref.GridNum1km, dfref.Year, dfref[refcol])

  # Joining reference data with the ML input data sets (training and validation)

#   dfref_train = dfref_train.withColumnRenamed("GridNum1km","GridNum1km_2").withColumnRenamed("Year","Year_2")
#   dfref_val = dfref_val.withColumnRenamed("GridNum1km","GridNum1km_2").withColumnRenamed("Year","Year_2")
  
# This is the MODIFICATION to allow more data for training of SOMO10 (where ref is missing for several years)  
#   dfref_train = dfref_train.filter(dfref_train[refcol] > 0)
#   dfref_val = dfref_val.filter(dfref_val[refcol] > 0)

  # here also fraction sampling is used
  dftrain = dftraining#.sample(frac)#.join(dfref_train,(dftraining.GridNum1km == dfref_train.GridNum1km_2) & (dftraining.Year == dfref_train.Year_2),how="inner")
  dfval = dfvalidation#.sample(frac)#.join(dfref_val,(dfvalidation.GridNum1km == dfref_val.GridNum1km_2) & (dfvalidation.Year == dfref_val.Year_2),how="inner")

  # Preparing data set as pandas data frame
  # The target data sets will include the reference data so that following splits (e.g. training and cv) remain consistent

  #target feature
  target = trainset + '_' + pollutant
  #refcol = refcol

  #training & cv data
  dftrainx = dftrain.drop(target,'GridNum1km','Year','GridNum1km_2','Year_2','AreaHa')#refcol,
  dftrainy = dftrain.select(target) # keeping both target and reference attribute #refcol
  X = dftrainx.toPandas()
  y = dftrainy.toPandas()

  #final validation data
  dfvalx = dfval.drop(target,'GridNum1km','Year','GridNum1km_2','Year_2','AreaHa')# refcol,
  dfvaly = dfval.select(target) # keeping both target and reference attribute # refcol
  X_val = dfvalx.toPandas()
  y_val = dfvaly.toPandas()

  # Splitting training data sets into training and cv parts 

  np.random.seed(12)
  X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, train_size=0.9)

  # Splitting the target-reference data sets for training, cv and validation into separate data sets

#   y_train = y_train[[target]] # target for training error checks
#   #y_train_ref = y_withref_train[[refcol]] # reference for training error checks

#   y_cv = y_cv[[target]]  # target for cv error checks
#   #y_cv_ref = y_withref_cv[[refcol]]  # reference for cv error checks

#   y_val = y_val[[target]]  # target for validation error checks
#   y_val_ref = y_withref_val[[refcol]]  # reference for validation error checks
  
#   df = dftrain.drop(refcol,'GridNum1km','Year','GridNum1km_2','Year_2').toPandas()
#   corr = df.corrwith(df[target])
#   dfcorr = corr.to_frame()
#   dfcorr = dfcorr.drop(target)
    
  return X_train, X_cv, X_val, y_train, y_cv, y_val


# COMMAND ----------

# AQ prediction ML data loading, solution with function
  
# # Define main path to files
# path = aq_predictions_path + '/ML_Input/'
# #print(path)

# # Set parameters to read the input such as years, pollutant, input data set, etc.
# train_start_year = '2015'
# train_end_year = '2019'
# predval_start_year = '2020'
# predval_end_year = '2020'
# trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
# pollutant = 'NO2' #,'PM10','PM25','O3_SOMO35','NO2'
# dateOfInput = '20220720'

# big data set requires 11.0 ML (includes Apache Spark 3.3.0, GPU, Scala 2.12) to read into pandas; to be used only after training
def loadAQMLPredData(path,predval_start_year,predval_end_year,pollutant,dateOfInput):

#   import pyspark.sql.functions as F
#   import pandas as pd
#   import numpy as np 
#   from sklearn import model_selection
  
  example = 'example'
  
  if pollutant == 'NO2': 
    example = 'example-03'
    refcol = 'no2_avg'
    cols = no2cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'PM10': 
    example = 'example-01'
    refcol = 'pm10_avg'
    cols = pm10cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'PM25': 
    example = 'example-02'  
    refcol = 'pm25_avg'
    cols = pm25cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'O3_SOMO35': 
    example = 'example-04'  
    refcol = 'o3_somo35'
    cols = o3somo35cols # used only for feature post-selection based on correlation and VIF
  elif  pollutant == 'O3_SOMO10': 
    example = 'example-05'  
    refcol = 'o3_somo10'
    cols = o3somo10cols # used only for feature post-selection based on correlation and VIF
  # Compiling path to files
  
  predfile = path + example + '_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '/' + dateOfInput + '/' + 'prediction_input_' + pollutant + '_' + predval_start_year + '-' + predval_end_year + '.parquet'

  # Reading ML input files into spark data frames
  
  dfpred = spark.read.parquet(predfile).select(cols[1:]) # used only for feature post-selection based on correlation and VIF
  # Preparing data set as pandas data frame

  X_pred = dfpred.toPandas() # big data set requires 11.0 ML (includes Apache Spark 3.3.0, GPU, Scala 2.12) to read into pandas; to be used only after training
  
  return X_pred # big data set requires 11.0 ML (includes Apache Spark 3.3.0, GPU, Scala 2.12) to read into pandas; to be used only after training

#X_pred = loadAQMLPredData(path,predval_start_year,predval_end_year,pollutant,dateOfInput)

# COMMAND ----------

# ML predictions

# Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutants = ['PM10','PM25','O3_SOMO35','O3_SOMO10','NO2']
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0

for pollutant in pollutants:
  
  print('Pollutant: ' + pollutant)

  # Load data, fit model and prep data for error calculations

  X_train, X_cv, X_val, y_train, y_cv, y_val = loadAQMLDataValTrain(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput)
  lr,md,gm,al,la,sub = bestModelParameters(pollutant)
  model = XGBRegressor(learning_rate = lr, max_depth = md, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)
  model.fit(X_train,y_train)

  X_pred = loadAQMLPredData(path,predval_start_year,predval_end_year,pollutant,dateOfInput)
  predictions = model.predict(X_pred.iloc[:, 3:])

  X_maps = X_pred.iloc[:, :2]
  preds = pd.DataFrame({pollutant: predictions})
  ML_maps = pd.concat([X_maps,preds], axis=1)
  ML_maps_spark = spark.createDataFrame(ML_maps) 

  file_name = '/dbfs' + aq_predictions_path+"/ML_Output/" + pollutant + "_" + predval_start_year + "_" + predval_end_year + "_" + dateOfInput+"_maps.parquet"
  ML_maps_spark.toPandas().to_parquet(file_name, compression='snappy')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")
