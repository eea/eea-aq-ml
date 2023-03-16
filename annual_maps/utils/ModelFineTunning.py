# Databricks notebook source
# Import EEA AQ Azure platform tools on Databricks.
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/eeadatabricksutils.py').read(), 'eeadatabricksutils.py', 'exec'))
exec(compile(eea_databricks_framework_initialize(), '', 'exec'))

# Initialize a Context Dictionary with useful data.
context_args = {
  'SAS_KEY': 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1. Model Fine Tune

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### A) Hyperparameter tuning using random search

# COMMAND ----------

# # Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutants = ['O3_SOMO35','O3_SOMO10'] #['PM10','PM25','O3_SOMO35','O3_SOMO10','NO2']
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0

# Set of initial bounds for model hyperparameters

# learning_rate: 1,2,3,4,5 (dividend), 10,100,1000 (divisor)
# max_depth: 3,4,5,6,7,8,9
# gamma: 0,1,2,3,5 (dividend), 1,10,100 (divisor)
# reg_alpha: 0,1,2,3,5 (dividend), 1,10 (divisor)
# reg_lambda: 1,2,3,5,7 (dividend), 1,10 (divisor)
# subsample: 0.5, 0.7, 0.9, 1.0

# learning_rate: 1,2,3,4,5 (dividend), 10,100,1000 (divisor)
lrs1 = [1,2,3,4,5]
lrs2 = [10,100] # reduce to 10,100
# max_depth: 3,4,5,6,7,8,9
maxds = [3,4,5,6] # reduce to 3-6
# gamma: 0,1,2,3,5 (dividend), 1,10,100 (divisor)
gms1 = [0,1,2,3,5]
gms2 = [1,10]
# reg_alpha: 0,1,2,3,5 (dividend), 1,10 (divisor)
als1 = [0,1,2,3,5]
als2 = [1,10]
# reg_lambda: 1,2,3,5,7 (dividend), 1,10 (divisor)
las1 = [1,2,3,5,7]
las2 = [1] #[1,10 ]
# subsample: 0.5, 0.7, 0.9, 1.0
subs = [0.7,0.8, 0.9, 1.0] # reduce to 0.7 - 1.0

# Number of samples for each hyper

lri = 3
maxdi = 4
gmi = 1
ali = 2
lai = 3
subi = 3



# COMMAND ----------

# Hyperparameter tuning using random search
# AQ ML data loading, solution with function
  


for pollutant in pollutants:
    X_train, X_cv, X_val, y_train_ref, y_train, y_cv, y_cv_ref, y_val, y_val_ref = loadAQMLData(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput)
    print('Pollutant: ',pollutant)

    diff = 1000000
    #diffref = 1000000
    
    for ilri in range (lri):
      lr1 = random.sample(lrs1,1)
      lr2 = random.sample(lrs2,1)
      lr = lr1[0]*1.0/lr2[0]
      for imaxdi in range (maxdi):
        maxd = random.sample(maxds,1)[0]
        for igmi in range (gmi):
          gm1 = random.sample(gms1,1)
          gm2 = random.sample(gms2,1)
          gm = gm1[0]*1.0/gm2[0]
          for iali in range (ali):
            al1 = random.sample(als1,1)
            al2 = random.sample(als2,1)
            al = al1[0]*1.0/al2[0]  
            for ilai in range (lai):
              la1 = random.sample(las1,1)
              la2 = random.sample(las2,1)
              la = la1[0]*1.0/la2[0]
              for isubi in range (subi):
                sub = random.sample(subs,1)[0]

                # Hypers in the model and model fit

                model = XGBRegressor(learning_rate = lr, max_depth = maxd, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)
                model.fit(X_train, y_train)

                # Getting RMSE

                rmse_train_ref, rmse_train = validateModel(model,X_train,y_train,y_train_ref)
                rmse_cv_ref, rmse_cv = validateModel(model,X_cv,y_cv,y_cv_ref)

                diffrmse = rmse_cv - rmse_train
                rmse_ref = (rmse_train_ref + rmse_cv_ref)/2
                diffrmse_ref = rmse_cv - rmse_ref

                if (diffrmse + diffrmse_ref) < diff: # and diffrmse_ref < diffref:
                  diff = (diffrmse + diffrmse_ref)
                  #diffref = diffrmse_ref
                  minrmse_train = rmse_train
                  minrmse_cv = rmse_cv
                  minrmse_ref = rmse_ref
                  minlr = lr
                  minmaxd = maxd
                  mingm = gm
                  minal = al
                  minla = la
                  minsub = sub
                  minmodel = model
                  print('Learning rate: ', minlr, ', max depth: ', minmaxd,', gamma: ', mingm,', alpha: ', minal,', lambda: ', minla,', subsample: ', minsub)
                  print('RMSE training: ', minrmse_train, ', RMSE CV: ', minrmse_cv,', RMSE ref: ', minrmse_ref)

    if pollutant == 'NO2': 
      no2_best_rmse_train = minrmse_train
      no2_best_rmse_cv = minrmse_cv
      no2_rmse_ref = minrmse_ref
      no2_best_model = minmodel
      
      no2_best_lr = minlr
      no2_best_maxd = minmaxd
      no2_best_gm = mingm
      no2_best_al = minal
      no2_best_la = minla
      no2_best_sub = minsub

    elif  pollutant == 'PM10': 
      pm10_best_rmse_train = minrmse_train
      pm10_best_rmse_cv = minrmse_cv
      pm10_rmse_ref = minrmse_ref
      pm10_best_model = minmodel
      
      pm10_best_lr = minlr
      pm10_best_maxd = minmaxd
      pm10_best_gm = mingm
      pm10_best_al = minal
      pm10_best_la = minla
      pm10_best_sub = minsub
      
    elif  pollutant == 'PM25': 
      pm25_best_rmse_train = minrmse_train
      pm25_best_rmse_cv = minrmse_cv
      pm25_rmse_ref = minrmse_ref
      pm25_best_model = minmodel
      
      pm25_best_lr = minlr
      pm25_best_maxd = minmaxd
      pm25_best_gm = mingm
      pm25_best_al = minal
      pm25_best_la = minla
      pm25_best_sub = minsub
      
    elif  pollutant == 'O3_SOMO35': 
      o3_somo35_best_rmse_train = minrmse_train
      o3_somo35_best_rmse_cv = minrmse_cv
      o3_somo35_rmse_ref = minrmse_ref
      o3_somo35_best_model = minmodel
      
      o3_somo35_best_lr = minlr
      o3_somo35_best_maxd = minmaxd
      o3_somo35_best_gm = mingm
      o3_somo35_best_al = minal
      o3_somo35_best_la = minla
      o3_somo35_best_sub = minsub


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### B) Hyperparameter tuning using grid search

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 2. ML model validation
# MAGIC >  **NOTE:** The model validation includes data loading and split, learning curve on data amount, error distributions, error variance, MQI, RMSE and correlation for validation data set
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### A) Learning curve on data amount vs training and cv errors

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ** Pasar todo esto a una función **

# COMMAND ----------

# Generating input for learning curve on data amount vs training and cv errors: first view on bias & variance

# Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutant = 'O3_SOMO10' #,'PM10','PM25','O3_SOMO35','O3_SOMO10','NO2'
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0

i = 0.01

# COMMAND ----------



# Create the empty initial pandas DataFrame for storing the score changes with growing amount of data
dfscores = pd.DataFrame({'rmse_train': [0], 'rmse_cv': [0], 'rmse_ref': [0], 'data':[0]} )

lr,md,gm,al,la,sub = bestModelParameters(pollutant)
model = XGBRegressor(learning_rate = lr, max_depth = md, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)

while i <= 1.0:

  #load data using loadAQMLData function with sample fraction defined by i
  X_train, X_cv, X_val, y_train_ref, y_train, y_cv, y_cv_ref, y_val, y_val_ref = loadAQMLData(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput,frac=i)
  
  # NN: fitting model to data
#   model.fit(X_train,y_train,
#     epochs=50)
  
  # SGD or XGB: fitting model to data
  model.fit(X_train,y_train)
  
  #getting metrics for training data and reference data (reference y vs y_train)
  rmse_train_ref, rmse_train = validateModel(model,X_train,y_train,y_train_ref)
  #getting metrics for cv data and reference data (reference y vs y_cv)
  rmse_cv_ref, rmse_cv = validateModel(model,X_cv,y_cv,y_cv_ref)
  rmse_ref = (rmse_train_ref + rmse_cv_ref)/2
    
  dfscores = dfscores.append(pd.DataFrame({'rmse_train': [rmse_train], 'rmse_cv': [rmse_cv], 'rmse_ref': [rmse_ref], 'data':[i * 100]} ))
  print('Calculations for : ' + str(i * 100) + ' % of data completed.')
  
  if i == 0.01:
    i += 0.09
  else:
    i += 0.1
    
print('Calculations completed.')

# COMMAND ----------

# Generating plot for learning curve on data amount vs training and cv errors: first view on bias & variance

import plotly.graph_objects as go

figure = go.Figure()
traces = []

# traces.append(go.Scatter(x = fechas, y = UTD_values, name = 'UTD',mode='lines'))
traces.append(go.Scatter(x = dfscores["data"], y = dfscores["rmse_train"], name = 'training error',mode='lines'))
traces.append(go.Scatter(x = dfscores["data"], y = dfscores["rmse_cv"], name = 'cv error',mode='lines'))
traces.append(go.Scatter(x = dfscores["data"], y = dfscores["rmse_ref"], name = 'ref error',mode='lines'))

for t in traces: figure.add_trace(t)
figure.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### B) Error analysis (2014 - 2020): ML predictions vs Reference ETC maps on validation data sets

# COMMAND ----------

# Generating input for error distributions and MQI

# Define main path to files
path = aq_predictions_path + '/ML_Input/'
#print(path)

# Set parameters to read the input such as years, pollutant, input data set, etc.
train_start_year = '2014'
train_end_year = '2020'
predval_start_year = '2014'
predval_end_year = '2021'
trainset = 'eRep' # 'e1b' -- modelled data, 'eRep' -- measurement data
pollutant = 'O3_SOMO10' #,'PM10','PM25','O3_SOMO35','O3_SOMO10','NO2'
dateOfInput = '20220826'
frac = 1.0 # this is fraction of training data sampling for evaluation purpose, default is 1.0

# COMMAND ----------


# Create the empty initial pandas DataFrame for storing the errors and reference errors
dferrors = pd.DataFrame({'urb': [0], 'error_val': [0], 'error_ref': [0], 'aq': [0], 'ml': [0], 'ref': [0]} )

# Get the urbanisation degree
dfvalidation = loadAQMLValDataForAreas(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput)

dfvalidation = dfvalidation.withColumn(
    'urb',
    F.when((F.col("sum_urbdegree_11") > 0), 'rural')\
    .when((F.col("sum_urbdegree_12") > 0), 'rural')\
    .when((F.col("sum_urbdegree_13") > 0), 'rural')\
    .when((F.col("sum_urbdegree_21") > 0), 'suburban')\
    .when((F.col("sum_urbdegree_22") > 0), 'suburban')\
    .when((F.col("sum_urbdegree_30") > 0), 'urban')\
    .otherwise('other')
)

dfvalidation = dfvalidation.drop(*filter(lambda col: 'urbdegree' in col, dfvalidation.columns))
dfurb = dfvalidation.toPandas()
urb = dfurb.to_numpy()
urb_rav = urb.ravel()

# Load data, fit model and prep data for error calculations

X_train, X_cv, X_val, y_train_ref, y_train, y_cv, y_cv_ref, y_val, y_val_ref = loadAQMLData(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput)
lr,md,gm,al,la,sub = bestModelParameters(pollutant)
model = XGBRegressor(learning_rate = lr, max_depth = md, gamma = gm, reg_alpha = al, reg_lambda = la, subsample = sub)
model.fit(X_train,y_train)
y_pred_val = model.predict(X_val)

y_np = y_val.to_numpy() # has to be numpy vector to use ravel() in feature selection libraries
y_rav = y_np.ravel() #

y_np_ref = y_val_ref.to_numpy() # has to be numpy vector to use ravel() in feature selection libraries
y_rav_ref = y_np_ref.ravel() #

n = y_val.shape[0]

for i in range(n):
  error_val = sqrt((y_pred_val[i] - y_rav[i])*(y_pred_val[i] - y_rav[i]))
  error_ref = sqrt((y_rav_ref[i] - y_rav[i])*(y_rav_ref[i] - y_rav[i]))
  urb = urb_rav[i]
  aq = y_rav[i]
  ml = y_pred_val[i]
  ref = y_rav_ref[i]
  dferrors = dferrors.append(pd.DataFrame({'urb': [urb], 'error_val': [error_val], 'error_ref': [error_ref], 'aq': [aq], 'ml': [ml], 'ref': [ref]} ))

# rmse_val_ref, rmse_val = validateModel(model,X_val,y_val,y_val_ref)
# rmse_val_ref, rmse_val

dferrors_urban=dferrors[dferrors["urb"] == 'urban'] 
dferrors_suburban=dferrors[dferrors["urb"] == 'suburban'] 
dferrors_rural=dferrors[dferrors["urb"] == 'rural'] 
dferrors_other=dferrors[dferrors["urb"] == 'other'] 


# COMMAND ----------


def modelQuality(pollutant,y_pred,y):
  
  import numpy as np
  
  if pollutant == 'NO2': 
    URV95r = 0.24
    RV = 200.00
    alfa = 0.20
    Np = 5.20
    Nnp = 5.50
    
  elif  pollutant == 'PM10': 
    URV95r = 0.28
    RV = 50.00
    alfa = 0.13
    Np = 30.00
    Nnp = 0.25

  elif  pollutant == 'PM25': 
    URV95r = 0.36
    RV = 25.00
    alfa = 0.30
    Np = 30.00
    Nnp = 0.25
    
  elif  pollutant == 'O3': 
    URV95r = 0.18
    RV = 120.00
    alfa = 0.79
    Np = 11.00
    Nnp = 3.00
       
  
  Uncertainty = URV95r*np.sqrt(((1-np.square(alfa))*np.square(y)/Np)+(np.square(alfa)*np.square(RV)/Nnp))
  Divid = abs(y-y_pred)
  
  mqi = Divid/(2*Uncertainty)
  
  return mqi
  


# mqi = modelQuality(pollutant,y_rav_ref,y_rav)
# MQI = np.percentile(mqi, 90)

def validateModelTrainVal(model,X,y,y_ref):
  
#   import mlflow
#   from sklearn.metrics import mean_squared_error
#   from numpy import sqrt
  
  y_np = y.to_numpy() # has to be numpy vector to use ravel() in feature selection libraries
  y_rav = y_np.ravel() #

  rmse_ref = sqrt(mean_squared_error(y_rav, y_ref, squared=True)) # reference rmse
  y_pred = model.predict(X)
  rmse = sqrt(mean_squared_error(y_rav, y_pred, squared=True)) #
  
  corr_ref = np.corrcoef(y_rav, y_ref, rowvar=False)
  corr = np.corrcoef(y_rav, y_pred, rowvar=False)
 
  return corr_ref, corr, rmse_ref, rmse

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Error distributions for ML predictions vs Reference ETC maps

# COMMAND ----------

import matplotlib.pyplot as plt
import plotly
import plotly.express as px

# COMMAND ----------

# ERROR DISTRIBUTIONS FOR ML PREDICTIONS VS REFERENCE ETC MAPS   --> pasar a una funcion



figure, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
dferrors[["error_val","error_ref"]].plot(ax=axes[0], kind='kde').set_title('All areas, ML Error Variance:' + str('{:.2f}'.format(dferrors.var()['error_val']))+', Ref Error Variance:' + str('{:.2f}'.format(dferrors.var()['error_ref'])),color='b',fontsize=10)
dferrors_urban[["error_val","error_ref"]].plot(ax=axes[1], kind='kde').set_title('Urban areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_urban.var()['error_val']))+', Ref Error Variance:' + str('{:.2f}'.format(dferrors_urban.var()['error_ref'])),color='b',fontsize=10)
dferrors_suburban[["error_val","error_ref"]].plot(ax=axes[2], kind='kde').set_title('Suburban areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_suburban.var()['error_val']))+', Ref Error Variance:' + str('{:.2f}'.format(dferrors_suburban.var()['error_ref'])),color='b',fontsize=10)
dferrors_rural[["error_val","error_ref"]].plot(ax=axes[3], kind='kde').set_title('Rural areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_rural.var()['error_val']))+', Ref Error Variance:' + str('{:.2f}'.format(dferrors_rural.var()['error_ref'])),color='b',fontsize=10)

# COMMAND ----------

# MQI --> pasar a una funcion



print(pollutant,'MQI, ML:','{:.2f}'.format(np.percentile(modelQuality(pollutant,y_pred_val,y_rav), 90)),', Ref:','{:.2f}'.format(np.percentile(modelQuality(pollutant,y_rav_ref,y_rav), 90)))
corr_ref, corr, rmse_ref, rmse = validateModelTrainVal(model,X_val,y_val,y_rav_ref)
print(pollutant,'Correlation with AQ data, ML:','{:.2f}'.format(corr[0,1]),', Ref:','{:.2f}'.format(corr_ref[0,1]))
print(pollutant,'RMSE, ML:','{:.2f}'.format(rmse),', Ref:','{:.2f}'.format(rmse_ref))


# COMMAND ----------

# SCATTERS FOR ML PREDICTIONS VS REFERENCE ETC MAPS  --> pasar a una funcion




figure, axes = plt.subplots(2, 4, figsize=(24, 6), sharey=True, sharex=True)

dferrors.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[0,0], legend = 0).set_title('Overall Scatter For ML',color='b',fontsize=10)
dferrors_urban.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[0,1], legend = 0).set_title('Urban Scatter For ML',color='b',fontsize=10)
dferrors_suburban.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[0,2], legend = 0).set_title('Suburban Scatter For ML',color='b',fontsize=10)
dferrors_rural.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[0,3], legend = 0).set_title('Rural Scatter For ML',color='b',fontsize=10)
dferrors.plot.scatter(x='aq', y='ref', c = 'error_ref', colormap = 'viridis', ax=axes[1,0], legend = 0).set_title('Overall Scatter For ETC maps',color='b',fontsize=10)
dferrors_urban.plot.scatter(x='aq', y='ref', c = 'error_ref', colormap = 'viridis', ax=axes[1,1], legend = 0).set_title('Urban Scatter For ETC maps',color='b',fontsize=10)
dferrors_suburban.plot.scatter(x='aq', y='ref', c = 'error_ref', colormap = 'viridis', ax=axes[1,2], legend = 0).set_title('Suburban Scatter For ETC maps',color='b',fontsize=10)
dferrors_rural.plot.scatter(x='aq', y='ref', c = 'error_ref', colormap = 'viridis', ax=axes[1,3], legend = 0).set_title('Rural Scatter For ETC maps',color='b',fontsize=10)

# COMMAND ----------

# CORRELATIONS AND RMSE  --> Pasar a una funcion


corr_ref, corr, rmse_ref, rmse = validateModelTrainVal(model,X_val,y_val,y_rav_ref)
print(pollutant,'Correlation with AQ data, ML:','{:.2f}'.format(corr[0,1]),', Ref:','{:.2f}'.format(corr_ref[0,1]))
print(pollutant,'RMSE, ML:','{:.2f}'.format(rmse),', Ref:','{:.2f}'.format(rmse_ref))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Error analysis 2021: ML predictions in Year without ETC reference on validation data sets

# COMMAND ----------

# MODIFIED (NO CUT OFF on REF) TO ALLOW VALIDATION OF YEARS WITHOUT ETC MAPS
# Data loading, split into train, cv and joining reference data from ETC maps
# Function to: 
# - read the data for training and validation
# - prepare data sets for training, cv and validation
# - add reference data from ETC interpolated maps to each of the data set for comparison purposes

def loadAQMLDataValTrainYear(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput,year):

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
  dfval = dfvalidation.filter((dfvalidation.Year == year))#.sample(frac)#.join(dfref_val,(dfvalidation.GridNum1km == dfref_val.GridNum1km_2) & (dfvalidation.Year == dfref_val.Year_2),how="inner")

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
  X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, train_size=0.7)

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




# MODIFIED (NO CUT OFF on REF) TO ALLOW VALIDATION OF YEARS WITHOUT ETC MAPS
# Data loading, split into train, cv and joining reference data from ETC maps
# Function to: 
# - read the data for training and validation
# - prepare data sets for training, cv and validation
# - add reference data from ETC interpolated maps to each of the data set for comparison purposes

def loadAQMLDataValTrainYear(path,train_start_year,train_end_year,predval_start_year,predval_end_year,trainset,pollutant,dateOfInput,year):

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
  dfval = dfvalidation.filter((dfvalidation.Year == year))#.sample(frac)#.join(dfref_val,(dfvalidation.GridNum1km == dfref_val.GridNum1km_2) & (dfvalidation.Year == dfref_val.Year_2),how="inner")

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
  X_train, X_cv, y_train, y_cv = model_selection.train_test_split(X, y, train_size=0.7)

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

import matplotlib.pyplot as plt
import plotly
import plotly.express as px


# COMMAND ----------

# Error Distributions for ML predictions in Year without ETC reference on Validation data sets     --> Pasar a una funcion



figure, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
dferrors[["error_val"]].plot(ax=axes[0], kind='kde').set_title('All areas, ML Error Variance:' + str('{:.2f}'.format(dferrors.var()['error_val'])),color='b',fontsize=10)
dferrors_urban[["error_val"]].plot(ax=axes[1], kind='kde').set_title('Urban areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_urban.var()['error_val'])),color='b',fontsize=10)
dferrors_suburban[["error_val"]].plot(ax=axes[2], kind='kde').set_title('Suburban areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_suburban.var()['error_val'])),color='b',fontsize=10)
dferrors_rural[["error_val"]].plot(ax=axes[3], kind='kde').set_title('Rural areas, ML Error Variance:' + str('{:.2f}'.format(dferrors_rural.var()['error_val'])),color='b',fontsize=10)

# COMMAND ----------

# MQI, Correlation and RMSE in Year without ETC reference on Validation data sets      --> Pasar a una funcion





print(pollutant,'MQI for ML:','{:.2f}'.format(np.percentile(modelQuality(pollutant,y_pred_val,y_rav), 90)))

corr,rmse = validateModelPredVal(model,X_val,y_val)
print(pollutant,'Correlation with AQ data:','{:.2f}'.format(corr[0,1]))
print(pollutant,'RMSE for ML:','{:.2f}'.format(rmse))

# COMMAND ----------

#  scatters in Year without ETC reference on Validation data sets      --> Pasar a una funcion







figure, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True, sharex=True)

dferrors.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[0], legend = 0).set_title('Overall Scatter For ML',color='b',fontsize=10)
dferrors_urban.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[1], legend = 0).set_title('Urban Scatter For ML',color='b',fontsize=10)
dferrors_suburban.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[2], legend = 0).set_title('Suburban Scatter For ML',color='b',fontsize=10)
dferrors_rural.plot.scatter(x='aq', y='ml', c = 'error_val', colormap = 'viridis', ax=axes[3], legend = 0).set_title('Rural Scatter For ML',color='b',fontsize=10)

# COMMAND ----------

#  values in grids for the Year without ETC reference on Validation data sets      --> Pasar a una funcion





y_pred = pd.DataFrame(y_pred_val)

results = pd.concat([y_val.reset_index(drop=True),y_pred.reset_index(drop=True)], axis=1)
results.columns = ['AQ','ML']
fig = px.line(results)
fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


