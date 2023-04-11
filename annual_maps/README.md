%md

# Air Quality Annual maps
********

**The scope of this project is to generate datasets containing polluntats' data so we can calculate an air-quality index (AQI) for specific regions. These scripts will collect and preprocess raw data so we can train a ML model to forecast its future values.**

Within this repo we have scripts we will need to execute manually so we can evaluate outputs and then decide if anything at config files should be modified. Preprocess data and predictions are the ones which should be automated for execution once we have an optimized pretrained ML algorithm.

<br />

![my_test_image](/files/shared_uploads/iborra@discomap.eea.europa.eu/AQAnnualMaps_drawio.png)

## Files
********
The project is structured into three main folders targetting specific objectives:

**1. Config folder:** Contains the only file we should need to edit (+widgets) every time we are willing to perform/test a different execution.
  
`ConfigFile.py:` Notebook to configure executions for the train/predict/FeatureSelection scripts by using two main classes:

<br />

>- *DataHandlerConfig:* anything related with the data we are willing to use: container storing data, paths to input/output data, cols we will use as features for our model.

>- *MLModelsConfig:* anything related with the ML model we are willing to use: ML algorithm, test/optimized parameters.


**2. Utils folder:** Contains generic functions used along the different notebooks and other scripts which should be executed out of the pipeline scope. 

`Lib.py:` Notebook containning core functions used along the project. Herein classes will inherit from config file.

<br />

>- *DataHandler:* class inheriting from DataHandlerConfig and containing core functionalities to read, store data into different formats (csv, parquet, tiff) and find duplicates.

>- *MLDatahandler:* class inheriting from DataHandler and focused on collecting/storing data with ML purposes.

>- *PreProcessDataHandler:* class inheriting from DataHandler and focused on collecting/storing data with preprocessing purposes.

>- *MLWorker:* class inheriting from MLModelsConfig and focused on ML purposes (split dataset, train/load model, evaluate model performance.).

`FeatureSelection.py:` Notebook to collect raw data and measure its variance inflation factor (VIF) + correlation so we can have a deeper understanding of our dataset features. VIF will provide us an idea of the correlation between our independent variables (multi/collinearity) and correlation will help us understanding the relationship between our features and the target. 
As a rule of thumb, we should select those features with VIF values lower than 5 and high correlation with the target.

`ParametersFineTunning.py:` WIP



**3. Pipeline folder:** Contains the core scripts of this project so we can execute trainnings, predictions and show outputs.

`DataPreprocessing.py:` WIP

`TrainModel.py:` Notebook to train a ML model used for predictions of the pollutants. We should only need to modify the widgets for normal executions. We will store (or not) a trained model into our AzureML experiments + metrics for evaluating results of the model.

`PredictData.py:` Notebook to execute predictions of the pollutants. We should only need to modify the widgets for normal executions. We will obtain a parquet and a tiff files containing the forecasted values.

`PlotMaps.py:` Notebook showing results in a Folium map. We should only need to modify the widgets for normal executions.

<br />

## Links & Resources
********

**VIF-Correlation:** https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f
