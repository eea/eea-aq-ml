# Databricks notebook source
"""
================================================================================
Notebook to configure the execution of the scripts performing the training/prediction of the pollutants.
Herein we will be ablo to modify input/output paths, features (cols) we are willing to use to execute predictions and ML algorithm + its parameters.

We should only need to modify strings we can find inside classes to perform different trainings/predictions.

================================================================================

Project  : EEA Azure platform tools.
EEA Task : https://taskman.eionet.europa.eu/issues/157021
Author   : aiborra-ext@tracasa.es

================================================================================
"""

from dataclasses import dataclass
from xgboost import XGBRegressor

exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))



# COMMAND ----------

class DataHandlerConfig:
  """Configuration to collect data.
      - You should only need to modify strings within this class
      """
  
  @staticmethod
  def select_container():
    """Azure container storing our desired data
    """
    storage_account_name:str = 'dis2datalake'
    blob_container_name: str = 'airquality-predictions'
    sas_key: str = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
    
    return storage_account_name, blob_container_name, sas_key
  
  @staticmethod
  def select_preprocess_paths():
    """Paths at our container storing our desired data for data preprocessing
    """
    input_data_path_struct:str = '/Ensemble/{}/{}/{}/CAMS_{}_{}-{}-{}.tiff'                                     # pollutant, year, month, pollutant, year, month, day
    output_data_path_struct:str = '/Ensemble/{}/{}/{}/CAMS_{}_{}-{}-{}_TEST.csv'                                     # pollutant, year, month, pollutant, year, month, day
    
    return input_data_path_struct, output_data_path_struct

  @staticmethod
  def select_ml_paths(path_to_return:str=None):
    """Paths at our container storing our desired data for ML purposes
    """

    train_path_struct: str = '/ML_Input/ANNUAL_DATA/data-{}_{}-{}/{}_{}/training_input_{}_{}_{}-{}.parquet'        # pollutant, predval_start_year, predval_end_year, date_of_input, version, target, pollutant, train_start_year, train_end_year
    validation_path_struct:str = '/ML_Input/ANNUAL_DATA/data-{}_{}-{}/{}_{}/validation_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
    prediction_path_struct:str = '/ML_Input/ANNUAL_DATA/data-{}_{}-{}/{}_{}/prediction_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year

    output_parquet_path_struct:str = '/ML_Output/{}_{}-{}_{}_maps_TEST.parquet'                                     # pollutant, predval_start_year, predval_end_year, date_of_input
    raster_outputs_path_struct:str = '/ML_Output/GeoTiffs/{}_{}_{}/{}_1km_{}_{}_0_Europe_EEA_ML_XGB_TEST.tiff'  # predyear, code, agg, predyear, code, agg, ml_models_config.model_str[:2]

    if path_to_return:
      return eval(path_to_return.lower())
    
    return train_path_struct, validation_path_struct, prediction_path_struct, output_parquet_path_struct, raster_outputs_path_struct
  
  
  @staticmethod
  def select_cols(pollutant_to_return:str=None):
    """Columns we are willing to collect from parquet files
    """
    no2 =  ['AreaHa',                           
            'GridNum1km',
            'Year',
            'avg_eudem',                                 
            'avg_fga2015_mean_sur',                      
            'avg_hrlgrass',
            'avg_imp2015_mean_sur',                      
            'avg_smallwoody_mean_sur',
            'biogeoregions_6',
            'cams_NO2',                                  
            'carbonseqcapacity_mean_sur',
            'droughtimp_mean_sur',
            'eRep_NO2',
            'ecoclimaregions_28',
            'pop2018',                                   
            'sum_clc18_111_mean_sur',                    
            'sum_clc18_121_mean_sur',
            'sum_clc18_122_mean_sur',
            'sum_clc18_141_mean_sur',
            'sum_clc18_211',
            'sum_clc18_311_mean_sur',
            'sum_clc18_312_mean_sur',
            'sum_clc18_313_mean_sur',
            'sum_elevbreak_Inlands',
            'sum_elevbreak_Mountains',
            'sum_envzones_ATC',
            'sum_envzones_LUS',
            'sum_urbdegree_11_mean_sur',
            'sum_urbdegree_12_mean_sur',
            'sum_urbdegree_13_mean_sur',
            'sum_urbdegree_30',
            'weight_tr',                                  
            'weight_tr_mean_sur',                         
            'weight_urb',
            'windspeed_mean_sur']                         

    pm10 = ['AreaHa',
            'GridNum1km',
            'Year',
            'avg_eudem',
            'avg_smallwoody_mean_sur',
            'biogeoregions_4',
            'biogeoregions_7',
            'cams_PM10',
            'carbonseqcapacity_mean_sur',
            'climate_HU',
            'climate_RR',
            'climate_TG',
            'climate_TX',
            'droughtimp',
            'eRep_PM10',
            'ecoclimaregions_1',
            'ecoclimaregions_7',
            'max_fga2015',
            'max_imp2015',
            'pop2018',
            'std_eudem',
            'sum_clc18_112',
            'sum_clc18_231_mean_sur',
            'sum_clc18_312_mean_sur',
            'sum_elevbreak_Inlands',
            'sum_elevbreak_Mountains',
            'sum_envzones_BOR',
            'sum_envzones_PAN',
            'sum_urbdegree_11_mean_sur',
            'sum_urbdegree_30',
            'weight_urb',
            'windspeed_mean_sur']

    pm25 = ['AreaHa',
            'GridNum1km',
            'Year',
            'biogeoregions_4',
            'biogeoregions_7',
            'cams_PM25',
            'carbonseqcapacity_mean_sur',
            'climate_PP', 
            'climate_RR',
            'climate_TX', 
            'eRep_PM25',
            'ecoclimaregions_5',
            'ecoclimaregions_7',
            'max_fga2015', 
            'max_imp2015', 
            'max_smallwoody',
            'pop2018',
            'std_eudem',
            'sum_clc18_112',
            'sum_clc18_211_mean_sur',
            'sum_clc18_231',
            'sum_clc18_312_mean_sur',
            'sum_clc18_523_mean_sur',
            'sum_elevbreak_Inlands',
            'sum_elevbreak_Low_coasts',
            'sum_elevbreak_Mountains',
            'sum_elevbreak_Uplands',
            'sum_envzones_BOR',
            'sum_envzones_CON',
            'sum_urbdegree_11_mean_sur',
            'sum_urbdegree_21_mean_sur',
            'weight_tr_mean_var_sur',
            'weight_urb', 
            'windspeed_mean_sur']
    
    o3      = ['AreaHa',
                'GridNum1km',
                'Year',
                'avg_fga2015_mean_sur',
                'avg_imp2015_mean_sur',
                'avg_smallwoody_mean_sur',
                'biogeoregions_1',
                'biogeoregions_4',
                'biogeoregions_6',
                'biogeoregions_9',
                'cams_O3',
                'carbonseqcapacity_mean_sur',
                'climate_HU',
                'climate_TG',
                'climate_TX',
                'droughtimp_mean_sur',
                'eRep_O3',
                'ecoclimaregions_1',
                'max_eudem',
                'p50_hrlgrass',
                'pop2018',
                'sum_clc18_112',
                'sum_clc18_121_mean_sur',
                'sum_clc18_141_mean_sur',
                'sum_clc18_243_mean_sur',
                'sum_clc18_323',
                'sum_elevbreak_Inlands',
                'sum_elevbreak_Low_coasts',
                'sum_elevbreak_Mountains',
                'sum_envzones_ATC',
                'sum_envzones_LUS',
                'sum_urbdegree_11_mean_sur',
                'sum_urbdegree_30',
                'weight_tr_mean_sur',
                'weight_urb',
                'windspeed_mean_sur']

    o3_somo10 = ['AreaHa',
                'GridNum1km',
                'Year',
                'avg_fga2015_mean_sur',
                'avg_imp2015_mean_sur',
                'avg_smallwoody_mean_sur',
                'biogeoregions_1',
                'biogeoregions_4',
                'biogeoregions_6',
                'biogeoregions_9',
                'cams_O3',
                'carbonseqcapacity_mean_sur',
                'climate_HU',
                'climate_TG',
                'climate_TX',
                'droughtimp_mean_sur',
                'eRep_O3_SOMO10',
                'ecoclimaregions_1',
                'max_eudem',
                'p50_hrlgrass',
                'pop2018',
                'sum_clc18_112',
                'sum_clc18_121_mean_sur',
                'sum_clc18_141_mean_sur',
                'sum_clc18_243_mean_sur',
                'sum_clc18_323',
                'sum_elevbreak_Inlands',
                'sum_elevbreak_Low_coasts',
                'sum_elevbreak_Mountains',
                'sum_envzones_ATC',
                'sum_envzones_LUS',
                'sum_urbdegree_11_mean_sur',
                'sum_urbdegree_30',
                'weight_tr_mean_sur',
                'weight_urb',
                'windspeed_mean_sur']

    o3_somo35 = ['AreaHa',
                'GridNum1km',
                'Year',
                'avg_fga2015_mean_sur',
                'avg_imp2015_mean_sur',
                'avg_smallwoody_mean_sur',
                'biogeoregions_1',
                'biogeoregions_4',
                'biogeoregions_6',
                'biogeoregions_9',
                'cams_O3',
                'carbonseqcapacity_mean_sur',
                'climate_HU',
                'climate_TG',
                'climate_TX',
                'droughtimp_mean_sur',
                'eRep_O3_SOMO35',
                'ecoclimaregions_1',
                'max_eudem',
                'p50_hrlgrass',
                'pop2018',
                'sum_clc18_112',
                'sum_clc18_121_mean_sur',
                'sum_clc18_141_mean_sur',
                'sum_clc18_243_mean_sur',
                'sum_clc18_323',
                'sum_elevbreak_Inlands',
                'sum_elevbreak_Low_coasts',
                'sum_elevbreak_Mountains',
                'sum_envzones_ATC',
                'sum_envzones_LUS',
                'sum_urbdegree_11_mean_sur',
                'sum_urbdegree_30',
                'weight_tr_mean_sur',
                'weight_urb',
                'windspeed_mean_sur'
                # 'sum_envzones_MDM',
                # 'sum_envzones_MDN',
                # 'sum_envzones_MDS'
                ]

    if pollutant_to_return:
      return eval(pollutant_to_return.lower())
      
    return no2, pm10, pm25, o3, o3_somo10, o3_somo25
                        

# COMMAND ----------

class MLModelsConfig:
  """Configure our ML algorithms. 
  
    - To TEST new paramters, edit the params dict you can find at the "else" statement within "def select_params()" function. If you find better results than the current ones, edit the if statement (optimized params)
    - To ADD a new model, we should only need to modify the self.model_str to execute our desired model + add a new if statement within "def select_params()" filtering by the ML name. Then, include a new if/else statement to choose between optimized and test paremeters.
  
  """
  
  def __init__(self, pollutant:str, type_of_params:str):
    self.model_str = 'XGBRegressor()'
    
    
    self.model = eval(self.model_str)
    self.pollutant = pollutant.lower()
    self.type_of_params = type_of_params

    
  def select_params(self):
    """Filter parameters we are willing tu use (optimized vs test) and our desired ML model.
    """
    
    if self.model_str == 'XGBRegressor()':
      # Select our OPTIMIZED parameters for XGBoost
      if self.type_of_params == 'optimized':                                    # MODIFY ONLY IF YOU FOUND BETTER PARAMETERS
        params = {'no2': {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 0.3, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7}, #  RMSE : 5.539, corr: 0.88
                  'pm10': {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.8}, # RMSE: 4.212 corr: 0.82 ,
                  'pm25':{'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.2, 'reg_lambda': 5, 'subsample': 0.7}, #  RMSE : 2.617, corr:0.87
                  'o3': {'learning_rate': 0.05, 'max_depth': 4, 'gamma': 3, 'reg_alpha': 0.2, 'reg_lambda': 7, 'subsample': 0.7}, # RMSE: 2400.018, CORR=0.77
                  'o3_somo10': {'learning_rate': 0.05, 'max_depth': 4, 'gamma': 3, 'reg_alpha': 0.2, 'reg_lambda': 7, 'subsample': 0.7}, # RMSE: 2400.018, CORR=0.77
                  'o3_somo35': {'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.8}} #  RMSE : 1379.97, corr: 0.81
        
      # Select our TESTING parameters for XGBoost
      if self.type_of_params == 'test':                                         # MODIFY HERE TO TEST NEW PARAMETERS COMBINATIONS
        params = {
          'no2': {'colsample_bytree': 0.7478329665318217, 'gamma': 12.529795965543993, 'learning_rate': 0.0947727234436404, 'max_depth': 18, 'min_child_weight': 4.049627400216884, 'n_estimators': 900, 'reg_alpha': 0.7812003855219977, 'random_state': 34,  'reg_lambda': 0.7935961566653076, 'subsample': 0.9820689712305682}, #  RMSE : 3.46 corr:0.96 --> normal error distr aroound 0 
          'pm10':   {'colsample_bytree': 0.5333519921187898, 'gamma': 1.6671133316082125, 'learning_rate': 0.05675918594133165, 'max_depth': 13, 'min_child_weight': 1.003930564311644, 'n_estimators': 1400, 'reg_alpha': 3.1062085685015397, 'random_state': 34, 'reg_lambda': 69.3928604857206, 'subsample': 0.8887362931876186}, #  RMSE : 3.144, corr: 0.92 norm distr around 0
          'pm25': {'colsample_bytree': 0.5854042704022238, 'gamma': 0.7776736808180962, 'learning_rate': 0.13463628223320084, 'max_depth': 15, 'min_child_weight': 21.1928560965576, 'n_estimators': 1200, 'reg_alpha': 6.655947811425231, 'random_state': 34, 'reg_lambda': 69.50926470959014, 'subsample': 0.7162214831401683},  #  RMSE : 2.139 corr:0.93 -> "normal" error distr around 0 
          'o3': {'colsample_bytree': 0.4164639157027386, 'gamma': 2.5744247273428362, 'learning_rate': 0.04387990470807894, 'max_depth': 16, 'min_child_weight': 0.37914611416124433, 'n_estimators': 1600, 'reg_alpha': 2.3107143505857826, 'random_state': 34, 'reg_lambda': 0.6849732843086682, 'subsample': 0.9718978418952275}, #  RMSE : 4.787 CORR: 0.91 -> "normal" error distr around 0 
          'o3_somo10': {'colsample_bytree': 0.5818604411677151, 'gamma': 7537.732469766586, 'learning_rate': 0.01557410162248457, 'max_depth': 20, 'min_child_weight': 8.474347267574483, 'n_estimators': 1600, 'reg_alpha': 1.6333118739027552, 'random_state': 34, 'reg_lambda': 24.29553144898469, 'subsample': 0.5838718603531566}, #  RMSE : 1806.561 CORR: 0.88 -> "normal" error distr around 0 
          'o3_somo35': {'colsample_bytree': 0.9964037507890667, 'gamma': 0.42789049428167736, 'learning_rate': 0.018137673476546504, 'max_depth': 18, 'min_child_weight': 5.645269787190145, 'n_estimators': 1200, 'reg_alpha': 2.482247191238739, 'random_state': 34, 'reg_lambda': 12.336462308744192, 'subsample': 0.7020440218684973} #RMSE : 1193.196 CORR:0.86 -> "normal" error distr around 0
                }

        
    # In case we did not configure a matching ML model
    else:
      print(f'There are no defined parameters for {self.model_str}')

    return params[self.pollutant]
  
  
  def prepare_model(self):
    """Set training parameters to our chosen ML model
    """
    # Setting selected parameters to our model
    ml_params = self.select_params()
    model_to_train = self.model.set_params(**ml_params)
    
    return model_to_train, ml_params
  
  def mq_thresholds(self):
    thresholds = {
      'no2':{'urv95r': 0.24, 'rv': 200, 'alfa': 0.20, 'np': 5.20, 'nnp':5.20},
      'pm10':{'urv95r': 0.28, 'rv': 50, 'alfa': 0.13, 'np': 30, 'nnp':0.25},
      'pm25':{'urv95r': 0.36, 'rv': 25, 'alfa': 0.30, 'np': 30, 'nnp':0.25},
      'o3':{'urv95r': 0.18, 'rv': 120, 'alfa': 0.79, 'np': 11, 'nnp':3},
    }

    return thresholds[self.pollutant]
  

# COMMAND ----------


