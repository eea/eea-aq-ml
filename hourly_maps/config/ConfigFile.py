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
  
#   @staticmethod
#   def select_preprocess_paths():
#     """Paths at our container storing our desired data for data preprocessing
#     """
#     input_data_path_struct:str = '/Ensemble/{}/{}/{}/CAMS_{}_{}-{}-{}.tiff'                                     # pollutant, year, month, pollutant, year, month, day
#     output_data_path_struct:str = '/Ensemble/{}/{}/{}/CAMS_{}_{}-{}-{}_TEST.csv'                                     # pollutant, year, month, pollutant, year, month, day
# #     input_data_path_struct:str = '/Ensemble/{}/{}/{}/CAMS_{}_{}-{}-{}_TEST3.tiff'                                     # pollutant, year, month, pollutant, year, month, day
    
#     return input_data_path_struct, output_data_path_struct

  @staticmethod
  def select_ml_paths(path_to_return:str=None):
    """Paths at our container storing our desired data for ML purposes
    """

    train_path_struct: str = '/ML_Input/episodes/data-{}_{}_{}-all_episodes.parquet'        # pollutant, start_year, end_year
    validation_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/validation_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
    prediction_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/prediction_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
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
            'CAMS'
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
      
      
  @staticmethod
  def validate_pollutant_values(pollutant):

    max_values = {
      'no2': 1000,
      'pm10': 1200,
      'pm25': 800,
      'o3': 800,
    }

    return max_values[pollutant.lower()]
                        

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
        params = {'no2': {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 0.3, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7},
                  'pm10':  {'colsample_bytree': 0.8194213556024217, 'gamma': 11.921650137104859, 'learning_rate': 0.24859242754464722, 'max_depth': 12, 'min_child_weight': 51.55354953185203, 'n_estimators': 1600, 'random_state': 34, 'reg_alpha': 80.35878394472556, 'reg_lambda': 24.659701744476212, 'subsample': 0.78511802332347},
                  'pm25':{'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.2, 'reg_lambda': 5, 'subsample': 0.7},
                  'o3_somo10': {'learning_rate': 0.05, 'max_depth': 4, 'gamma': 3, 'reg_alpha': 0.2, 'reg_lambda': 7, 'subsample': 0.7},
                  'o3_somo35': {'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.8}}
        
      # Select our TESTING parameters for XGBoost
      if self.type_of_params == 'test':                                         # MODIFY HERE TO TEST NEW PARAMETERS COMBINATIONS
        params = {
        
        'no2': {'colsample_bytree': 0.9191129357795635, 'gamma': 169.7947857595013, 'learning_rate': 0.0194839514250874, 'max_depth': 20, 'min_child_weight': 111.21556358941127, 'n_estimators': 1200, 'reg_alpha': 1.075686299090691, 'reg_lambda': 60.061260322547376, 'subsample': 0.8994717169909148},
          # {'colsample_bytree': 0.9236337335422324, 'gamma': 86.75073133684309,'learning_rate': 0.0170875888527856,'max_depth': 20,'min_child_weight': 90.67186267878384,'n_estimators': 1200,'reg_alpha': 0.6811811821856812,'reg_lambda': 24.205850982781875,'subsample': 0.749840459197933}, #  RMSE : 11.751 corr: 0.76
            
            # {'colsample_bytree': 0.7478329665318217, 'gamma': 12.529795965543993, 'learning_rate': 0.0947727234436404, 'max_depth': 18, 'min_child_weight': 4.049627400216884, 'n_estimators': 900, 'reg_alpha': 0.7812003855219977, 'reg_lambda': 0.7935961566653076, 'subsample': 0.9820689712305682}, 
          'pm10': {'colsample_bytree': 0.640661248372002, 'gamma': 3.3225412207697786, 'learning_rate': 0.04880164831457527, 'max_depth': 18, 'min_child_weight': 44.14849644350974, 'n_estimators': 800, 'random_state': 34, 'reg_alpha': 5.772821839486555, 'reg_lambda': 14066.872321270297, 'subsample': 0.4039445671863813}, #  RMSE : 24.335 corr: 0.63,
          'pm25': {'colsample_bytree': 0.6807219214496826, 'gamma': 7.327517165412715, 'learning_rate': 0.013499690373782793, 'max_depth': 14, 'min_child_weight': 9.048684175255634, 'n_estimators': 1200, 'random_state': 34, 'reg_alpha': 0.44345862493575816, 'reg_lambda': 21.29546152013203, 'subsample': 0.651234245125773}, # RMSE: 9.49 corr:0.78
          'o3': {'colsample_bytree': 0.4164639157027386, 'gamma': 2.5744247273428362, 'learning_rate': 0.04387990470807894, 'max_depth': 16, 'min_child_weight': 0.37914611416124433, 'n_estimators': 1600, 'random_state': 34, 'reg_alpha': 2.3107143505857826, 'reg_lambda': 0.6849732843086682, 'subsample': 0.9718978418952275}, #  RMSE : 11.545 corr: 0.77
          'o3_somo_10': {'learning_rate': 1, 'max_depth': 1, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 1},
          'o3_somo_35': {'learning_rate': 1, 'max_depth': 1, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 1}}

        
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
      'no2':{'beta':2, 'urv95r': 0.24, 'rv': 200, 'alfa': 0.2, 'np': 5.2, 'nnp':5.5},
      'pm10':{'beta':2, 'urv95r': 0.28, 'rv': 50, 'alfa': 0.25, 'np': 20, 'nnp':1.5},
      'pm25':{'beta':2, 'urv95r': 0.36, 'rv': 25, 'alfa': 0.5, 'np': 20, 'nnp':1.5},
      'o3':{'beta':2, 'urv95r': 0.18, 'rv': 120, 'alfa': 0.79, 'np': 11, 'nnp':3},
    }
    
    return thresholds[self.pollutant]
  