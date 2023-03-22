# Databricks notebook source
from dataclasses import dataclass
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
  def select_paths():
    """Paths at our container storing our desired data
    """
    train_path_struct: str = '/ML_Input/data-{}_{}-{}/{}_{}/training_input_{}_{}_{}-{}.parquet'        # pollutant, predval_start_year, predval_end_year, date_of_input, version, target, pollutant, train_start_year, train_end_year
    validation_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/validation_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
    prediction_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/prediction_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
    output_path_struct:str = '/ML_Output/{}_{}-{}_{}_maps_TEST.parquet'                                     # pollutant, predval_start_year, predval_end_year, date_of_input
    
    return train_path_struct, validation_path_struct, prediction_path_struct, output_path_struct
  
  
  @staticmethod
  def select_cols(pollutant):
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
    
    return eval(pollutant.lower())
                        

# COMMAND ----------

class MLModelsConfig:
  """Configure our ML algorithms. 
  
    - To TEST new paramters, edit the params dict you can find at the "else" statement within "def select_params()" function. If you find better results than the current ones, edit the if statement (optimized params)
    - To ADD a new model, we should only need to modify the self.model_str to execute our desired model + add a new if statement within "def select_params()" filtering by the ML name. Then, include a new if/else statement to choose between optimized and test paremeters.
  
  """
  
  def __init__(self, pollutant:str, type_of_params:str = 'optimized'):
    self.model_str = 'XGBRegressor()'
    
    
    self.model = eval(self.model_str)
    self.pollutant = pollutant.lower()
    self.type_of_params = type_of_params

    
  def select_params(self):
    """Filter parameters we are willing tu use (optimized vs test) and our desired ML model.
    """
    
    if self.model_str == 'XGBRegressor()':
      # Select our OPTIMIZED parameters for XGBoost
      if self.type_of_params == 'optimized':           # MODIFY ONLY IF YOU FOUND BETTER PARAMETERS
        params = {'no2': {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 0.3, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.7},
                  'pm10': {'learning_rate': 0.2, 'max_depth': 4, 'gamma': 5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.8},
                  'pm25':{'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.2, 'reg_lambda': 5, 'subsample': 0.7},
                  'o3_somo_10': {'learning_rate': 0.05, 'max_depth': 4, 'gamma': 3, 'reg_alpha': 0.2, 'reg_lambda': 7, 'subsample': 0.7},
                  'o3_somo_35': {'learning_rate': 0.1, 'max_depth': 5, 'gamma': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 1, 'subsample': 0.8}}
        
      # Select our TESTING parameters for XGBoost
      if self.type_of_params == 'test':                                         # MODIFY HERE TO TEST NEW PARAMETERS COMBINATIONS
        params = {'no2': {'learning_rate': 1, 'max_depth': 1, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 1},
                  'pm10': {'learning_rate': 1, 'max_depth': 1, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 1},
                  'pm25': {'learning_rate': 1, 'max_depth': 1, 'gamma': 1, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 1},
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
  

# COMMAND ----------


