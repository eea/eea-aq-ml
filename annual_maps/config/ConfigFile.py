# Databricks notebook source
from dataclasses import dataclass
exec(compile(open('/dbfs/FileStore/scripts/eea/databricks/fsutils.py').read(), 'fsutils.py', 'exec'))


# COMMAND ----------

# class MLConfig:
  

# COMMAND ----------

class MLModels:
  """Class to configure our ML algorithms. 
  
  We will only need to modify the self.model_str to execute our desired model. In case we are willing to test new paramters, edit the params dict you can find at the "else" statement within "def select_params()" function.
  
  In case you are willing to add a new model, you will need to modify the self.model_str setting the desired algorithm and add a new if statement within "def select_params()" filtering by its name. Herein we will include a new if statement to set the optimized and test paremeters. Remember to import the needed library into the notebook.
  """
  
  def __init__(self, pollutant:str, type_of_params:str = 'optimized'):
    self.model_str = 'XGBRegressor()'
    
    
    self.model = eval(self.model_str)
    self.pollutant = pollutant.lower()
    self.type_of_params = type_of_params

  def select_params(self):
    
    # Filter parameters depending on the ML model we are willing to use and its type (optimized vs test)
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
    # Setting selected parameters to our model
    ml_params = self.select_params()
    model_to_train = self.model.set_params(**ml_params)
    
    return model_to_train, ml_params
  

# COMMAND ----------


# @dataclass 
# class ColsPollutants:
#   """Selected columns to include into our ML pipeline"""
  
#   no2 =  ['AreaHa',                           
#           'GridNum1km',
#           'Year',
#           'avg_eudem',                                
#           'avg_fga2015_mean_sur',                      
#           'avg_hrlgrass',
#           'avg_imp2015_mean_sur',                      
#           'avg_smallwoody_mean_sur',
#           'biogeoregions_6',
#           'cams_NO2',                                  
#           'carbonseqcapacity_mean_sur',
#           'droughtimp_mean_sur',
#           'eRep_NO2',
#           'ecoclimaregions_28',
#           'pop2018',                                 
#           'sum_clc18_111_mean_sur',                    
#           'sum_clc18_121_mean_sur',
#           'sum_clc18_122_mean_sur',
#           'sum_clc18_141_mean_sur',
#           'sum_clc18_211',
#           'sum_clc18_311_mean_sur',
#           'sum_clc18_312_mean_sur',
#           'sum_clc18_313_mean_sur',
#           'sum_elevbreak_Inlands',
#           'sum_elevbreak_Mountains',
#           'sum_envzones_ATC',
#           'sum_envzones_LUS',
#           'sum_urbdegree_11_mean_sur',
#           'sum_urbdegree_12_mean_sur',
#           'sum_urbdegree_13_mean_sur',
#           'sum_urbdegree_30',
#           'weight_tr',                                  
#           'weight_tr_mean_sur',                         
#           'weight_urb',
#           'windspeed_mean_sur']                         
  
#   pm10 = ['AreaHa',
#           'GridNum1km',
#           'Year',
#           'avg_eudem',
#           'avg_smallwoody_mean_sur',
#           'biogeoregions_4',
#           'biogeoregions_7',
#           'cams_PM10',
#           'carbonseqcapacity_mean_sur',
#           'climate_HU',
#           'climate_RR',
#           'climate_TG',
#           'climate_TX',
#           'droughtimp',
#           'eRep_PM10',
#           'ecoclimaregions_1',
#           'ecoclimaregions_7',
#           'max_fga2015',
#           'max_imp2015',
#           'pop2018',
#           'std_eudem',
#           'sum_clc18_112',
#           'sum_clc18_231_mean_sur',
#           'sum_clc18_312_mean_sur',
#           'sum_elevbreak_Inlands',
#           'sum_elevbreak_Mountains',
#           'sum_envzones_BOR',
#           'sum_envzones_PAN',
#           'sum_urbdegree_11_mean_sur',
#           'sum_urbdegree_30',
#           'weight_urb',
#           'windspeed_mean_sur']
  
#   pm25 = ['AreaHa',
#           'GridNum1km',
#           'Year',
#           'biogeoregions_4',
#           'biogeoregions_7',
#           'cams_PM25',
#           'carbonseqcapacity_mean_sur',
#           'climate_PP', 
#           'climate_RR',
#           'climate_TX', 
#           'eRep_PM25',
#           'ecoclimaregions_5',
#           'ecoclimaregions_7',
#           'max_fga2015', 
#           'max_imp2015', 
#           'max_smallwoody',
#           'pop2018',
#           'std_eudem',
#           'sum_clc18_112',
#           'sum_clc18_211_mean_sur',
#           'sum_clc18_231',
#           'sum_clc18_312_mean_sur',
#           'sum_clc18_523_mean_sur',
#           'sum_elevbreak_Inlands',
#           'sum_elevbreak_Low_coasts',
#           'sum_elevbreak_Mountains',
#           'sum_elevbreak_Uplands',
#           'sum_envzones_BOR',
#           'sum_envzones_CON',
#           'sum_urbdegree_11_mean_sur',
#           'sum_urbdegree_21_mean_sur',
#           'weight_tr_mean_var_sur',
#           'weight_urb', 
#           'windspeed_mean_sur']
  
#   o3_somo10 = ['AreaHa',
#               'GridNum1km',
#               'Year',
#               'avg_fga2015_mean_sur',
#               'avg_imp2015_mean_sur',
#               'avg_smallwoody_mean_sur',
#               'biogeoregions_1',
#               'biogeoregions_4',
#               'biogeoregions_6',
#               'biogeoregions_9',
#               'cams_O3',
#               'carbonseqcapacity_mean_sur',
#               'climate_HU',
#               'climate_TG',
#               'climate_TX',
#               'droughtimp_mean_sur',
#               'eRep_O3_SOMO10',
#               'ecoclimaregions_1',
#               'max_eudem',
#               'p50_hrlgrass',
#               'pop2018',
#               'sum_clc18_112',
#               'sum_clc18_121_mean_sur',
#               'sum_clc18_141_mean_sur',
#               'sum_clc18_243_mean_sur',
#               'sum_clc18_323',
#               'sum_elevbreak_Inlands',
#               'sum_elevbreak_Low_coasts',
#               'sum_elevbreak_Mountains',
#               'sum_envzones_ATC',
#               'sum_envzones_LUS',
#               'sum_urbdegree_11_mean_sur',
#               'sum_urbdegree_30',
#               'weight_tr_mean_sur',
#               'weight_urb',
#               'windspeed_mean_sur']

#   o3_somo35 = ['AreaHa',
#               'GridNum1km',
#               'Year',
#               'avg_fga2015_mean_sur',
#               'avg_imp2015_mean_sur',
#               'avg_smallwoody_mean_sur',
#               'biogeoregions_1',
#               'biogeoregions_4',
#               'biogeoregions_6',
#               'biogeoregions_9',
#               'cams_O3',
#               'carbonseqcapacity_mean_sur',
#               'climate_HU',
#               'climate_TG',
#               'climate_TX',
#               'droughtimp_mean_sur',
#               'eRep_O3_SOMO35',
#               'ecoclimaregions_1',
#               'max_eudem',
#               'p50_hrlgrass',
#               'pop2018',
#               'sum_clc18_112',
#               'sum_clc18_121_mean_sur',
#               'sum_clc18_141_mean_sur',
#               'sum_clc18_243_mean_sur',
#               'sum_clc18_323',
#               'sum_elevbreak_Inlands',
#               'sum_elevbreak_Low_coasts',
#               'sum_elevbreak_Mountains',
#               'sum_envzones_ATC',
#               'sum_envzones_LUS',
#               'sum_urbdegree_11_mean_sur',
#               'sum_urbdegree_30',
#               'weight_tr_mean_sur',
#               'weight_urb',
#               'windspeed_mean_sur'
#               # 'sum_envzones_MDM',
#               # 'sum_envzones_MDN',
#               # 'sum_envzones_MDS'
#               ]
                        

# COMMAND ----------


class CollectDataConfig:
  """Selected columns to include into our ML pipeline"""
  
  @staticmethod
  def select_cols(pollutant):
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
  
  @staticmethod
  def select_container():
    storage_account_name:str = 'dis2datalake'
    blob_container_name: str = 'airquality-predictions'
    sas_key: str = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
    
    return storage_account_name, blob_container_name, sas_key
  
  @staticmethod
  def select_paths():
    train_path_struct: str = '/ML_Input/data-{}_{}-{}/{}_{}/training_input_{}_{}_{}-{}.parquet'        # pollutant, predval_start_year, predval_end_year, date_of_input, version, target, pollutant, train_start_year, train_end_year
    validation_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/validation_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
    prediction_path_struct:str = '' 
    
    return train_path_struct, validation_path_struct, prediction_path_struct
                        

# COMMAND ----------

# class CollectData(CollectDataConfig):
#   """Paths storing data"""
#   def __init__(self, pollutant:str):
    
#     config = CollectDataConfig(pollutant)
#     self.storage_account_name, self.blob_container_name, self.sas_key = config.select_container()
#     self.train_path_struct, self.validation_path_struct, self.prediction_path_struct = config.select_paths()
#     self.selected_cols_pollutants = config.select_cols()
#     self.pollutant = pollutant.upper()

#   def header(self):
#     """Mounts the Azure Blob Storage Container as a File System.
#     Params
#     ------
#       :self.storage_account_name: str = Name for the storage account we are willing to connect
#       :self.blob_container_name: str = Name for the container storing the desired data
#       :self.sas_key: str = API key

#     Returns
#     -------
#       :file_system_path: str = Path to /mnt datalake 
#     """

#     file_system_path = fsutils.mount_azure_container(
#     storage_account_name = self.storage_account_name, 
#     container_name = self.blob_container_name, 
#     sas_key = self.sas_key
#     )

#     return file_system_path    
    
#   def build_path(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str): 
#     """Builds path where we are storing our datafile by following the structure determined at init
#     """
#     if train_start_year:
#       train_path:str = self.train_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, target, self.pollutant, train_start_year, train_end_year)
#       validation_path:str = self.validation_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.pollutant, predval_start_year, predval_end_year)

#       return train_path, validation_path
    
#     else:
#       prediction_path:str = self.prediction_path_struct.format()
      
#       return prediction_path, _

#   def parquet_reader(self, file_system_path:str, path_to_parket:str, features:list=['*']):
#     """Connects to the datasources and queries the desired parquet file to return a dataframe
#     Params
#     ------
#       :file_system_path: str = path to /mnt datalake
#       :path_to_parket: str = Name of the parquet file storing the desired data
#       :cols_to_select: str = Columns' name we are willing to query

#     Returns
#     -------
#       :temp_df_filtered: str = Dataframe stored in the target parquet file
#     """
    
#     temp_df = spark.read.parquet(file_system_path+path_to_parket)
#     temp_df_filtered = temp_df.select(self.selected_cols_pollutants)
    
#     return temp_df_filtered
  
#   def data_collector(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str, features:list=['*']):
#     """Pipeline to execute previous functions so we can collect desired data by calling just one function.
    
#     Returns
#     -------
#       :train_data: str = Dataframe stored in the target parquet file
#       :validation_data: str = Dataframe stored in the target parquet file
      
#       OR
#       :prediction_data: str = Dataframe stored in the target parquet file
#     """
      
#     file_system_path = self.header()
#     if train_start_year:
#       train_path, validation_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year)
      
#       train_data = self.parquet_reader(file_system_path, train_path, features)
#       validation_data = self.parquet_reader(file_system_path, validation_path, features)
      
#       return train_data, validation_data
    
#     else:
#       prediction_path, _ = self.build_path()
      
#       prediction_data = self.parquet_reader(file_system_path, prediction_path, features)
      
#       return prediction_data

  


# COMMAND ----------

# class CollectData(CollectDataConfig):
#   """Paths storing data"""
#   def __init__(self, pollutant:str):

#     self.storage_account_name:str = 'dis2datalake'
#     self.blob_container_name: str = 'airquality-predictions'
#     self.sas_key: str = 'sv=2019-12-12&ss=b&srt=co&sp=rwdlacx&se=2025-11-12T12:26:12Z&st=2020-11-12T12:26:12Z&spr=https&sig=TmnGlsXBelFacWPNZiOD2q%2BNHl7vyTl5OhKwQ6Eh1n8%3D'
    
#     self.train_path_struct: str = '/ML_Input/data-{}_{}-{}/{}_{}/training_input_{}_{}_{}-{}.parquet'        # pollutant, predval_start_year, predval_end_year, date_of_input, version, target, pollutant, train_start_year, train_end_year
#     self.validation_path_struct:str = '/ML_Input/data-{}_{}-{}/{}_{}/validation_input_{}_{}-{}.parquet'     # pollutant, predval_start_year, predval_end_year, date_of_input, version, pollutant, predval_start_year, predval_end_year
#     self.prediction_path_struct:str = '' 
    
#     self.selected_cols_pollutants = ColsPollutants()
#     self.pollutant = pollutant.upper()

#   def header(self):
#     """Mounts the Azure Blob Storage Container as a File System.
#     Params
#     ------
#       :self.storage_account_name: str = Name for the storage account we are willing to connect
#       :self.blob_container_name: str = Name for the container storing the desired data
#       :self.sas_key: str = API key

#     Returns
#     -------
#       :file_system_path: str = Path to /mnt datalake 
#     """

#     file_system_path = fsutils.mount_azure_container(
#     storage_account_name = self.storage_account_name, 
#     container_name = self.blob_container_name, 
#     sas_key = self.sas_key
#     )

#     return file_system_path    
    
#   def build_path(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str): 
#     """Builds path where we are storing our datafile by following the structure determined at init
#     """
#     if train_start_year:
#       train_path:str = self.train_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, target, self.pollutant, train_start_year, train_end_year)
#       validation_path:str = self.validation_path_struct.format(self.pollutant, predval_start_year, predval_end_year, date_of_input, version, self.pollutant, predval_start_year, predval_end_year)

#       return train_path, validation_path
    
#     else:
#       prediction_path:str = self.prediction_path_struct.format()
      
#       return prediction_path, _

#   def parquet_reader(self, file_system_path:str, path_to_parket:str, features:list=['*']):
#     """Connects to the datasources and queries the desired parquet file to return a dataframe
#     Params
#     ------
#       :file_system_path: str = path to /mnt datalake
#       :path_to_parket: str = Name of the parquet file storing the desired data
#       :cols_to_select: str = Columns' name we are willing to query

#     Returns
#     -------
#       :temp_df_filtered: str = Dataframe stored in the target parquet file
#     """
    
#     cols_to_select =  eval('self.selected_cols_pollutants.'+ str(self.pollutant).lower()) if features[0] == 'selected' else features 

#     temp_df = spark.read.parquet(file_system_path+path_to_parket)
#     temp_df_filtered = temp_df.select(cols_to_select)
    
#     return temp_df_filtered
  
#   def data_collector(self, predval_start_year:str, predval_end_year:str, date_of_input:str, version:str, target:str, train_start_year:str, train_end_year:str, features:list=['*']):
#     """Pipeline to execute previous functions so we can collect desired data by calling just one function.
    
#     Returns
#     -------
#       :train_data: str = Dataframe stored in the target parquet file
#       :validation_data: str = Dataframe stored in the target parquet file
      
#       OR
#       :prediction_data: str = Dataframe stored in the target parquet file
#     """
      
#     file_system_path = self.header()
#     if train_start_year:
#       train_path, validation_path = self.build_path(predval_start_year, predval_end_year, date_of_input, version, target, train_start_year, train_end_year)
      
#       train_data = self.parquet_reader(file_system_path, train_path, features)
#       validation_data = self.parquet_reader(file_system_path, validation_path, features)
      
#       return train_data, validation_data
    
#     else:
#       prediction_path, _ = self.build_path()
      
#       prediction_data = self.parquet_reader(file_system_path, prediction_path, features)
      
#       return prediction_data

  


# COMMAND ----------


