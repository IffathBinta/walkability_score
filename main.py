from src.generate_static_data import data_generation_pipeline as static_data_generation_pipeline
from src.dynamic_walkscore import data_generation_pipeline as dynamic_data_generation_pipeline
from config.static_ws import STATIC_DATA_PATH, STATIC_WEIGHTS
from config.dynamic_ws import DYN_WEIGHTS
import os
import pandas as pd
from src.generate_grid import grid_pipeline
from src.static_walkscore import static_pipeline
from config.dynamic_ws import DYNAMIC_DATA_PATH
from eval.static_feature_eval import run_static_feature_eval
from eval.sensitivity_test import sensitivity_analysis
from eval.dynamic_feature_eval import run_dynamic_feature_eval





if __name__ == "__main__":
    
    # create folder if not exists
    # if static_data_path does not exist, generate static data
    if not os.path.exists(STATIC_DATA_PATH):
        df_static = static_data_generation_pipeline()
        df_static = static_pipeline(df_static, STATIC_WEIGHTS)
    else:
        df_static = pd.read_csv(STATIC_DATA_PATH)
    # if dynamic_data_path does not exist, generate dynamic data
    if not os.path.exists(DYNAMIC_DATA_PATH):
        df_dyn = dynamic_data_generation_pipeline()
    else:
        df_dyn = pd.read_csv(DYNAMIC_DATA_PATH)
    df_grid = grid_pipeline()
    
    
