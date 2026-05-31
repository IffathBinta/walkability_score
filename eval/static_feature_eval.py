from src.helpers import compute_vif
from config.static_ws import VIF_FEATS_STATIC, STATIC_DATA_PATH, STATIC_WEIGHTS
import pandas as pd
from eval.sensitivity_test import sensitivity_analysis


def run_static_feature_eval(df):
    vif_static = compute_vif(df[VIF_FEATS_STATIC])
    return vif_static

