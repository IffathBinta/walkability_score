import numpy as np
from sklearn import metrics
from src.helpers import compute_vif
from config.dynamic_ws import VIF_FEATS_DYNAMIC, DYN_WEIGHTS
import pandas as pd
from eval.sensitivity_test import sensitivity_analysis


def run_dynamic_feature_eval(df):    # df is the dynamic data
    vif_static = compute_vif(df[VIF_FEATS_DYNAMIC])
    return vif_static

def qlabel(w, r, walk_thr, risk_thr):
    """Quadrant label function
        0 "A: High Walk / Low Vehicle Exposure"
        1 "B: High Walk / High Vehicle Exposure"
        2 "C: Low Walk / High Vehicle Exposure"
        3 "D: Low Walk / Low Vehicle Exposure"
    """
    if pd.isna(r): return None
    if w >= walk_thr and r < risk_thr:   return 0
    if w >= walk_thr and r >= risk_thr:  return 1
    if w <  walk_thr and r >= risk_thr:  return 2
    return 3

def metrics_series(sub):
    return metrics(sub["walkscore_redfin"], sub["walkscore_composite"])
    
def get_quad(df_merge):
    risk_series = df_merge["veh_density_p90"] if "veh_density_p90" in df_merge.columns else pd.Series(np.nan, index=df_merge.index)
    walk_thr = float(df_merge["walkscore_composite"].median(skipna=True))
    risk_thr = float(pd.to_numeric(risk_series, errors='coerce').median(skipna=True))
    df_merge["quad_class"] = [qlabel(w, r, walk_thr, risk_thr) for w, r in zip(df_merge["walkscore_composite"], risk_series)]
    by_quad = df_merge.dropna(subset=["walkscore_redfin", "walkscore_composite"]).groupby("quad_class").apply(metrics_series)
    return df_merge, by_quad