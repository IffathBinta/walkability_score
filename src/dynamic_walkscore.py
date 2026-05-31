import numpy as np
import pandas as pd
from config.dynamic_ws import  DYN_WEIGHTS
from src.helpers import percentile_normalize
from src.generate_dynamic_data import load_boxes, load_oxts, align_frames, compute_dynamic_features
from config.dynamic_ws import DYNAMIC_DATA_PATH
from src.helpers import _pick_boxes_xml

def walkscore_dynamic_math(df,weights, mask, col_name="walkscore_dyn"):
    # v1 score
    df.loc[mask, col_name] = (
        weights["ped_density_n"]   * (df.loc[mask, "ped_density_n"]) +
        weights["veh_density_n"]   * (1 - df.loc[mask, "veh_density_n"]) +
        weights["conflict_rate_n"] * (1 - df.loc[mask, "conflict_rate_n"]) +
        weights["ped_exposure_n"]  * (1 - df.loc[mask, "ped_exposure_n"])
    )
    df.loc[mask, col_name] *= 100
    return df

def compute_walkscore_dynamic(df_grid, weights=DYN_WEIGHTS) -> pd.DataFrame:

    mask_valid = df_grid["conf_dyn"] > 0

    df_grid['ped_density_n'] = percentile_normalize(df_grid['ped_density_sum'], lower_pct=5, upper_pct=95)
    df_grid['veh_density_n'] = percentile_normalize(df_grid['veh_density_sum'], lower_pct=5, upper_pct=95)
    df_grid['conflict_rate_n'] = percentile_normalize(df_grid['conflict_rate'], lower_pct=5, upper_pct=95)
    df_grid['ped_exposure_n'] = percentile_normalize(df_grid['ped_exposure'], lower_pct=5, upper_pct=95)

    df_grid = walkscore_dynamic_math(df_grid, weights, mask_valid)

    return df_grid


def compute_walkscore_dynamic_frames(df,weights=DYN_WEIGHTS):
    df['ped_density_n'] = percentile_normalize(df['ped_density'], lower_pct=5, upper_pct=95)
    df['veh_density_n'] = percentile_normalize(df['veh_density'], lower_pct=5, upper_pct=95)
    df['conflict_rate_n'] = percentile_normalize(df['conflict_rate'], lower_pct=5, upper_pct=95)
    df['ped_exposure_n'] = percentile_normalize(df['ped_exposure'], lower_pct=5, upper_pct=95)
    # mask all
    mask = df.index.notna()

    df = walkscore_dynamic_math(df, weights, mask, col_name="walkscore_dynamic_frames")
    return df

def data_generation_pipeline():
    # 1) Load inputs
    xml_path = _pick_boxes_xml()
    df_boxes = load_boxes(xml_path)
    dfgps    = load_oxts()

    assert not df_boxes.empty, "No boxes loaded; check XML path/classes."
    assert not dfgps.empty,    "No OXTS rows; check OXTS directory."

    # 2) Align frames (time-based if possible, else rank)
    df_map, dt = align_frames(df_boxes, dfgps)

    # 3) Dynamic features per frame
    df_all = compute_dynamic_features(df_boxes, df_map, dt)
    df_all = compute_walkscore_dynamic_frames(df_all)
    df_all.to_csv(DYNAMIC_DATA_PATH, index=False)
    return df_all