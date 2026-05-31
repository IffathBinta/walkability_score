import numpy as np
import pandas as pd
from config.composite import CONFIDENCE_A, CONFIDENCE_K

def confidence_weight(df_grid, a=CONFIDENCE_A,  k=CONFIDENCE_K):
    """
    Confidence weight function
    """
    df_grid["alpha"] = (a*df_grid["conf_dyn"].clip(0,1) 
        + (1-a)*np.tanh(df_grid["frames_contrib"]/k)).clip(0,1)
    return df_grid

def composite_score(df_grid):
    # ---- 6) composite ----
    valid = df_grid["walkscore_static"].notna() & df_grid["walkscore_dyn"].notna()
    df_grid.loc[valid, "walkscore_composite"] = (
        (1.0 - df_grid.loc[valid,"alpha"]) * df_grid.loc[valid,"walkscore_static"] +
        df_grid.loc[valid,"alpha"]         * df_grid.loc[valid,"walkscore_dyn"]
    )
    return df_grid


def compute_conf_dyn(df_grid):
    """
    Compute confidence dynamic
    """
    frame_normalized = (
        (df_grid["total_frames"] - df_grid["total_frames"].min()) /
        (df_grid["total_frames"].max() - df_grid["total_frames"].min())
    )

    std_factor = 1 - (df_grid["walkscore_dynamic_frames_std"] /
         df_grid["walkscore_dynamic_frames_std"].max())

    df_grid["conf_dyn"] = frame_normalized * std_factor
    return df_grid, frame_normalized, std_factor