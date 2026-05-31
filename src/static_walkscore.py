import pandas as pd
import numpy as np
from src.helpers import percentile_normalize, latlon_to_EN
from scipy.spatial import cKDTree
from config.static_ws import GPS_DATA_PATH, STATIC_DATA_PATH, RADIUS, IDW_POWER, EPS
SCALE_M = 300.0 

# Load data
def static_pipeline(out: pd.DataFrame, weights: dict,) -> pd.DataFrame:
    """
    Static walkscore pipeline
    """
    td = pd.to_numeric(out["transit_distance"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_finite = td.max(skipna=True)
    fill_val = (max_finite * 1.1) if pd.notna(max_finite) else 1e6
    td = td.fillna(fill_val)
    out["transit_proximity"] = 1.0 / (1.0 + (td / SCALE_M))
    out["intersection_density_n"] = percentile_normalize(out["intersection_density"],upper_pct=95, lower_pct=5)
    out["poi_entropy_n"]          = percentile_normalize(out["poi_entropy"],upper_pct=95, lower_pct=5)
    out["essential_poi_count_n"]  = percentile_normalize(out["essential_poi_count"],upper_pct=95, lower_pct=5)
    out["transit_proximity_n"]    = percentile_normalize(out["transit_proximity"],upper_pct=95, lower_pct=5)
    out["ped_crossing_n"]         = percentile_normalize(out["ped_crossing"],upper_pct=95, lower_pct=5)
    out["walkway_n"]              = percentile_normalize(out["walkway"],upper_pct=95, lower_pct=5)
    out["sidewalk_n"]             = percentile_normalize(out["sidewalk"],upper_pct=95, lower_pct=5)
    out["drivable_area_n"]        = percentile_normalize(out["drivable_area"],upper_pct=95, lower_pct=5)
    out["drivable_inv_n"]         = percentile_normalize(out["drivable_area"],upper_pct=95, lower_pct=5,invert=True)

    w = pd.Series(weights)
    w = w / w.abs().sum()  # normalize so scale is consistent

    # Composite 0–100
    out["walkscore_static"] = (out[w.index] * w.values).sum(axis=1) * 100.0
    out["walkscore_static"] = out["walkscore_static"].clip(0, 100)

    # merge with gps data
    gps_data = pd.read_csv(GPS_DATA_PATH)
    out = out.merge(gps_data, on=["lat", "lon"], how="left")
    out.to_csv(STATIC_DATA_PATH, index=False)
    return out

def latlon_to_east_north(df_ll: pd.DataFrame, lat_col="lat", lon_col="lon", lat0=None, lon0=None):
    R = 6378137.0
    cos0 = np.cos(np.deg2rad(lat0))
    east  = (np.deg2rad(df_ll[lon_col]) - np.deg2rad(lon0)) * R * cos0
    north = (np.deg2rad(df_ll[lat_col]) - np.deg2rad(lat0)) * R
    return east, north

def walkscore_to_grid(df_grid,df_static):
    lat0 = float(df_grid["lat_center"].mean())
    lon0 = float(df_grid["lon_center"].mean())

    df_grid["x_m"], df_grid["y_m"] = latlon_to_east_north(df_grid, "lat_center", "lon_center", lat0, lon0)
    df_static["x_m"], df_static["y_m"] = latlon_to_east_north(df_static, "lat", "lon", lat0, lon0)
    tree = cKDTree(np.c_[df_static["x_m"].values, df_static["y_m"].values])

    uniq_x = np.sort(df_grid["x_m"].unique())
    cell_guess = np.median(np.diff(uniq_x)) if len(uniq_x) > 1 else 10.0
    cell = float(cell_guess)
    radius = RADIUS(cell)

    idxs_list = tree.query_ball_point(np.c_[df_grid["x_m"], df_grid["y_m"]], radius)

    static_cell = np.full(len(df_grid), np.nan, dtype=float)

    for i, idxs in enumerate(idxs_list):
        if len(idxs) == 0:
            # fallback to nearest-k=3 for sparse areas
            d, j = tree.query([df_grid["x_m"].iat[i], df_grid["y_m"].iat[i]], k=min(3, len(df_static)))
            if np.isscalar(j):
                static_cell[i] = float(df_static["walkscore_static"].iat[j])
            else:
                w = 1.0 / (np.asarray(d) + EPS)**IDW_POWER
                static_cell[i] = float(np.average(df_static["walkscore_static"].iloc[j], weights=w))
        else:
            pts = df_static.iloc[idxs]
            dx = pts["x_m"].values - df_grid["x_m"].iat[i]
            dy = pts["y_m"].values - df_grid["y_m"].iat[i]
            d = np.hypot(dx, dy)
            w = 1.0 / (d + EPS)**IDW_POWER
            static_cell[i] = float(np.average(pts["walkscore_static"].values, weights=w))

    df_grid["walkscore_static"] = static_cell
    
    return df_grid
    
    

