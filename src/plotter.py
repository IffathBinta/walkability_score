import pandas as pd
import folium
import branca.colormap as cm
import src.dynamic_walkscore as dynamic_walkscore
from config.dynamic_ws import  DYN_WEIGHTS
from config.grid import CELL_M
import numpy as np


def plot_grid(df_grid, html_path,score_col,caption=None,cmap=None):
    
    df_grid = dynamic_walkscore.compute_walkscore_dynamic(df_grid, DYN_WEIGHTS)
    cos0 = np.cos(np.deg2rad(df_grid["lat_center"].iloc[0])) 
    plot = df_grid[df_grid["conf_dyn"] > 0.2].copy()
    if score_col and plot[score_col].notna().any():
        vmin, vmax = np.nanpercentile(plot[score_col].dropna(), [5, 95])
    else:
        score_col, vmin, vmax = None, 0, 1

    m = folium.Map(
        location=[float(plot["lat_center"].mean()), float(plot["lon_center"].mean())] if not plot.empty else [0,0],
        zoom_start=16 if not plot.empty else 2, tiles="cartodbpositron"
    )

    half = CELL_M / 2.0
    R = 6378137.0
    dlat = (half / R) * (180.0 / np.pi)
    dlon = (half / (R * cos0)) * (180.0 / np.pi)

    if cmap is None:
        cmap = cm.linear.RdYlGn_09.scale(vmin, vmax).to_step(9)


    for _, r in plot.iterrows():
        lat, lon = float(r["lat_center"]), float(r["lon_center"])
        bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
        value = r[score_col] if (score_col and pd.notna(r[score_col])) else np.nan
        color_val = cmap(value) if (score_col and pd.notna(value)) else "#cccccc"
        folium.Rectangle(
            bounds=bounds, color=color_val, weight=0.5,
            fill=True, fill_opacity=0.85,
            tooltip=(f"cell=({int(r['ix'])},{int(r['iy'])}) | "
                        f"{score_col}={r.get(f'{score_col}', np.nan):.1f} | "
                        f"conf={r['conf_dyn']:.2f} | frames={int(r['frames_contrib'])}")
        ).add_to(m)

    if score_col:
        cmap.caption = caption
        m.add_child(cmap)
    m.save(html_path)