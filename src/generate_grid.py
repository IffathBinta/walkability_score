
import numpy as np
import pandas as pd
from src.plotter import plot_grid
from config.static_ws import STATIC_WEIGHTS
from config.dynamic_ws import  DYNAMIC_DATA_PATH, DYN_WEIGHTS
from config.grid import (
    CELL_M, MARGIN_M, BIN_M, 
    BUFFER_W, DECAY_L, OUT_HTML, 
    OXTS_DATA_DIR, OXTS_TS_TXT, 
    GRID_OUT_CSV,
)
from src.helpers import (
     load_oxts_latlon, latlon_to_EN, 
     segmentize_by_distance, 
     safe_build_grid, xy_to_cell,
     decay_weight,
     EN_to_latlon,
)
import src.dynamic_walkscore as dynamic_walkscore
import src.static_walkscore as static_walkscore
from config.static_ws import STATIC_DATA_PATH
import src.composite_score as composite_score
import branca.colormap as cm

def get_data():
    df_dyn = pd.read_csv(DYNAMIC_DATA_PATH)

    df_dyn.rename(columns={"frame_id": "frame"}, inplace=True)
    if {"lat","lon"}.issubset(df_dyn.columns) and df_dyn["lat"].notna().any():
        df_gps = df_dyn[["frame","lat","lon"]].dropna().drop_duplicates(subset="frame")
    else:
        df_gps = load_oxts_latlon(OXTS_DATA_DIR, OXTS_TS_TXT)

    df_en = latlon_to_EN(df_gps, lat_col="lat", lon_col="lon")
    lat0  = float(df_en["_lat0"].iloc[0])
    lon0  = float(df_en["_lon0"].iloc[0])
    cos0  = float(df_en["_coslat0"].iloc[0])

    # Merge EN coords into dynamic rows
    df_merge = df_dyn.merge(df_en[["frame","x","y"]], on="frame", how="inner")
    if df_merge.empty:
        raise ValueError("Frame mismatch between dynamic CSV and GPS.")

    df_seginfo = segmentize_by_distance(df_merge[["frame","x","y"]], frame_col="frame", bin_m=BIN_M)
    df_all = df_merge.merge(df_seginfo[["frame","s","seg_id"]], on="frame", how="left")
    df_all['walkscore_dynamic'] = df_all['ped_density_n'] + df_all['veh_density_n'] + df_all['conflict_rate_n'] + df_all['ped_exposure_n']
    return df_all, lat0, lon0, cos0


def aggregate_data(df_all):
    agg_cols = dict(
        x=("x","mean"), y=("y","mean"),
        frames=("frame","count"),

        ped_density=("ped_density","mean"),
        veh_density=("veh_density","mean"),
        ped_exposure=("ped_exposure","mean"),
        conflict_rate=("conflict_rate","mean"),
        ped_density_sum=("ped_density","sum"),
        veh_density_sum=("veh_density","sum"),
        ped_exposure_sum=("ped_exposure","sum"),
        conflict_rate_sum=("conflict_rate","sum"),
        ped_density_p90=("ped_density_n", lambda s: np.nanpercentile(s, 90)),
        veh_density_p90=("veh_density_n", lambda s: np.nanpercentile(s, 90)),
        conflict_rate_p90=("conflict_rate_n", lambda s: np.nanpercentile(s, 90)),
        ped_exposure_p90=("ped_exposure_n", lambda s: np.nanpercentile(s, 90)),
        ped_density_std=("ped_density_n","std"),
        veh_density_std=("veh_density_n","std"),
        conflict_rate_std=("conflict_rate_n","std"),
        ped_exposure_std=("ped_exposure_n","std"),
        walkscore_dynamic_frames_std=("walkscore_dynamic_frames","std"),
        frame_ids=("frame", lambda s: tuple(sorted(set(s)))),
        total_frames=("frame","count"),
    )

    seg = df_all.groupby("seg_id", dropna=True).agg(**agg_cols).reset_index()
    return seg



def build_grid(seg, lat0, lon0, cos0):
    xmin, ymin, nx, ny, cell = safe_build_grid(seg[["x","y"]], cell=CELL_M, margin=MARGIN_M)
    x_centers = xmin + (np.arange(nx) + 0.5) * cell
    y_centers = ymin + (np.arange(ny) + 0.5) * cell
        
    # features to spread
    F = [c for c in [
        'ped_density', 'veh_density',
        'ped_exposure', 'conflict_rate', 'ped_density_sum', 'veh_density_sum',
        'ped_exposure_sum', 'conflict_rate_sum', 'ped_density_p90',
        'veh_density_p90', 'conflict_rate_p90', 'ped_exposure_p90',
        'ped_density_std', 'veh_density_std', 'conflict_rate_std',
        'ped_exposure_std',  'total_frames', 'walkscore_dynamic_frames_std'
    ] if c in seg.columns]

    acc  = {f: np.zeros((ny, nx), dtype=float) for f in F}
    mass = np.zeros((ny, nx), dtype=float)

    # track contributing frame ids per cell (as sets)
    frame_sets = {}

    def add_frames(ix0, iy0, ix1, iy1, frames_tuple):
        if not frames_tuple:
            return
        for iy in range(iy0, iy1+1):
            for ix in range(ix0, ix1+1):
                key = (iy, ix)
                s = frame_sets.get(key)
                if s is None:
                    frame_sets[key] = set(frames_tuple)
                else:
                    s.update(frames_tuple)

    for _, r in seg.iterrows():
        x0, y0 = float(r["x"]), float(r["y"])
        ix0, iy0 = xy_to_cell(x0 - BUFFER_W, y0 - BUFFER_W, xmin, ymin, cell)
        ix1, iy1 = xy_to_cell(x0 + BUFFER_W, y0 + BUFFER_W, xmin, ymin, cell)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, nx-1), min(iy1, ny-1)
        if ix1 < ix0 or iy1 < iy0:
            continue

        xs = x_centers[ix0:ix1+1]; ys = y_centers[iy0:iy1+1]
        XX, YY = np.meshgrid(xs, ys)
        D = np.hypot(XX - x0, YY - y0)
        M = (D <= BUFFER_W)
        if not M.any():
            continue

        W = decay_weight(D, DECAY_L) * M

        for f in F:
            val = float(r[f]) if pd.notna(r[f]) else 0.0
            acc[f][iy0:iy1+1, ix0:ix1+1] += W * val
        mass[iy0:iy1+1, ix0:ix1+1]     += W

        # record contributing frames for these cells
        add_frames(ix0, iy0, ix1, iy1, r.get("frame_ids", ()))

    eps = 1e-9
    grid_mean = {f: acc[f] / (mass + eps) for f in F}
    conf_dyn  = np.clip(mass / (mass.max() + eps), 0.0, 1.0)

    rows = []
    for iy in range(ny):
        for ix in range(nx):
            if mass[iy, ix] <= 0:
                continue
            key = (iy, ix)
            frames_here = sorted(frame_sets.get(key, set()))
            rows.append({
                "ix": ix, "iy": iy,
                "x_center": float(x_centers[ix]),
                "y_center": float(y_centers[iy]),
                # frames that contributed (unique) and their count
                "frames_contrib": int(len(frames_here)),
                "frame_ids": ";".join(map(str, frames_here)) if frames_here else "",
                # "conf_dyn": float(conf_dyn[iy, ix]),
                **{f: float(grid_mean[f][iy, ix]) for f in F}
            })
            
    df_grid = pd.DataFrame(rows)
    df_grid, _, _ = composite_score.compute_conf_dyn(df_grid)

    rows = []
    for iy in range(ny):
        for ix in range(nx):
            if mass[iy, ix] <= 0:
                continue
            key = (iy, ix)
            frames_here = sorted(frame_sets.get(key, set()))
            rows.append({
                "ix": ix, "iy": iy,
                "x_center": float(x_centers[ix]),
                "y_center": float(y_centers[iy]),
                # frames that contributed (unique) and their count
                "frames_contrib": int(len(frames_here)),
                "frame_ids": ";".join(map(str, frames_here)) if frames_here else "",
                "conf_dyn": float(conf_dyn[iy, ix]),
                **{f: float(grid_mean[f][iy, ix]) for f in F}
            })
    df_grid = pd.DataFrame(rows)

    df_grid["lat_center"], df_grid["lon_center"] = EN_to_latlon(
        df_grid, x="x_center", y="y_center", lat0=lat0, lon0=lon0, coslat0=cos0
    )
    
    # compute dynamic walkscore
    df_grid = dynamic_walkscore.compute_walkscore_dynamic(df_grid, DYN_WEIGHTS)
    # compute static walkscore
    df_static = pd.read_csv(STATIC_DATA_PATH)
    df_grid = static_walkscore.walkscore_to_grid(df_grid, df_static)

    # compute composite score
    df_grid = composite_score.confidence_weight(df_grid)
    df_grid = composite_score.composite_score(df_grid)

    # plot grid
    plot_grid(df_grid, OUT_HTML("dynamic"),"walkscore_dyn","Dynamic Walkability")
    plot_grid(df_grid, OUT_HTML("static"),"walkscore_static","Static Walkability")
    plot_grid(df_grid, OUT_HTML("composite"),"walkscore_composite","Composite Walkability")

    # plot composite vs static
    df_grid["composite_vs_static"] = df_grid["walkscore_composite"] - df_grid["walkscore_static"]
    p5, p95 = np.nanpercentile(df_grid["composite_vs_static"], [5, 95])
    cmap = cm.linear.PiYG_09.scale(p5, p95).to_step(9)
    plot_grid(df_grid, OUT_HTML("composite_vs_static"),"composite_vs_static","Composite vs Static",cmap=cmap)
    # plot composive vs dynamic
    df_grid["composite_vs_dynamic"] = df_grid["walkscore_composite"] - df_grid["walkscore_dyn"]
    p5, p95 = np.nanpercentile(df_grid["composite_vs_dynamic"], [5, 95])
    cmap = cm.linear.PiYG_09.scale(p5, p95).to_step(9)
    plot_grid(df_grid, OUT_HTML("composite_vs_dynamic"),"composite_vs_dynamic","Composite vs Dynamic",cmap=cmap)

    df_grid.to_csv(GRID_OUT_CSV, index=False)
    
    return df_grid



def grid_pipeline():
    df_all, lat0, lon0, cos0 = get_data()
    seg = aggregate_data(df_all)
    df_grid = build_grid(seg, lat0, lon0, cos0)
    return df_grid