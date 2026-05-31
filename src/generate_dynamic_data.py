# %%
import os, math, glob, json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from config.dynamic_ws import (
    KEEP_ALL, KEEP_PED, KEEP_VEH, 
    BASE_ROOT, SEQ, DYN_WEIGHTS, 
    SENS_N_SAMPLES, SENS_SCALE, 
    SENS_TOPK, SENS_SEED, OUT_DIR,
    PATH_TS, DIR_DATA, DYNAMIC_DATA_PATH,
    CONFLICT_D, AREA_EQUIV_M2, HEADING_BINS, SPEED_PCT_CAP
)
from src.helpers import (
    _heading_entropy_discrete,
    _try_int_timestamp_to_datetime, _opencv_matrix, _parse_dt_str,
    _winsorize, _pick_boxes_xml
)

# ----------------------- Data Loaders -----------------------
def load_boxes(xml_path) -> pd.DataFrame:
    """
    Parse OpenCV XML into per-object rows.
    Tries to decode 'timestamp' as int; also stores a 'box_time' datetime if interpretable.
    """
    rows = []
    for _, node in ET.iterparse(xml_path, events=("end",)):
        if node.find("label") is None or node.find("timestamp") is None:
            continue
        label = (node.findtext("label") or "").strip().lower()
        if label not in KEEP_ALL:
            node.clear(); continue

        ts_raw = node.findtext("timestamp", "0").strip()
        try:
            frame_id_raw = int(ts_raw)
        except ValueError:
            # fallback: hash-like id if truly non-numeric (unlikely)
            frame_id_raw = int(abs(hash(ts_raw)) & 0x7FFFFFFF)

        # Best-effort parse to datetime (UTC) if ts_raw looks like epoch-based int
        box_time = _try_int_timestamp_to_datetime(frame_id_raw)

        tid = node.findtext("instanceId", "")
        tid = int(tid) if tid and tid.isdigit() else None

        V = _opencv_matrix(node.find("vertices"))
        if V.size >= 3:
            ctr = V.mean(axis=0)         # (x,y,z)
            mins, maxs = V.min(axis=0), V.max(axis=0)
            L, W, H = (maxs - mins)      # (l,w,h) extents
            x,y,z = map(float, ctr)
            l,w,h = float(L), float(W), float(H)
        else:
            x=y=z=l=w=h=0.0

        yaw = 0.0
        M = _opencv_matrix(node.find("transform"))
        if M.shape == (4,4):
            yaw = float(math.atan2(M[1,0], M[0,0]))

        rows.append(dict(frame_id=frame_id_raw, box_time=box_time,
                         cls=label, track_id=tid,
                         x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw))
        node.clear()
    df = pd.DataFrame(rows).sort_values(["frame_id","cls"]).reset_index(drop=True)
    return df

def load_oxts() -> pd.DataFrame:
    """Read OXTS timestamps + first 3 values (lat, lon, alt) from each data file."""
    if not os.path.exists(PATH_TS):  raise FileNotFoundError(f"Missing {PATH_TS}")
    if not os.path.isdir(DIR_DATA):  raise FileNotFoundError(f"Missing {DIR_DATA}")
    with open(PATH_TS) as f:
        ts_lines = [ln.strip() for ln in f if ln.strip()]
    files = sorted(glob.glob(os.path.join(DIR_DATA, "*.txt")))
    recs = []
    for i, fp in enumerate(files):
        with open(fp) as f:
            vals = f.readline().strip().split()
        lat, lon, alt = map(float, vals[:3])
        ts = ts_lines[i] if i < len(ts_lines) else ts_lines[-1]
        recs.append(dict(oxts_idx=i, timestamp=_parse_dt_str(ts), lat=lat, lon=lon, alt=alt))
    dfg = pd.DataFrame(recs)
    return dfg

# --------------------- Alignment ---------------------
def align_frames(df_boxes: pd.DataFrame, dfgps: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Prefer time-based nearest join if df_boxes.box_time is mostly available.
    Otherwise, fall back to rank-based linear mapping.
    Returns (df_map, dt_seconds).
    """
    if len(dfgps) < 2:
        raise ValueError("Not enough OXTS rows to estimate dt.")
    # dt from OXTS timeline
    ts_sec = dfgps["timestamp"].values.astype("datetime64[ns]").astype("int64") / 1e9
    dt = float(np.median(np.diff(ts_sec)))

    # Try time-based alignment
    has_time = df_boxes["box_time"].notna().mean() > 0.8  # at least 80% have time
    frames = np.array(sorted(df_boxes["frame_id"].unique()))
    df_frames = (df_boxes.groupby("frame_id", as_index=False)
                        .agg(box_time=("box_time","first")))

    if has_time:
        # Nearest merge on time
        a = df_frames.dropna(subset=["box_time"]).copy()
        a["box_time"] = pd.to_datetime(a["box_time"], utc=True)
        b = dfgps.copy()
        b["timestamp_utc"] = pd.to_datetime(b["timestamp"], utc=True)

        a = a.sort_values("box_time")
        b = b.sort_values("timestamp_utc")

        df_map = pd.merge_asof(a, b,
                               left_on="box_time",
                               right_on="timestamp_utc",
                               direction="nearest")
        df_map = df_map[["frame_id","timestamp","lat","lon","alt","oxts_idx"]]
        # Recover any frames without time via rank fallback
        missing = set(frames) - set(df_map["frame_id"].unique())
        if missing:
            df_rank, _ = align_frames_rank_only(df_boxes, dfgps)
            df_map = pd.concat([df_map, df_rank[df_rank["frame_id"].isin(missing)]],
                               ignore_index=True)
        return df_map.sort_values("frame_id").reset_index(drop=True), max(dt, 1e-6)

    # Fallback: rank-only alignment
    df_rank, dt2 = align_frames_rank_only(df_boxes, dfgps)
    return df_rank, max(dt2, 1e-6)

def align_frames_rank_only(df_boxes, dfgps) -> Tuple[pd.DataFrame, float]:
    frames = np.array(sorted(df_boxes["frame_id"].unique()))
    n_box = len(frames)
    n_gps = len(dfgps)
    if n_box == 0 or n_gps == 0:
        raise ValueError("No frames in boxes or GPS.")
    box_positions = np.linspace(0, n_gps - 1, num=n_box)
    mapped_idx = np.rint(box_positions).astype(int).clip(0, n_gps - 1)
    df_map = pd.DataFrame({"frame_id": frames, "oxts_idx": mapped_idx
                          }).merge(dfgps, on="oxts_idx", how="left")
    ts_sec = dfgps["timestamp"].values.astype("datetime64[ns]").astype("int64") / 1e9
    dt = float(np.median(np.diff(ts_sec))) if len(ts_sec) > 1 else 0.1
    return df_map, dt

# --------------------- Dynamic Features ---------------------
def compute_dynamic_features(df_boxes, df_map, dt):
    """
    Returns per-frame dynamic features joined with timestamp+lat/lon.
    - Conflicts via KD-tree (fast).
    - Heading entropy 0..1 normalized.
    - Speeds winsorized at SPEED_PCT_CAP.
    - Ped exposure graded (0..1) = min(ped_density, veh_density) rescaled to [0,1] later by min–max.
      (We keep the column name 'ped_exposure' for compatibility.)
    """
    box_by_frame = {k: v for k, v in df_boxes.groupby("frame_id")}
    frames_ordered = df_map["frame_id"].tolist()

    recs = []
    for i, fid in enumerate(frames_ordered):
        dff  = box_by_frame.get(fid, pd.DataFrame(columns=df_boxes.columns))
        peds = dff[dff.cls.isin(KEEP_PED)]
        vehs = dff[dff.cls.isin(KEEP_VEH)]
        n_p, n_v = len(peds), len(vehs)

        conf = 0
        if n_p and n_v:
            P = peds[["x","y"]].to_numpy()
            V = vehs[["x","y"]].to_numpy()
            tree = cKDTree(V)
            conf = int(tree.query_ball_point(P, r=CONFLICT_D, return_length=True).sum())

        # Speeds/Headings via previous frame same track_id
        veh_spd, ped_spd, veh_head = [], [], []
        if i > 0:
            prev_fid = frames_ordered[i-1]
            prev = box_by_frame.get(prev_fid, pd.DataFrame(columns=df_boxes.columns))
            if not prev.empty and ("track_id" in dff.columns):
                prev_map = prev.dropna(subset=["track_id"]).set_index("track_id")
                for _, r in peds.dropna(subset=["track_id"]).iterrows():
                    if r.track_id in prev_map.index:
                        pr = prev_map.loc[r.track_id]
                        dx, dy = r.x - pr.x, r.y - pr.y
                        v = math.hypot(dx, dy) / dt
                        ped_spd.append(v)
                for _, r in vehs.dropna(subset=["track_id"]).iterrows():
                    if r.track_id in prev_map.index:
                        pr = prev_map.loc[r.track_id]
                        dx, dy = r.x - pr.x, r.y - pr.y
                        v = math.hypot(dx, dy) / dt
                        veh_spd.append(v)
                        if dx*dx + dy*dy > 1e-8:
                            veh_head.append(math.atan2(dy, dx))

        # winsorize speeds (protect against occasional dt glitches or tracking jumps)
        if veh_spd:
            veh_spd = _winsorize(veh_spd, SPEED_PCT_CAP)
        if ped_spd:
            ped_spd = _winsorize(ped_spd, SPEED_PCT_CAP)

        ped_density = n_p / (AREA_EQUIV_M2 * dt)
        veh_density = n_v / (AREA_EQUIV_M2 * dt)

        # Graded co-presence intensity (still under the column name 'ped_exposure')
        ped_exposure = min(ped_density, veh_density)  # will be normalized in scoring

        rec = dict(
            frame_id=fid,
            ped_count=n_p,
            veh_count=n_v,
            ped_density=ped_density,
            veh_density=veh_density,
            conflict_count=int(conf),
            conflict_rate=float(conf) / dt,
            veh_speed_median=float(np.median(veh_spd)) if veh_spd else 0.0,
            veh_speed_std=float(np.std(veh_spd)) if len(veh_spd) > 1 else 0.0,
            ped_speed_median=float(np.median(ped_spd)) if ped_spd else 0.0,
            ped_speed_std=float(np.std(ped_spd)) if len(ped_spd) > 1 else 0.0,
            veh_heading_entropy=_heading_entropy_discrete(veh_head, bins=HEADING_BINS),
            ped_exposure=ped_exposure
        )
        recs.append(rec)

    df_feat = pd.DataFrame(recs)

    # Join timestamp + lat/lon (from alignment)
    df_all = df_feat.merge(df_map[["frame_id","timestamp","lat","lon","alt","oxts_idx"]],
                           on="frame_id", how="left")

    # Temporal diagnostics
    for col in ["ped_density", "veh_density", "conflict_rate"]:
        df_all[f"{col}_change"] = df_all[col].diff().fillna(0.0)
        df_all[f"{col}_rollstd5"] = df_all[col].rolling(5, center=True, min_periods=1).std()

    return df_all



