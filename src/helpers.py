import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os
import math
from datetime import datetime, timezone
from typing import Optional
from config.grid import CELL_M, MARGIN_M
from config.dynamic_ws import XML_TRAIN, XML_TRAIN_FULL


# Helper to sum line lengths per buffer (returns km per buffer)
def sum_length_km_by_buf(gdf_lines, g_buffers, buf_col="buf_id"):
    if gdf_lines.empty:
        return np.zeros(len(g_buffers))
    g = gdf_lines.to_crs(3857).copy()
    g["len_km"] = g.geometry.length / 1000.0   # compute lengths first
    summed = g.groupby(buf_col)["len_km"].sum()
    return summed.reindex(g_buffers["buf_id"]).fillna(0.0).to_numpy()


# ---------- Tiny, reusable helpers ----------
def to_3857(gdf_or_geom, crs_in=4326):
    if isinstance(gdf_or_geom, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return gdf_or_geom.to_crs(3857)
    return gpd.GeoSeries([gdf_or_geom], crs=crs_in).to_crs(3857).iloc[0]

def circle_buffer(lat, lon, radius_m):
    return (gpd.GeoSeries([Point(lon, lat)], crs=4326)
            .to_crs(3857).buffer(radius_m)
            .to_crs(4326).iloc[0])

def shannon_entropy(counts):
    total = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def compute_vif(X: pd.DataFrame):
    out = {}
    Xc = X.copy().astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    if Xc.shape[0] < 10:
        return pd.Series(dtype=float)

    # z-score standardize (avoid scale effects)
    Xc = (Xc - Xc.mean()) / Xc.std(ddof=0).replace(0, np.nan)
    Xc = Xc.dropna(axis=1, how="any")   # drop constant columns

    for col in Xc.columns:
        y = Xc[col].values
        A = np.c_[np.ones(len(Xc)), Xc.drop(columns=[col]).values]  # intercept + others
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ beta

        ssr = np.sum((yhat - y.mean())**2)
        sst = np.sum((y - y.mean())**2) + 1e-12
        R2  = float(ssr / sst)

        out[col] = 1.0 / max(1.0 - R2, 1e-6)  # VIF = 1/(1-R^2)

    return pd.Series(out).sort_values(ascending=False)

def minmax_winsor(s, lo=0.01, hi=0.99):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    a, b = s.quantile(lo), s.quantile(hi)
    s = s.clip(lower=a, upper=b)
    vmin, vmax = s.min(), s.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return pd.Series(0.0, index=s.index)
    return (s - vmin) / (vmax - vmin)

def percentile_normalize(
    x,
    invert=False,
    lower_pct=None,
    upper_pct=None,
    method="average",
):
    is_series = isinstance(x, pd.Series)
    s = x.astype(float) if is_series else pd.Series(np.asarray(x, dtype=float))

    valid = s.notna() & np.isfinite(s)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    if not valid.any():
        return out if is_series else out.to_numpy()

    if lower_pct is not None or upper_pct is not None:
        if lower_pct is None or upper_pct is None:
            raise ValueError("clip mode requires both lower_pct and upper_pct")
        lo, hi = np.percentile(s.loc[valid].values, [lower_pct, upper_pct])
        if hi > lo:
            scaled = ((s - lo) / (hi - lo)).clip(0.0, 1.0)
        else:
            scaled = pd.Series(0.5, index=s.index)
        out.loc[valid] = (1.0 - scaled.loc[valid]) if invert else scaled.loc[valid]
    else:
        ranks = s.loc[valid].rank(pct=True, method=method)
        out.loc[valid] = (1.0 - ranks) if invert else ranks

    return out if is_series else out.to_numpy()

def length_km_of_lines(gdf):
    if gdf is None or gdf.empty: return 0.0
    lines = gdf[gdf.geometry.type.isin(["LineString","MultiLineString"])]
    if lines.empty: return 0.0
    return float(lines.to_crs(3857).geometry.length.sum() / 1000.0)


def attach_buf_ids_chunked(features: gpd.GeoDataFrame,
                           buffers: gpd.GeoDataFrame,
                           chunk_size: int = 10000,
                           predicate: str = "intersects") -> gpd.GeoDataFrame:
    """
    Tag each feature with the buf_id(s) of intersecting buffers, processed in chunks.
    Works across old/new GeoPandas/rtree versions. Avoids kernel crashes from giant sjoins.
    """
    # Ensure same CRS
    if features.crs != buffers.crs:
        features = features.to_crs(buffers.crs)

    sidx = buffers.sindex
    out_chunks = []
    n = len(features)

    have_query = hasattr(sidx, "query")              # GeoPandas >= 0.12
    have_intersection = hasattr(sidx, "intersection")  # older rtree API

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        feats_chunk = features.iloc[start:end].copy()

        pairs_feat_local = []
        pairs_buf_id = []

        for i_local, g in enumerate(feats_chunk.geometry.values):
            try:
                if have_query:
                    # New API
                    buf_idx = sidx.query(g, predicate=predicate)
                elif have_intersection:
                    # Old API: candidates by bounds, then exact predicate filter
                    cand = list(sidx.intersection(g.bounds))
                    if not cand:
                        continue
                    cand_gdf = buffers.iloc[cand]
                    buf_idx = cand_gdf.index[cand_gdf.geometry.intersects(g)]
                else:
                    # Last resort: full scan (only used if neither API exists)
                    cand_gdf = buffers
                    buf_idx = cand_gdf.index[cand_gdf.geometry.intersects(g)]

                # --- normalize to list so checks are unambiguous ---
                buf_idx = list(buf_idx)
                if len(buf_idx) == 0:
                    continue

                pairs_feat_local.extend([i_local] * len(buf_idx))
                pairs_buf_id.extend(buffers.loc[buf_idx, "buf_id"].tolist())
            except Exception:
                # Skip any problematic geometry and keep going
                continue

        if pairs_feat_local:
            sub = feats_chunk.iloc[pairs_feat_local].reset_index(drop=True).copy()
            sub["buf_id"] = pairs_buf_id
            out_chunks.append(sub)

    if not out_chunks:
        return gpd.GeoDataFrame(
            columns=list(features.columns) + ["buf_id"],
            geometry="geometry",
            crs=buffers.crs,
        )

    return gpd.GeoDataFrame(pd.concat(out_chunks, ignore_index=True),
                            geometry="geometry", crs=buffers.crs)


# ----------------------------- Utils -----------------------------
def _opencv_matrix(elem):
    if elem is None: return np.zeros((0,0), float)
    rows = int(elem.findtext("rows", "0")); cols = int(elem.findtext("cols", "0"))
    vals = [float(x) for x in (elem.findtext("data","") or "").replace(",", " ").split()]
    if rows*cols != len(vals): return np.zeros((rows, cols), float)
    return np.array(vals, float).reshape(rows, cols)

def _parse_dt_str(ts: str) -> datetime:
    """Parse 'YYYY-mm-dd HH:MM:SS[.micro]' into timezone-naive datetime."""
    ts = ts.strip()
    if "." in ts:
        base, frac = ts.split(".", 1)
        micro = int(("".join(ch for ch in frac if ch.isdigit()) + "000000")[:6])
        return datetime.strptime(base, "%Y-%m-%d %H:%M:%S").replace(microsecond=micro)
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

def _try_int_timestamp_to_datetime(v: int) -> Optional[datetime]:
    """
    Try to interpret an integer as epoch-based timestamp:
      - 19 digits: ns
      - 16 digits: us
      - 13 digits: ms
      - 10 digits: s
    Returns timezone-aware UTC datetime if plausible, else None.
    """
    s = str(abs(int(v)))
    try:
        if len(s) >= 19:
            return datetime.fromtimestamp(v/1e9, tz=timezone.utc)
        if len(s) >= 16:
            return datetime.fromtimestamp(v/1e6, tz=timezone.utc)
        if len(s) >= 13:
            return datetime.fromtimestamp(v/1e3, tz=timezone.utc)
        if len(s) >= 10:
            return datetime.fromtimestamp(v, tz=timezone.utc)
    except Exception:
        return None
    return None

def _heading_entropy_discrete(angles, bins=12):
    """0..1 normalized discrete entropy from heading angles."""
    if len(angles) < 2:
        return 0.0
    edges = np.linspace(-math.pi, math.pi, bins + 1)
    counts, _ = np.histogram(angles, bins=edges)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(H / np.log(bins))  # normalized to 0..1

def _pick_boxes_xml():
    if os.path.exists(XML_TRAIN_FULL): return XML_TRAIN_FULL
    if os.path.exists(XML_TRAIN):      return XML_TRAIN
    raise FileNotFoundError(f"No boxes XML found:\n  {XML_TRAIN_FULL}\n  {XML_TRAIN}")

def _winsorize(values, pct=99.0):
    """Return a winsorized copy at given upper percentile (lower is 0)."""
    if not values:
        return values
    cap = np.percentile(values, pct)
    return [min(v, cap) for v in values]

# ===============================
# HELPERS
# ===============================
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def latlon_to_EN(df_ll: pd.DataFrame, lat_col="lat", lon_col="lon", lat0=None, lon0=None):
    """Small-angle ENU approx around mean(lat,lon)."""
    R = 6378137.0
    if lat0 is None:
        lat0 = float(df_ll[lat_col].mean())
    if lon0 is None:
        lon0 = float(df_ll[lon_col].mean())
    coslat0 = np.cos(np.deg2rad(lat0))
    east  = (np.deg2rad(df_ll[lon_col]) - np.deg2rad(lon0)) * R * coslat0
    north = (np.deg2rad(df_ll[lat_col]) - np.deg2rad(lat0)) * R
    out = df_ll.copy()
    out["x"] = east
    out["y"] = north
    out["_lat0"] = lat0
    out["_lon0"] = lon0
    out["_coslat0"] = coslat0
    return out

def EN_to_latlon(df_en: pd.DataFrame, x="x_center", y="y_center", lat0=0.0, lon0=0.0, coslat0=1.0):
    R = 6378137.0
    dlon = (df_en[x] / (R * coslat0)) * (180.0 / np.pi)
    dlat = (df_en[y] / R)            * (180.0 / np.pi)
    return (lat0 + dlat, lon0 + dlon)

def segmentize_by_distance(df_xy: pd.DataFrame, frame_col="frame", bin_m=10.0):
    D = df_xy.sort_values(frame_col).copy()
    dx = D["x"].diff().fillna(0.0)
    dy = D["y"].diff().fillna(0.0)
    D["ds"] = np.hypot(dx, dy)
    D["s"]  = D["ds"].cumsum()
    D["seg_id"] = (D["s"] // bin_m).astype(int)
    return D

def safe_build_grid(df_xy: pd.DataFrame, cell=CELL_M, margin=MARGIN_M):
    df_xy = df_xy[["x", "y"]].dropna()
    xmin, xmax = df_xy["x"].min() - margin, df_xy["x"].max() + margin
    ymin, ymax = df_xy["y"].min() - margin, df_xy["y"].max() + margin
    nx = int(np.ceil((xmax - xmin) / cell))
    ny = int(np.ceil((ymax - ymin) / cell))
    return xmin, ymin, nx, ny, cell

def xy_to_cell(x, y, xmin, ymin, cell):
    ix = int(np.floor((x - xmin) / cell))
    iy = int(np.floor((y - ymin) / cell))
    return ix, iy

def decay_weight(d, lam):
    if lam is None:
        return np.ones_like(d, dtype=float)
    return np.exp(-d / lam)

def load_oxts_latlon(data_dir, ts_file):
    if not os.path.exists(data_dir) or not os.path.exists(ts_file):
        raise FileNotFoundError("OXTS dir/timestamps not found.")
    with open(ts_file, "r") as f:
        timestamps = [line.strip() for line in f if line.strip()]
    rows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        idx = int(os.path.splitext(fname)[0])
        with open(os.path.join(data_dir, fname)) as f:
            vals = list(map(float, f.readline().split()))
            lat, lon = vals[0], vals[1]
        ts = timestamps[idx] if idx < len(timestamps) else None
        rows.append({"frame": idx, "timestamp": ts, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)

    