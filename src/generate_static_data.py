import numpy as np, osmnx as ox, geopandas as gpd
import warnings
import math
import pandas as pd
from numpy.linalg import lstsq
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import shapely
import osmnx as ox
from src.helpers import to_3857, attach_buf_ids_chunked, shannon_entropy, sum_length_km_by_buf
from config.static_ws import (
    BUFFER_M, FETCH_PAD_M, 
    LANE_WIDTH_M, ESSENTIAL_TAGS, 
    FETCH_TAGS, STATIC_DATA_PATH, GPS_DATA_PATH
    )

# --- street-count helper that works across OSMnx versions ---
try:
    # OSMnx ≥ 1.x: function lives in the stats submodule
    from osmnx.stats import count_streets_per_node
except Exception:
    # Fallback: approximate with undirected node degree
    import networkx as nx
    import pandas as pd
    def count_streets_per_node(G):
        Gu = nx.Graph(G)             # undirected, collapses parallel edges
        return pd.Series(dict(Gu.degree()))

_TRANSIT_PUBLIC_TRANSPORT = ("stop_position", "platform", "stop_area", "station")
_TRANSIT_RAILWAY = ("station", "halt", "tram_stop", "subway_entrance")


def pre_process(csv_path: str):
    # ---------- 0) Input ----------
    df_in = pd.read_csv(csv_path)
    latlon = df_in.iloc[:, -2:].copy()
    latlon.columns = ["lat","lon"]

    # quick sanity checks
    latlon["lat"] = pd.to_numeric(latlon["lat"], errors="coerce")
    latlon["lon"] = pd.to_numeric(latlon["lon"], errors="coerce")
    valid = latlon["lat"].between(-90,90) & latlon["lon"].between(-180,180)
    latlon = latlon[valid].dropna().reset_index(drop=True)

    # Build point & buffer GeoDataFrames
    g_points = gpd.GeoDataFrame(latlon, geometry=gpd.points_from_xy(latlon["lon"], latlon["lat"]), crs=4326)
    g_buffers = g_points.copy()
    g_buffers["geometry"] = g_points.to_crs(3857).buffer(BUFFER_M).to_crs(4326)
    g_buffers["buf_id"] = np.arange(len(g_buffers))

    # Fetch polygon: convex hull of points padded by FETCH_PAD_M
    hull = g_points.unary_union.convex_hull
    fetch_poly = to_3857(hull).buffer(FETCH_PAD_M)
    fetch_poly = gpd.GeoSeries([fetch_poly], crs=3857).buffer(0).to_crs(4326).iloc[0]  # clean & back to WGS84

    return g_points, g_buffers, fetch_poly


def fetch_data(fetch_poly):
    # ---------- 1) Fetch OSM once ----------
    features_gdf = ox.features_from_polygon(fetch_poly, FETCH_TAGS)
    if features_gdf is None:
        features_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)
    else:
        features_gdf = features_gdf.set_crs(4326)

    try:
        G = ox.graph_from_polygon(fetch_poly, network_type="drive", retain_all=True, simplify=True)
        nodes, edges = ox.graph_to_gdfs(G)
    except Exception:
        G = None
        nodes = gpd.GeoDataFrame(geometry=[], crs=4326)
        edges = gpd.GeoDataFrame(geometry=[], crs=4326)
    return features_gdf, nodes, edges, G


def get_intersection_density(nodes, G, g_buffers):
    # ---------- 2) Intersection density (per buffer) ----------
    if len(nodes):
        nodes_3857 = nodes.to_crs(3857)
        sc = count_streets_per_node(G)  # <- robust import / fallback
        nodes_3857["street_count"] = nodes_3857.index.map(sc).fillna(0).astype(int)
        intersections = nodes_3857.loc[nodes_3857["street_count"] >= 3, ["geometry"]].to_crs(4326)
    else:
        intersections = gpd.GeoDataFrame(geometry=[], crs=4326)

    if intersections.empty:
        inter_per_buffer = np.zeros(len(g_buffers))
    else:
        inter_join = gpd.sjoin(
            intersections,
            g_buffers[["buf_id","geometry"]],
            predicate="within",
            how="left"
        ).dropna(subset=["buf_id"])
        inter_counts = inter_join.groupby("buf_id").size()
        areas_km2 = g_buffers.to_crs(3857).area.values / 1e6
        inter_per_buffer = inter_counts.reindex(g_buffers["buf_id"]).fillna(0.0).to_numpy() / np.where(areas_km2 > 0, areas_km2, 1.0)
    return inter_per_buffer

def get_drivable_area_ratio(edges, lane_width_m, g_buffers):
    if len(edges):
        edges_3857 = edges.to_crs(3857)
        edge_bufs = edges_3857.geometry.buffer(lane_width_m/2.0)
        try:
            drive_union = edge_bufs.union_all()   # shapely 2.0+
        except AttributeError:
            drive_union = edge_bufs.unary_union   # fallback
        bufs_3857 = g_buffers.to_crs(3857)
        buf_areas_m2 = bufs_3857.area.values
        drive_area_m2 = np.array([drive_union.intersection(geom).area for geom in bufs_3857.geometry])
        drivable_ratio = np.divide(drive_area_m2, buf_areas_m2, out=np.zeros_like(drive_area_m2), where=buf_areas_m2>0)
    else:
        drivable_ratio = np.zeros(len(g_buffers))
        bufs_3857 = g_buffers.to_crs(3857)
        drivable_ratio = np.zeros(len(bufs_3857))
    return drivable_ratio, bufs_3857

def join_feature_buffers(features_gdf, g_buffers, chunk_size=10000):
    keep_cols = ["amenity","shop","healthcare","highway","public_transport","railway","footway","sidewalk","geometry"]
    features_gdf = features_gdf[[c for c in keep_cols if c in features_gdf.columns]].copy()
    features_gdf = features_gdf[features_gdf.geometry.notna() & ~features_gdf.geometry.is_empty].copy()
    try:
        features_gdf["geometry"] = shapely.make_valid(features_gdf.geometry)
    except Exception:
        features_gdf["geometry"] = features_gdf.buffer(0)
    features_gdf = features_gdf.explode(ignore_index=True)

    f_join = attach_buf_ids_chunked(
        features_gdf, 
        g_buffers[["buf_id","geometry"]],
        chunk_size=chunk_size
    )
    return f_join
    
def get_essential_count(f_join, g_buffers):
    ess_mask = (
        f_join.get("amenity", pd.Series(dtype=object)).isin(ESSENTIAL_TAGS["amenity"]).fillna(False)
        | f_join.get("shop", pd.Series(dtype=object)).isin(ESSENTIAL_TAGS["shop"]).fillna(False)
        | f_join.get("healthcare", pd.Series(dtype=object)).isin(ESSENTIAL_TAGS["healthcare"]).fillna(False)
    )

    if not f_join.empty:
        ess_counts = f_join[ess_mask].groupby("buf_id").size()
    else:
        ess_counts = pd.Series(dtype=int)

    ess_counts = (
        ess_counts
        .reindex(g_buffers["buf_id"])
        .fillna(0)
        .astype(int)
        .values
    )
    return ess_counts

def poi_entropy(f_join, g_buffers):
    if not f_join.empty and "amenity" in f_join.columns:
        ent = []
        for bid, sub in f_join.dropna(subset=["amenity"]).groupby("buf_id"):
            counts = sub["amenity"].astype(str).value_counts()
            ent.append((bid, shannon_entropy(counts)))
        ent = pd.DataFrame(ent, columns=["buf_id","entropy"]).set_index("buf_id")
        poi_entropy_vals = ent.reindex(g_buffers["buf_id"]).fillna(0.0)["entropy"].values
    else:
        poi_entropy_vals = np.zeros(len(g_buffers))
    return poi_entropy_vals




def transit_feature_mask(gdf):
    return (
        gdf.get("highway", pd.Series(dtype=object)).eq("bus_stop").fillna(False)
        | gdf.get("public_transport", pd.Series(dtype=object)).isin(_TRANSIT_PUBLIC_TRANSPORT).fillna(False)
        | gdf.get("railway", pd.Series(dtype=object)).isin(_TRANSIT_RAILWAY).fillna(False)
    )


def get_transit_distance_m(features_gdf, g_buffers):
    """Nearest transit stop/station (m) from each buffer center; uses all features in fetch area."""
    if len(features_gdf):
        transit_all = features_gdf[transit_feature_mask(features_gdf)].copy()
    else:
        transit_all = gpd.GeoDataFrame(geometry=[], crs=4326)

    centers_3857 = g_buffers.geometry.centroid.to_crs(3857)
    if transit_all.empty:
        return np.full(len(g_buffers), np.inf)

    tpts = transit_all.copy()
    tpts["geometry"] = tpts.geometry.apply(
        lambda g: g if g.geom_type == "Point" else g.representative_point()
    )
    tpts_3857 = tpts.set_crs(4326).to_crs(3857)
    tree = shapely.STRtree(tpts_3857.geometry.values)
    return np.array(
        [tree.geometries[tree.nearest(c)].distance(c) for c in centers_3857.values]
    )


def get_crossings_count(f_join, g_buffers):
    cross_mask = f_join.get("highway", pd.Series(dtype=object)).eq("crossing").fillna(False)
    cross_counts = f_join[cross_mask].groupby("buf_id").size() if not f_join.empty else pd.Series(dtype=int)
    cross_counts = cross_counts.reindex(g_buffers["buf_id"]).fillna(0).astype(int).values
    cross_per_km2 = cross_counts / (g_buffers.to_crs(3857).area.values / 1e6)
    return cross_per_km2

def get_sidewalk_per_km2(f_join, g_buffers):
    # ---------- 9) Walkway & sidewalk lengths (km per km²) ----------
    lines = f_join[f_join.geometry.type.isin(["LineString","MultiLineString"])].copy() if not f_join.empty else f_join

    # Walkways
    walk_mask = lines.get("highway", pd.Series(dtype=object)).isin(
        ["footway","path","pedestrian","steps","cycleway"]
    ).fillna(False)
    walk_lines = lines[walk_mask]
    walkway_km = sum_length_km_by_buf(walk_lines, g_buffers)

    # Sidewalks: footway=sidewalk OR any highway with sidewalk tag
    side_mask = (
        lines.get("footway", pd.Series(dtype=object)).eq("sidewalk").fillna(False) |
        lines.get("sidewalk", pd.Series(dtype=object)).isin(["left","right","both","yes"]).fillna(False)
    )
    side_lines = lines[side_mask]
    sidewalk_km = sum_length_km_by_buf(side_lines, g_buffers)

    return sidewalk_km, walkway_km


def get_sidewalk_proximity(features_gdf, g_points): # Sidewalk proximity at the frame point (meters + booleans)
    lines_all = features_gdf[
    features_gdf.geometry.type.isin(["LineString","MultiLineString"])].copy()

    # 1) Masks
    side_exp_mask = lines_all.get("footway", pd.Series(dtype=object)).eq("sidewalk")
    side_tag_mask = lines_all.get("sidewalk", pd.Series(dtype=object)).isin(["left","right","both","yes"])

    sidewalks_explicit = lines_all[side_exp_mask]
    roads_with_sidewalk = lines_all[side_tag_mask]

    # 2) Put everything in meters CRS
    frames_3857 = g_points.to_crs(3857)
    geoms_3857 = []
    if len(sidewalks_explicit):
        geoms_3857.extend(list(sidewalks_explicit.to_crs(3857).geometry.values))
    if len(roads_with_sidewalk):
        geoms_3857.extend(list(roads_with_sidewalk.to_crs(3857).geometry.values))

    # 3) Nearest distance to ANY sidewalk source
    if geoms_3857:
        tree = shapely.STRtree(geoms_3857)
        dist_to_sidewalk_m = np.array([ tree.geometries[tree.nearest(p)].distance(p)
                                        for p in frames_3857.geometry.values ], dtype=float)
    else:
        dist_to_sidewalk_m = np.full(len(frames_3857), np.inf, dtype=float)

    # 4) Flags
    HAS_5M, HAS_10M = 5.0, 10.0
    has_sidewalk_5m  = dist_to_sidewalk_m <= HAS_5M
    has_sidewalk_10m = dist_to_sidewalk_m <= HAS_10M
    return dist_to_sidewalk_m, has_sidewalk_5m, has_sidewalk_10m


def data_generation_pipeline():
    g_points, g_buffers, fetch_poly = pre_process(GPS_DATA_PATH)
    features_gdf, nodes, edges, G = fetch_data(fetch_poly)
    inter_per_buffer = get_intersection_density(nodes, G, g_buffers)
    drivable_ratio, bufs_3857 = get_drivable_area_ratio(edges, LANE_WIDTH_M, g_buffers)
    f_join = join_feature_buffers(features_gdf, g_buffers)
    ess_counts = get_essential_count(f_join, g_buffers)
    poi_entropy_vals = poi_entropy(f_join, g_buffers)
    transit_dist_m = get_transit_distance_m(features_gdf, g_buffers)
    cross_per_km2 = get_crossings_count(f_join, g_buffers)
    sidewalk_km, walkway_km = get_sidewalk_per_km2(f_join, g_buffers)
    dist_to_sidewalk_m, has_sidewalk_5m, has_sidewalk_10m = get_sidewalk_proximity(features_gdf, g_points)
    
    area_km2 = bufs_3857.area.values / 1e6
    area_km2_safe = np.where(area_km2>0, area_km2, 1.0)

    out = pd.DataFrame({
        "lat": g_points["lat"].values,
        "lon": g_points["lon"].values,
        "buffer_m": BUFFER_M,
        "intersection_density": inter_per_buffer,          # per km²
        "poi_entropy": poi_entropy_vals,
        "transit_distance": transit_dist_m,                # meters
        "essential_poi_count": ess_counts,                 # count
        "drivable_area": drivable_ratio,                   # fraction
        "ped_crossing": cross_per_km2,                     # per km²
        "walkway": walkway_km / area_km2_safe,             # km per km²
        "sidewalk": sidewalk_km / area_km2_safe,           # km per km²
        "walkway_km_raw": walkway_km,
        "sidewalk_km_raw": sidewalk_km,
        "dist_to_sidewalk_m": dist_to_sidewalk_m,
        "has_sidewalk_5m":    has_sidewalk_5m.astype(bool),
        "has_sidewalk_10m":   has_sidewalk_10m.astype(bool),
        "area_buffer_m2": bufs_3857.area.values
    })
    out.to_csv(STATIC_DATA_PATH, index=False)
    return out

