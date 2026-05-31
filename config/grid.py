# ----------------------------
# CONFIG — edit these paths
# ----------------------------
KITTI360_ROOT = "/home/abid/binta/KITTI_360"
SEQ           = "2013_05_28_drive_0000_sync"

# Your input dynamic CSV (frame-by-frame)
OXTS_DATA_DIR = f"{KITTI360_ROOT}/data_poses_oxts_sync/{SEQ}/oxts/data"
OXTS_TS_TXT   = f"{KITTI360_ROOT}/data_poses_oxts_sync/{SEQ}/oxts/timestamps.txt"

# Outputs
GRID_OUT_CSV  = f"outputs/dynamic_spatial_grid_gps.csv"
OUT_HTML = lambda label: f"outputs/grid_map_{label}.html"

# Grid / route params
CELL_M   = 10.0     # grid cell size (meters)
MARGIN_M = 15.0     # margin around route extent (meters)
BIN_M    = 10.0     # along-route segment length (meters)
BUFFER_W = 12.0     # half-width (m) to spread segment values into nearby cells
DECAY_L  = 8.0      # distance-decay (m); None => uniform weighting inside buffer



