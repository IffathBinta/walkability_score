import os

# ========================== CONFIG ==========================
BASE_ROOT = os.path.expanduser("/home/abid/binta/KITTI_360")
SEQ       = "2013_05_28_drive_0000_sync"
DYNAMIC_DATA_PATH     = "outputs/dynamic_data.csv"   # <-- change this
DYNAMIC_GRID_PATH     = "outputs/dynamic_grid.csv"   # <-- change this
# Feature knobs
CONFLICT_D     = 3.0          # meters (center-to-center)
AREA_EQUIV_M2  = 10.0 * 10.0  # area proxy for densities (m²)
HEADING_BINS   = 12
SPEED_PCT_CAP  = 99.0         # winsorize per-frame speed lists at this percentile

KEEP_PED = {"pedestrian","person"}
KEEP_VEH = {"car","van","truck","bus","trailer","caravan","bus_rigid"}
KEEP_ALL = KEEP_PED | KEEP_VEH

VIF_FEATS_DYNAMIC = [
    "ped_density_n",
    "veh_density_n",
    "conflict_rate_n",
    "ped_exposure_n",
]
DYN_WEIGHTS = {
    "ped_density_n":         1/4,
    "veh_density_n":         1/4,
    "conflict_rate_n":       1/4,
    "ped_exposure_n":        1/4,  
}

# Sensitivity analysis knobs
SENS_N_SAMPLES = 200
SENS_SCALE     = 0.25   # ±25%
SENS_TOPK      = 50     # Top-k set stability
SENS_SEED      = 0

OUT_DIR = os.path.join("outputs")



# ----------------------------- Paths -----------------------------
DIR_BBOX_TRAIN      = os.path.join(BASE_ROOT, "data_3d_bboxes","train")
DIR_BBOX_TRAIN_FULL = os.path.join(BASE_ROOT, "data_3d_bboxes","train_full")
XML_TRAIN      = os.path.join(DIR_BBOX_TRAIN,      f"{SEQ}.xml")
XML_TRAIN_FULL = os.path.join(DIR_BBOX_TRAIN_FULL, f"{SEQ}.xml")

DIR_OXTS = os.path.join(BASE_ROOT, "data_poses_oxts_sync", SEQ, "oxts")
PATH_TS  = os.path.join(DIR_OXTS, "timestamps.txt")
DIR_DATA = os.path.join(DIR_OXTS, "data")