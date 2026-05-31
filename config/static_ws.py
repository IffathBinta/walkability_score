GPS_DATA_PATH = "/home/abid/binta/KITTI_360/final_code/outputs/gps_data.csv"
STATIC_DATA_PATH     = "outputs/static_data.csv"   # <-- change this
BUFFER_M     = 300                   # analysis radius
FETCH_PAD_M  = 1200                  # padding around the convex hull of points
LANE_WIDTH_M = 7.0                   # drivable area proxy (~2 lanes)
SCALE_M = 300.0 

STATIC_WEIGHT_SCALE = 1/7
STATIC_WEIGHTS = {
    "intersection_density_n": STATIC_WEIGHT_SCALE,
    "poi_entropy_n":          STATIC_WEIGHT_SCALE,
    "essential_poi_count_n":  STATIC_WEIGHT_SCALE,
    "transit_proximity_n":    STATIC_WEIGHT_SCALE,
    "walkway_n":              STATIC_WEIGHT_SCALE,
    "sidewalk_n":             STATIC_WEIGHT_SCALE,
    "drivable_inv_n":         STATIC_WEIGHT_SCALE,
}

VIF_FEATS_STATIC = [
    "intersection_density_n",
    "poi_entropy_n",
    "essential_poi_count_n",
    "transit_proximity_n",
    "walkway_n",
    "sidewalk_n",
    "drivable_inv_n",
]

ESSENTIAL_TAGS = {
    "amenity": [
        "school","college","university",
        "hospital","clinic","doctors","pharmacy",
        "bank","library","post_office","childcare"
    ],
    "shop": ["supermarket","convenience","chemist"],
    "healthcare": ["hospital","clinic","doctor","pharmacy"]
}

# Combined tags for one-shot Overpass call
FETCH_TAGS = {
    "amenity": True,    # RADIUS = max(0.75*cell, 5.0)                 # meters
    # IDW_POWER = 2.0                                  # IDW power
    # EPS = 1e-9
    "shop": ["supermarket","convenience","chemist"],
    "healthcare": ["hospital","clinic","doctor","pharmacy"],
    "public_transport": ["stop_position","platform","stop_area","station"],
    "railway": ["station","halt","tram_stop","subway_entrance"],
    "highway": ["crossing","footway","path","pedestrian","steps","cycleway"],
    "footway": ["sidewalk"],
    "sidewalk": ["left","right","both","yes"],
}

# IDW parameters

RADIUS = lambda cell: max(0.75*cell, 5.0)                 # meters
# RADIUS = max(0.75*cell, 5.0)                 # meters
IDW_POWER = 2.0                                  # IDW power
EPS = 1e-9