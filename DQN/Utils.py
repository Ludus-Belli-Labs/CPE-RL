import math
import random
import sys
import logging
from datetime import datetime
import os

log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Redirect all print() calls to the logger
class _PrintToLogger:
    def __init__(self, logger): self.logger = logger
    def write(self, msg):
        if msg.strip(): self.logger.info(msg.rstrip())
    def flush(self): pass

sys.stdout = _PrintToLogger(logging.getLogger())


def random_point_around(lat, lon, radius_km):
    EARTH_RADIUS = 6371
    distance = radius_km * math.sqrt(random.random())
    bearing  = random.uniform(0, 360)

    lat_rad, lon_rad, bearing_rad = map(math.radians, [lat, lon, bearing])
    ang = distance / EARTH_RADIUS

    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(ang)
        + math.cos(lat_rad) * math.sin(ang) * math.cos(bearing_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(ang) * math.cos(lat_rad),
        math.cos(ang) - math.sin(lat_rad) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def parse_start_episode(load_model: str) -> int:
    import re
    if load_model:
        match = re.search(r"ep(\d+)", os.path.basename(load_model))
        if match:
            return int(match.group(1))
    return 0