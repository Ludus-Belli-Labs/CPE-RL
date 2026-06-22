import math
import random

# Geographic helpers

def random_point_around(lat: float, lon: float, radius_km: float) -> tuple[float, float]:
    """Return a random lat/lon within radius_km of the given point."""
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))