import pandas as pd
import numpy as np

# These coordinates are used only to design the synthetic demo scenario.
# Tab 3 still reads the actual spill origin dynamically from session state.
spill_longitude = 54.56
spill_latitude = 25.33
earth_radius_m = 6_371_000.0

# Fictional vessel specifications: name, type, target distance in metres,
# bearing in degrees, speed, and a synthetic timestamp.
vessel_specs = [
    ("Ocean Pioneer", "Tanker", 400, 15, 11.2, "2026-09-12T06:00:00Z"),
    ("Gulf Horizon", "Cargo", 1_100, 105, 14.8, "2026-09-12T06:05:00Z"),
    ("Sea Falcon", "Container", 1_700, 220, 18.1, "2026-09-12T06:10:00Z"),
    ("Arabian Trader", "Supply Vessel", 2_800, 35, 9.5, "2026-09-12T06:15:00Z"),
    ("Blue Meridian", "Bulk Carrier", 4_500, 150, 13.5, "2026-09-12T06:20:00Z"),
    ("Coastal Voyager", "Fishing", 6_800, 265, 7.4, "2026-09-12T06:25:00Z"),
    ("Eastern Star", "Passenger", 9_200, 320, 16.0, "2026-09-12T06:30:00Z"),
    ("Marine Sentinel", "Tanker", 12_500, 80, 12.7, "2026-09-12T06:35:00Z"),
    ("Pacific Trader", "Container", 18_000, 195, 18.6, "2026-09-12T06:40:00Z"),
    ("Ocean Crest", "Cargo", 22_000, 285, 13.1, "2026-09-12T06:45:00Z"),
    ("Gulf Navigator", "Tanker", 28_500, 125, 11.8, "2026-09-12T06:50:00Z"),
    ("Horizon Carrier", "Bulk Carrier", 30_000, 345, 15.4, "2026-09-12T06:55:00Z"),
]


def offset_coordinate(distance_m, bearing_degrees):
    """Create a geographic offset using a local spherical approximation."""

    bearing = np.radians(bearing_degrees)
    latitude = spill_latitude + (
        distance_m * np.cos(bearing) / earth_radius_m * 180 / np.pi
    )
    longitude = spill_longitude + (
        distance_m
        * np.sin(bearing)
        / (earth_radius_m * np.cos(np.radians(spill_latitude)))
        * 180
        / np.pi
    )
    return float(longitude), float(latitude)


vessel_data = []
for index, (name, vessel_type, distance_m, bearing, speed, timestamp) in enumerate(
    vessel_specs,
    start=1,
):
    longitude, latitude = offset_coordinate(distance_m, bearing)
    vessel_data.append(
        {
            "MMSI": f"999000{index:03d}",
            "VesselName": name,
            "VesselType": vessel_type,
            # X/Y are retained as geographic compatibility aliases.
            "X": round(longitude, 7),
            "Y": round(latitude, 7),
            "Latitude": round(latitude, 7),
            "Longitude": round(longitude, 7),
            "Timestamp": timestamp,
            "SOG": speed,
        }
    )

df = pd.DataFrame(vessel_data)
df.to_csv("marine_cadastre_ais.csv", index=False)
print("✅ Created marine_cadastre_ais.csv successfully!")
