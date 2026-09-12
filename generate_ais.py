import pandas as pd
import numpy as np

# Coordinates from your test file: X=220650.00, Y=2719350.02
spill_x = 220650.00
spill_y = 2719350.02

# Generate 5 vessels following Marine Cadastre AIS standards
vessel_data = [
    {"MMSI": "419001234", "VesselName": "MT ARABIAN STAR", "VesselType": "Oil Tanker", "X": spill_x + 150, "Y": spill_y - 200, "SOG": 11.2},  # PRIME SUSPECT (~250m away)
    {"MMSI": "211394080", "VesselName": "INDEPENDENCE II", "VesselType": "Cargo Ship", "X": spill_x + 4500, "Y": spill_y + 3000, "SOG": 14.8}, # Cleared (~5.4km away)
    {"MMSI": "563029110", "VesselName": "SEAWAVE EXPRESS", "VesselType": "Container Ship", "X": spill_x - 12000, "Y": spill_y - 8000, "SOG": 18.1},
    {"MMSI": "311009820", "VesselName": "GLOBAL TRANSPORT", "VesselType": "Bulk Carrier", "X": spill_x + 18000, "Y": spill_y - 15000, "SOG": 13.5},
]

df = pd.DataFrame(vessel_data)
df.to_csv("marine_cadastre_ais.csv", index=False)
print("✅ Created marine_cadastre_ais.csv successfully!")