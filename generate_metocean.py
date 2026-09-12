import xarray as xr
import numpy as np
import pandas as pd

# Define spatial grid around our test coordinates
x = np.linspace(200000, 250000, 10)
y = np.linspace(2700000, 2750000, 10)
time = pd.date_range("2026-09-11", periods=24, freq="h")

# Velocity vectors: u = east/west speed, v = north/south speed
u_current = np.full((len(time), len(y), len(x)), 0.45)  # 0.45 m/s East current
v_current = np.full((len(time), len(y), len(x)), 0.20)  # 0.20 m/s North current
u_wind = np.full((len(time), len(y), len(x)), 5.2)      # 5.2 m/s East wind
v_wind = np.full((len(time), len(y), len(x)), 2.1)      # 2.1 m/s North wind

# Build INCOIS/ERA5 NetCDF Data Structure
ds = xr.Dataset(
    {
        "u_current": (["time", "y", "x"], u_current),
        "v_current": (["time", "y", "x"], v_current),
        "u_wind": (["time", "y", "x"], u_wind),
        "v_wind": (["time", "y", "x"], v_wind),
    },
    coords={"x": x, "y": y, "time": time},
)

# Export directly to NetCDF format
ds.to_netcdf("ocean_currents.nc")
print("✅ Created ocean_currents.nc successfully!")