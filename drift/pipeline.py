"""HTTP-facing wrappers around the existing Python drift calculations.

The numerical operations follow the implementation already used by the
Streamlit Tab 2: NetCDF current/wind forcing, CRS-aware displacement, and
cumulative trajectory generation.  This module only packages those results
for the React client.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.warp import transform as transform_coordinates


EARTH_RADIUS_METERS = 6_371_000.0
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "ocean_currents.nc"
ALLOWED_MODES = {"forecast", "hindcast"}


class DriftPipelineError(ValueError):
    """Raised for invalid drift inputs or unavailable environmental data."""


def calculate_direction(east_m: float, north_m: float) -> str:
    """Return a compass direction from east/north displacement components."""

    if math.isclose(east_m, 0.0, abs_tol=1e-9) and math.isclose(
        north_m, 0.0, abs_tol=1e-9
    ):
        return "Stationary"

    angle_from_north = (math.degrees(math.atan2(east_m, north_m)) + 360) % 360
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return directions[int((angle_from_north + 22.5) // 45) % 8]


def _fill_temporal_values(values: np.ndarray, variable_name: str) -> np.ndarray:
    """Fill limited temporal gaps without allowing NaNs into the forecast."""

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise DriftPipelineError(
            f"NetCDF variable '{variable_name}' contains no valid values."
        )
    if valid.all():
        return values

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 1:
        return np.full(values.shape, values[valid_indices[0]], dtype=float)
    return np.interp(
        np.arange(len(values), dtype=float),
        valid_indices.astype(float),
        values[valid],
    )


def build_environmental_forcing(dataset: xr.Dataset, hours: int) -> dict:
    """Prepare hourly current/wind forcing using the existing Tab 2 logic."""

    required_variables = ("u_current", "v_current", "u_wind", "v_wind")
    missing_variables = [
        name for name in required_variables if name not in dataset.data_vars
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise DriftPipelineError(f"NetCDF is missing expected variable(s): {missing}.")

    time_size = int(dataset.sizes.get("time", 1))
    temporal_series = {}
    for variable_name in required_variables:
        data_array = dataset[variable_name]
        if "time" in data_array.dims:
            spatial_dimensions = [
                dimension for dimension in data_array.dims if dimension != "time"
            ]
            if spatial_dimensions:
                data_array = data_array.mean(dim=spatial_dimensions, skipna=True)
            values = np.asarray(data_array.values, dtype=float).reshape(-1)
        else:
            scalar_value = float(data_array.mean(skipna=True).values)
            values = np.full(time_size, scalar_value, dtype=float)

        if len(values) != time_size:
            values = np.full(time_size, float(np.nanmean(values)), dtype=float)
        temporal_series[variable_name] = _fill_temporal_values(
            values, variable_name
        )

    temporal_change = any(
        np.ptp(values) > 1e-9 for values in temporal_series.values()
    )
    if "time" in dataset.dims and time_size > 1 and temporal_change:
        source_hours = np.arange(time_size, dtype=float)
        target_hours = np.arange(hours, dtype=float)
        forcing = {
            variable_name: np.interp(
                target_hours,
                source_hours,
                values,
            )
            for variable_name, values in temporal_series.items()
        }
        mode = "NetCDF time-varying forcing"
        synthetic_variation = False
    else:
        initial_values = {
            variable_name: float(values[0])
            for variable_name, values in temporal_series.items()
        }
        forcing = {
            variable_name: np.full(hours, value, dtype=float)
            for variable_name, value in initial_values.items()
        }

        base_net_u = initial_values["u_current"] + 0.03 * initial_values["u_wind"]
        base_net_v = initial_values["v_current"] + 0.03 * initial_values["v_wind"]
        for hour in range(hours):
            rotation = math.radians(8.0) * math.sin(2.0 * math.pi * hour / 48.0)
            cosine = math.cos(rotation)
            sine = math.sin(rotation)
            forcing_net_u = base_net_u * cosine - base_net_v * sine
            forcing_net_v = base_net_u * sine + base_net_v * cosine
            forcing.setdefault("net_u", np.zeros(hours, dtype=float))[hour] = (
                forcing_net_u
            )
            forcing.setdefault("net_v", np.zeros(hours, dtype=float))[hour] = (
                forcing_net_v
            )
        mode = "Prototype Environmental Variation"
        synthetic_variation = True

    if not synthetic_variation:
        forcing["net_u"] = forcing["u_current"] + 0.03 * forcing["u_wind"]
        forcing["net_v"] = forcing["v_current"] + 0.03 * forcing["v_wind"]

    for component_name in ("net_u", "net_v"):
        if not np.isfinite(forcing[component_name]).all():
            raise DriftPipelineError("Environmental forcing contains invalid values.")

    return {
        "forcing": forcing,
        "mode": mode,
        "synthetic_variation": synthetic_variation,
    }


def build_drift_trajectory(
    hours: int,
    net_u: float,
    net_v: float,
    origin_x: float,
    origin_y: float,
    crs: str,
    forcing_u=None,
    forcing_v=None,
) -> pd.DataFrame:
    """Build cumulative physical displacement and display coordinates."""

    try:
        crs_obj = rasterio.crs.CRS.from_user_input(crs)
    except Exception as exc:
        raise DriftPipelineError(f"CRS could not be interpreted: {crs}") from exc

    is_geographic = crs_obj.is_geographic
    is_metric_projected = crs_obj.is_projected and getattr(
        crs_obj, "linear_units", None
    ) in ("metre", "meter", "m")
    if not is_geographic and not is_metric_projected:
        raise DriftPipelineError(
            "Drift supports geographic CRS or projected CRS with metre units."
        )

    if forcing_u is None:
        forcing_u = np.full(hours, net_u, dtype=float)
    if forcing_v is None:
        forcing_v = np.full(hours, net_v, dtype=float)
    if len(forcing_u) < hours or len(forcing_v) < hours:
        raise DriftPipelineError("Environmental forcing does not cover the duration.")

    east_displacement_m = 0.0
    north_displacement_m = 0.0
    trajectory = []
    origin_latitude_radians = math.radians(origin_y)
    for hour in range(hours + 1):
        if hour > 0:
            timestep_seconds = 3600.0
            east_displacement_m += float(forcing_u[hour - 1]) * timestep_seconds
            north_displacement_m += float(forcing_v[hour - 1]) * timestep_seconds
        total_drift_m = math.hypot(east_displacement_m, north_displacement_m)

        if is_geographic:
            latitude = origin_y + math.degrees(
                north_displacement_m / EARTH_RADIUS_METERS
            )
            longitude = origin_x + math.degrees(
                east_displacement_m
                / (EARTH_RADIUS_METERS * math.cos(origin_latitude_radians))
            )
        else:
            projected_x = origin_x + east_displacement_m
            projected_y = origin_y + north_displacement_m
            longitude_values, latitude_values = transform_coordinates(
                crs_obj,
                "EPSG:4326",
                [projected_x],
                [projected_y],
            )
            longitude = longitude_values[0]
            latitude = latitude_values[0]

        trajectory.append(
            {
                "Hour": hour,
                "East_Drift_km": east_displacement_m / 1000.0,
                "North_Drift_km": north_displacement_m / 1000.0,
                "Total_Drift_km": total_drift_m / 1000.0,
                "Latitude": latitude,
                "Longitude": longitude,
            }
        )

    return pd.DataFrame(trajectory)


def build_hindcast_trajectory(
    hours: int,
    net_u: float,
    net_v: float,
    origin_x: float,
    origin_y: float,
    crs: str,
    forcing_u=None,
    forcing_v=None,
) -> pd.DataFrame:
    """Backtrack from the detected spill by reversing the environmental vector."""

    forcing_u = -np.asarray(
        forcing_u if forcing_u is not None else np.full(hours, net_u),
        dtype=float,
    )
    forcing_v = -np.asarray(
        forcing_v if forcing_v is not None else np.full(hours, net_v),
        dtype=float,
    )
    return build_drift_trajectory(
        hours,
        -net_u,
        -net_v,
        origin_x,
        origin_y,
        crs,
        forcing_u=forcing_u,
        forcing_v=forcing_v,
    )


def _wgs84_origin(origin_x: float, origin_y: float, crs: str) -> tuple[float, float]:
    crs_obj = rasterio.crs.CRS.from_user_input(crs)
    if crs_obj.is_geographic:
        return float(origin_x), float(origin_y)
    longitude, latitude = transform_coordinates(
        crs_obj,
        "EPSG:4326",
        [origin_x],
        [origin_y],
    )
    return float(longitude[0]), float(latitude[0])


def _trajectory_payload(trajectory: pd.DataFrame, mode: str) -> list[dict]:
    payload = []
    for row in trajectory.to_dict(orient="records"):
        item = {
            "eastKm": float(row["East_Drift_km"]),
            "northKm": float(row["North_Drift_km"]),
            "totalDriftKm": float(row["Total_Drift_km"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
        }
        if mode == "forecast":
            item["hour"] = int(row["Hour"])
        else:
            item["hoursBeforeDetection"] = -int(row["Hour"])
        payload.append(item)
    return payload


def _checkpoints(trajectory: pd.DataFrame, hours: int, mode: str) -> list[dict]:
    selected_hours = [hour for hour in (0, 6, 12, 24, 36, 48) if hour <= hours]
    if hours not in selected_hours:
        selected_hours.append(hours)
    selected_hours = sorted(set(selected_hours))
    rows = trajectory[trajectory["Hour"].isin(selected_hours)]
    return [
        {
            "time": int(row["Hour"] if mode == "forecast" else -row["Hour"]),
            "eastKm": float(row["East_Drift_km"]),
            "northKm": float(row["North_Drift_km"]),
            "totalMovementKm": float(row["Total_Drift_km"]),
        }
        for row in rows.to_dict(orient="records")
    ]


def build_drift_analysis(
    *,
    origin_x: float,
    origin_y: float,
    crs: str,
    hours: int,
    mode: str = "forecast",
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict:
    """Return a typed, JSON-safe forecast or hindcast response."""

    if mode not in ALLOWED_MODES:
        raise DriftPipelineError("Mode must be either 'forecast' or 'hindcast'.")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        raise DriftPipelineError("Forecast hours must be numeric.")
    if int(hours) != hours or not 1 <= int(hours) <= 48:
        raise DriftPipelineError("Forecast hours must be an integer between 1 and 48.")
    hours = int(hours)
    try:
        origin_x = float(origin_x)
        origin_y = float(origin_y)
    except (TypeError, ValueError) as exc:
        raise DriftPipelineError("Origin coordinates must be numeric.") from exc
    if not np.isfinite([origin_x, origin_y]).all():
        raise DriftPipelineError("Origin coordinates must be finite.")

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise DriftPipelineError("Environmental dataset is unavailable.")

    try:
        crs_obj = rasterio.crs.CRS.from_user_input(crs)
        longitude, latitude = _wgs84_origin(origin_x, origin_y, crs)
    except Exception as exc:
        raise DriftPipelineError(f"CRS could not be interpreted: {crs}") from exc

    try:
        with xr.open_dataset(dataset_path) as dataset:
            environmental_result = build_environmental_forcing(dataset, hours)
    except FileNotFoundError as exc:
        raise DriftPipelineError("Environmental dataset is unavailable.") from exc
    except DriftPipelineError:
        raise
    except Exception as exc:
        raise DriftPipelineError(f"Environmental dataset could not be loaded: {exc}") from exc

    forcing = environmental_result["forcing"]
    u_current = float(forcing["u_current"][0])
    v_current = float(forcing["v_current"][0])
    u_wind = float(forcing["u_wind"][0])
    v_wind = float(forcing["v_wind"][0])
    net_u = float(forcing["net_u"][0])
    net_v = float(forcing["net_v"][0])

    if mode == "forecast":
        trajectory = build_drift_trajectory(
            hours,
            net_u,
            net_v,
            origin_x,
            origin_y,
            crs,
            forcing_u=forcing["net_u"],
            forcing_v=forcing["net_v"],
        )
    else:
        trajectory = build_hindcast_trajectory(
            hours,
            net_u,
            net_v,
            origin_x,
            origin_y,
            crs,
            forcing_u=forcing["net_u"],
            forcing_v=forcing["net_v"],
        )

    final_point = trajectory.iloc[-1]
    total_movement_km = float(final_point["Total_Drift_km"])
    direction = calculate_direction(
        float(final_point["East_Drift_km"]) * 1000.0,
        float(final_point["North_Drift_km"]) * 1000.0,
    )
    average_drift_speed = float(
        np.mean(np.hypot(forcing["net_u"], forcing["net_v"]))
    )

    environmental_forcing = [
        {
            "hour": hour,
            "currentSpeed": float(math.hypot(forcing["u_current"][hour], forcing["v_current"][hour])),
            "windSpeed": float(math.hypot(forcing["u_wind"][hour], forcing["v_wind"][hour])),
            "netSpeed": float(math.hypot(forcing["net_u"][hour], forcing["net_v"][hour])),
        }
        for hour in range(hours)
    ]

    summary = {
        "direction": direction,
        "averageDriftSpeedMps": average_drift_speed,
    }
    if mode == "forecast":
        summary["predictedMovementKm"] = total_movement_km
    else:
        summary["estimatedSourceDisplacementKm"] = total_movement_km

    return {
        "mode": mode,
        "forecastHours": hours,
        "origin": {
            "x": origin_x,
            "y": origin_y,
            "longitude": longitude,
            "latitude": latitude,
            "crs": str(crs_obj),
        },
        "environment": {
            "current": {
                "u": u_current,
                "v": v_current,
                "speed": float(math.hypot(u_current, v_current)),
            },
            "wind": {
                "u": u_wind,
                "v": v_wind,
                "speed": float(math.hypot(u_wind, v_wind)),
            },
            "combinedDrift": {
                "u": net_u,
                "v": net_v,
                "speed": float(math.hypot(net_u, net_v)),
            },
            "forcingMode": environmental_result["mode"],
            "syntheticVariation": environmental_result["synthetic_variation"],
        },
        "environmentalForcing": environmental_forcing,
        "summary": summary,
        "trajectory": _trajectory_payload(trajectory, mode),
        "checkpoints": _checkpoints(trajectory, hours, mode),
        "note": (
            "Prototype environmental drift simulation. Results are screening estimates "
            "and are not an operational hydrodynamic forecast."
            if mode == "forecast"
            else "Hindcast positions are estimated backtracked positions based on the available "
            "environmental forcing; they do not establish the spill's source."
        ),
    }
