"""Distance-based matching for synthetic AIS vessel records.

The current demo CSV stores projected X/Y coordinates in EPSG:32618 and also
includes geographic Latitude/Longitude columns. Geographic spill coordinates
are matched with the Haversine formula; projected coordinates are matched in
metres after transforming them into the mock AIS CRS when necessary.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.warp import transform


MOCK_AIS_CRS = "EPSG:4326"
REQUIRED_ID_COLUMNS = ("VesselName", "MMSI", "VesselType")
EARTH_RADIUS_METERS = 6_371_000.0


__all__ = [
    "VesselMatcherError",
    "load_ais_data",
    "load_ais_data_full",
    "rank_vessels",
    "find_closest_vessel",
    "haversine_distance_m",
    "classify_vessel_risk",
]


class VesselMatcherError(ValueError):
    """Raised for invalid or incompatible AIS/spill coordinate data."""


def load_ais_data(csv_path, deduplicate=True):
    """Load and lightly validate the synthetic AIS CSV."""

    path = Path(csv_path)
    if not path.is_file():
        raise VesselMatcherError(f"AIS dataset not found: {path}")

    try:
        ais_df = pd.read_csv(path)
    except Exception as exc:
        raise VesselMatcherError(f"Could not read AIS dataset: {exc}") from exc

    if ais_df.empty:
        raise VesselMatcherError("AIS dataset is empty.")

    has_geographic = {"Latitude", "Longitude"}.issubset(ais_df.columns)
    has_projected = {"X", "Y"}.issubset(ais_df.columns)
    if not has_geographic and not has_projected:
        raise VesselMatcherError(
            "AIS dataset must contain Latitude/Longitude or projected X/Y columns."
        )

    for column in REQUIRED_ID_COLUMNS:
        if column not in ais_df.columns:
            ais_df[column] = "Unknown"
        ais_df[column] = ais_df[column].fillna("Unknown").astype(str)

    duplicate_count = int(ais_df["MMSI"].duplicated(keep="first").sum())
    if deduplicate and duplicate_count:
        ais_df = ais_df.drop_duplicates(subset="MMSI", keep="first").copy()
    ais_df.attrs["duplicate_count"] = duplicate_count

    if "Timestamp" in ais_df.columns:
        ais_df["Timestamp"] = pd.to_datetime(ais_df["Timestamp"], errors="coerce")

    return ais_df


def load_ais_data_full(csv_path):
    """Load the AIS CSV preserving all rows, including duplicate MMSIs.

    Unlike :func:`load_ais_data` with ``deduplicate=True``, this function does **not**
    deduplicate by MMSI. Multiple records per vessel are needed for trajectory and
    AIS-gap analysis in the suspect-scoring pipeline.
    """

    return load_ais_data(csv_path, deduplicate=False)


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Calculate geographic distance in metres between latitude/longitude points."""

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        (float(lat1), float(lon1), np.asarray(lat2, dtype=float), np.asarray(lon2, dtype=float)),
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine_a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(np.clip(haversine_a, 0, 1)))


def _parse_crs(crs_value):
    if crs_value is None or str(crs_value).strip() in {"", "None", "CRS unavailable"}:
        return None
    try:
        return CRS.from_user_input(crs_value)
    except Exception as exc:
        raise VesselMatcherError(f"Invalid spill CRS: {crs_value}") from exc


def _valid_coordinate_mask(ais_df, columns, limits):
    numeric = ais_df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1)
    for column, (lower, upper) in zip(columns, limits):
        mask &= numeric[column].between(lower, upper)
    return numeric, mask


def _add_common_result_columns(ais_df, distances, screening_radius_m):
    result = ais_df.copy()
    result["Distance_Meters"] = np.asarray(distances, dtype=float)
    result = result[np.isfinite(result["Distance_Meters"])].copy()
    result["Distance_Formatted"] = result["Distance_Meters"].map(
        lambda distance: f"{distance / 1000:.2f} km"
        if distance >= 1000
        else f"{distance:.1f} m"
    )
    result["Status"] = result["Distance_Meters"].map(classify_vessel_risk)
    result["Within_Screening_Radius"] = (
        result["Distance_Meters"] <= screening_radius_m
    )
    result = result.sort_values("Distance_Meters", ascending=True).reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    return result


def classify_vessel_risk(distance_m):
    """Return a transparent prototype status based on distance only."""

    if distance_m < 500:
        return "🔴 PRIME SUSPECT"
    if distance_m < 2_000:
        return "🟠 NEARBY"
    if distance_m <= 10_000:
        return "🟡 MONITORED"
    return "🟢 DISTANT"


def rank_vessels(ais_df, spill_x, spill_y, spill_crs, screening_radius_m=5_000):
    """Calculate numeric distances, risk labels, and ascending vessel ranks.

    Returns ``(ranked_dataframe, coordinate_method, invalid_row_count)``.
    """

    if not np.isfinite(float(spill_x)) or not np.isfinite(float(spill_y)):
        raise VesselMatcherError("Spill coordinates are invalid.")
    if screening_radius_m <= 0:
        raise VesselMatcherError("Screening radius must be positive.")

    spill_crs_obj = _parse_crs(spill_crs)
    has_geographic = {"Latitude", "Longitude"}.issubset(ais_df.columns)
    has_projected = {"X", "Y"}.issubset(ais_df.columns)

    working = ais_df.copy()
    invalid_row_count = 0

    # Prefer explicit geographic fields whenever available. This allows the
    # same synthetic CSV to work with both geographic and projected GeoTIFFs.
    if has_geographic:
        geographic, valid = _valid_coordinate_mask(
            working,
            ("Latitude", "Longitude"),
            ((-90, 90), (-180, 180)),
        )
        invalid_row_count = int((~valid).sum())
        working = working.loc[valid].copy()
        if spill_crs_obj is None:
            raise VesselMatcherError(
                "A spill CRS is required to transform the spill point before geographic matching."
            )
        if spill_crs_obj.is_geographic:
            spill_lon, spill_lat = float(spill_x), float(spill_y)
        else:
            spill_lon_list, spill_lat_list = transform(
                spill_crs_obj,
                "EPSG:4326",
                [float(spill_x)],
                [float(spill_y)],
            )
            spill_lon, spill_lat = spill_lon_list[0], spill_lat_list[0]
        distances = haversine_distance_m(
            spill_lat,
            spill_lon,
            geographic.loc[valid, "Latitude"].to_numpy(),
            geographic.loc[valid, "Longitude"].to_numpy(),
        )
        method = "Haversine distance using Latitude/Longitude"
    elif has_projected:
        projected, valid = _valid_coordinate_mask(
            working,
            ("X", "Y"),
            ((-np.inf, np.inf), (-np.inf, np.inf)),
        )
        invalid_row_count = int((~valid).sum())
        working = working.loc[valid].copy()
        projected = projected.loc[valid]
        if spill_crs_obj is None or not spill_crs_obj.is_projected:
            raise VesselMatcherError(
                "Projected AIS X/Y coordinates require a projected spill CRS."
            )
        transformed_x, transformed_y = transform(
            spill_crs_obj,
            MOCK_AIS_CRS,
            [float(spill_x)],
            [float(spill_y)],
        )
        distances = np.hypot(
            projected["X"].to_numpy() - transformed_x[0],
            projected["Y"].to_numpy() - transformed_y[0],
        )
        method = f"Euclidean distance in metres using projected AIS CRS {MOCK_AIS_CRS}"
    else:
        raise VesselMatcherError("No usable AIS coordinate columns were found.")

    if working.empty:
        raise VesselMatcherError("No valid AIS coordinates remain after validation.")

    result = _add_common_result_columns(working, distances, screening_radius_m)
    result.attrs["invalid_row_count"] = invalid_row_count
    return result, method, invalid_row_count


def find_closest_vessel(ranked_df):
    """Return the closest row, or ``None`` when no ranked rows exist."""

    if ranked_df.empty:
        return None
    return ranked_df.iloc[0]
