"""Multi-factor suspect scoring for the vessel-tracking pipeline.

This module calculates five independent evidence signals and combines them
into a single **Suspicion Score** (0–100) that ranks vessels by
investigative priority — *not* by probability of guilt.

Evidence categories and target weights::

    Distance from spill origin    40 %
    Trajectory relationship       25 %
    Vessel speed / behaviour      15 %
    AIS gap                       10 %
    Vessel type                   10 %

When a signal cannot be calculated (e.g. only one AIS record exists per
vessel, so trajectory analysis is impossible), the weight is removed and
the remaining weights are renormalized so that missing data never unfairly
penalizes a vessel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tracking.vessel_matcher import haversine_distance_m

# ---------------------------------------------------------------------------
# Default evidence weights (must sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "distance": 0.40,
    "trajectory": 0.25,
    "speed": 0.15,
    "ais_gap": 0.10,
    "vessel_type": 0.10,
}

# ---------------------------------------------------------------------------
# Vessel-type relevance lookup (case-insensitive keyword matching)
# ---------------------------------------------------------------------------
_VESSEL_TYPE_RELEVANCE: list[tuple[list[str], float]] = [
    # High relevance — vessels designed to carry oil / chemicals
    (["crude", "oil tanker", "chemical tanker"], 1.0),
    # Tanker (generic) — still high relevance
    (["tanker"], 0.9),
    # Moderate relevance — may carry fuel / cargo oil
    (["supply", "cargo", "barge", "offshore"], 0.6),
    # Lower relevance — possible but less typical
    (["fishing", "bulk carrier", "bulk"], 0.4),
    # Low relevance — generally not oil-carrying
    (["container", "passenger", "sailing", "pleasure", "tug"], 0.3),
]
_VESSEL_TYPE_DEFAULT_RELEVANCE = 0.5  # unknown / unclassified


# ===================================================================== #
#                      Individual evidence functions                     #
# ===================================================================== #

def calculate_distance_evidence(
    distance_km: float,
    screening_radius_km: float,
) -> float:
    """Smooth exponential decay of distance from the spill origin.

    Returns a value in [0, 1] where closer vessels score higher.

    Formula::

        score = exp(-distance_km / screening_radius_km)
    """
    if not math.isfinite(distance_km) or distance_km < 0:
        return 0.0
    if screening_radius_km <= 0:
        return 0.0
    return float(np.clip(math.exp(-distance_km / screening_radius_km), 0.0, 1.0))


def calculate_trajectory_evidence(
    vessel_lats: np.ndarray | list[float],
    vessel_lons: np.ndarray | list[float],
    spill_lat: float,
    spill_lon: float,
    screening_radius_km: float,
) -> dict[str, Any] | None:
    """Minimum-distance trajectory analysis for a single vessel.

    Parameters
    ----------
    vessel_lats, vessel_lons
        Chronologically ordered positions for one MMSI.
    spill_lat, spill_lon
        Spill-origin geographic coordinates (EPSG:4326).
    screening_radius_km
        Reference scale for the exponential decay.

    Returns
    -------
    dict with ``score``, ``min_distance_km``, and ``geojson`` keys,
    or ``None`` when fewer than 2 positions exist (trajectory is undefined).
    """
    lats = np.asarray(vessel_lats, dtype=float)
    lons = np.asarray(vessel_lons, dtype=float)
    valid = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[valid]
    lons = lons[valid]

    if len(lats) < 2:
        return None

    # Point-to-spill distances for every position on the track
    distances_m = haversine_distance_m(spill_lat, spill_lon, lats, lons)
    min_distance_m = float(np.nanmin(distances_m))
    min_distance_km = min_distance_m / 1_000.0

    score = float(
        np.clip(math.exp(-min_distance_km / screening_radius_km), 0.0, 1.0)
    )

    # GeoJSON LineString (longitude, latitude order per RFC 7946)
    geojson = {
        "type": "LineString",
        "coordinates": [
            [float(lon), float(lat)] for lon, lat in zip(lons, lats)
        ],
    }

    return {
        "score": score,
        "min_distance_km": round(min_distance_km, 3),
        "geojson": geojson,
    }


def calculate_speed_evidence(
    speed_knots: float,
    reference_speed: float,
) -> float | None:
    """Smooth behavioural-relevance score based on vessel speed.

    Slower speeds near the spill origin receive modestly higher evidence
    because a vessel operating/manoeuvring at low speed may be more
    relevant to a localized spill event.

    Formula::

        score = 1.0 - speed / (speed + reference_speed)

    This gives:

    * SOG ≈ 0 → score ≈ 1.0
    * SOG = reference → score = 0.5
    * SOG → ∞ → score → 0.0

    Returns ``None`` when speed is not a valid finite number.
    """
    if not math.isfinite(speed_knots) or speed_knots < 0:
        return None
    if reference_speed <= 0:
        return None
    return float(np.clip(1.0 - speed_knots / (speed_knots + reference_speed), 0.0, 1.0))


def calculate_ais_gap_evidence(
    timestamps: pd.Series | list,
    gap_threshold_minutes: float = 30.0,
) -> dict[str, Any] | None:
    """Detect meaningful temporal gaps in a vessel's AIS record history.

    Parameters
    ----------
    timestamps
        Chronologically ordered datetime values for one MMSI.
    gap_threshold_minutes
        A gap exceeding this many minutes is considered meaningful.

    Returns
    -------
    dict with ``score``, ``gap_detected`` (bool), and
    ``max_gap_minutes`` (float), or ``None`` when fewer than 2
    valid timestamps exist.
    """
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return None

    deltas = ts.diff().dropna()
    max_gap = deltas.max()
    max_gap_minutes = max_gap.total_seconds() / 60.0

    gap_detected = max_gap_minutes > gap_threshold_minutes
    score = 1.0 if gap_detected else 0.0

    return {
        "score": score,
        "gap_detected": gap_detected,
        "max_gap_minutes": round(max_gap_minutes, 1),
    }


def classify_vessel_type_relevance(vessel_type: str) -> float:
    """Return a relevance score in [0, 1] for the given vessel type.

    Uses case-insensitive substring matching against known categories.
    Unknown or empty types receive a neutral default score.
    """
    if not vessel_type or not isinstance(vessel_type, str):
        return _VESSEL_TYPE_DEFAULT_RELEVANCE

    normalized = vessel_type.strip().lower()
    if not normalized or normalized == "unknown":
        return _VESSEL_TYPE_DEFAULT_RELEVANCE

    for keywords, relevance in _VESSEL_TYPE_RELEVANCE:
        for keyword in keywords:
            if keyword in normalized:
                return relevance

    return _VESSEL_TYPE_DEFAULT_RELEVANCE


# ===================================================================== #
#                     Composite suspicion score                          #
# ===================================================================== #

def calculate_suspicion_score(
    evidence: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the weighted suspicion score with automatic renormalization.

    Parameters
    ----------
    evidence
        Mapping of evidence name → score (0–1), or ``None`` when that
        signal is unavailable.
    weights
        Per-signal weights (must align with *evidence* keys).  Defaults
        to :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    dict with keys:

    * ``suspicion_score`` – final score in [0, 100]
    * ``available_count`` – how many signals were available
    * ``total_count`` – total number of signals
    * ``available_weight`` – sum of weights for available signals
    * ``weighted_components`` – per-signal ``(score, weight, contribution)``
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    available_weight = 0.0
    weighted_sum = 0.0
    components: dict[str, dict[str, Any]] = {}
    available_count = 0
    total_count = len(evidence)

    for signal_name, score in evidence.items():
        weight = weights.get(signal_name, 0.0)
        if score is not None and math.isfinite(score):
            available_weight += weight
            contribution = score * weight
            weighted_sum += contribution
            available_count += 1
            components[signal_name] = {
                "score": round(score, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "available": True,
            }
        else:
            components[signal_name] = {
                "score": None,
                "weight": weight,
                "contribution": 0.0,
                "available": False,
            }

    if available_weight > 0:
        suspicion_score = (weighted_sum / available_weight) * 100.0
    else:
        suspicion_score = 0.0

    suspicion_score = float(np.clip(suspicion_score, 0.0, 100.0))

    return {
        "suspicion_score": round(suspicion_score, 1),
        "available_count": available_count,
        "total_count": total_count,
        "available_weight": round(available_weight, 4),
        "weighted_components": components,
    }


# ===================================================================== #
#                       Dataset introspection                            #
# ===================================================================== #

def inspect_ais_capabilities(ais_raw_df: pd.DataFrame) -> dict[str, bool]:
    """Determine which evidence signals the AIS dataset can support.

    Checks column presence, row multiplicity per MMSI, and data validity.
    """
    has_latlon = {"Latitude", "Longitude"}.issubset(ais_raw_df.columns)
    has_timestamp = "Timestamp" in ais_raw_df.columns
    has_speed = "SOG" in ais_raw_df.columns
    has_vessel_type = "VesselType" in ais_raw_df.columns

    # Multiple records per MMSI enable trajectory & gap analysis
    if "MMSI" in ais_raw_df.columns:
        records_per_vessel = ais_raw_df.groupby("MMSI").size()
        has_multi_record = bool((records_per_vessel > 1).any())
    else:
        has_multi_record = False

    return {
        "distance": has_latlon,
        "trajectory": has_latlon and has_multi_record,
        "speed": has_speed,
        "ais_gap": has_timestamp and has_multi_record,
        "vessel_type": has_vessel_type,
        "has_timestamp": has_timestamp,
        "has_course": "COG" in ais_raw_df.columns or "Heading" in ais_raw_df.columns,
        "has_multi_record": has_multi_record,
    }


# ===================================================================== #
#                         Orchestrator                                   #
# ===================================================================== #

def build_suspect_ranking(
    ranked_df: pd.DataFrame,
    ais_raw_df: pd.DataFrame,
    spill_lat: float,
    spill_lon: float,
    screening_radius_km: float,
) -> pd.DataFrame:
    """Add multi-factor suspicion scores to the distance-ranked dataframe.

    Parameters
    ----------
    ranked_df
        Output of ``rank_vessels()`` — one row per vessel, already
        containing ``Distance_Meters`` and other base columns.
    ais_raw_df
        The *full* (non-deduplicated) AIS dataframe, used for trajectory
        and AIS-gap analysis when multiple records per MMSI exist.
    spill_lat, spill_lon
        Spill-origin coordinates in EPSG:4326.
    screening_radius_km
        The user-selected screening radius (km).

    Returns
    -------
    A copy of *ranked_df* with additional evidence columns, re-sorted
    descending by ``Suspicion_Score`` and re-ranked starting from 1.
    """
    capabilities = inspect_ais_capabilities(ais_raw_df)

    # Compute reference speed from the dataset (median SOG)
    reference_speed = _compute_reference_speed(ais_raw_df)

    # Prepare per-MMSI grouped raw data for trajectory / gap analysis
    if "MMSI" in ais_raw_df.columns:
        grouped_raw = dict(list(ais_raw_df.groupby("MMSI")))
    else:
        grouped_raw = {}

    # Storage for new columns
    distance_evidence = []
    trajectory_evidence = []
    trajectory_min_dist = []
    trajectory_geojson = []
    speed_evidence = []
    speed_values = []
    ais_gap_evidence = []
    ais_gap_detected = []
    ais_gap_duration = []
    vessel_type_evidence = []
    suspicion_scores = []
    evidence_availability = []
    evidence_breakdowns = []

    for _, row in ranked_df.iterrows():
        mmsi = str(row.get("MMSI", ""))

        # --- Distance evidence ---
        dist_km = float(row["Distance_Meters"]) / 1_000.0
        dist_ev = calculate_distance_evidence(dist_km, screening_radius_km)

        # --- Trajectory evidence ---
        traj_result = None
        if capabilities["trajectory"] and mmsi in grouped_raw:
            vessel_group = grouped_raw[mmsi]
            if "Timestamp" in vessel_group.columns:
                vessel_group = vessel_group.sort_values("Timestamp")
            if len(vessel_group) >= 2:
                traj_result = calculate_trajectory_evidence(
                    vessel_group["Latitude"].values,
                    vessel_group["Longitude"].values,
                    spill_lat,
                    spill_lon,
                    screening_radius_km,
                )

        traj_score = traj_result["score"] if traj_result else None
        traj_min_d = traj_result["min_distance_km"] if traj_result else None
        traj_geo = traj_result["geojson"] if traj_result else None

        # --- Speed evidence ---
        sog = _get_speed(row, grouped_raw.get(mmsi))
        spd_ev = calculate_speed_evidence(sog, reference_speed) if sog is not None else None

        # --- AIS gap evidence ---
        gap_result = None
        if capabilities["ais_gap"] and mmsi in grouped_raw:
            vessel_group = grouped_raw[mmsi]
            if "Timestamp" in vessel_group.columns and len(vessel_group) >= 2:
                gap_result = calculate_ais_gap_evidence(vessel_group["Timestamp"])

        gap_score = gap_result["score"] if gap_result else None
        gap_flag = gap_result["gap_detected"] if gap_result else None
        gap_dur = gap_result["max_gap_minutes"] if gap_result else None

        # --- Vessel type evidence ---
        vtype_ev = classify_vessel_type_relevance(str(row.get("VesselType", "")))

        # --- Composite score ---
        evidence = {
            "distance": dist_ev,
            "trajectory": traj_score,
            "speed": spd_ev,
            "ais_gap": gap_score,
            "vessel_type": vtype_ev,
        }
        result = calculate_suspicion_score(evidence)

        # Append to column lists
        distance_evidence.append(dist_ev)
        trajectory_evidence.append(traj_score)
        trajectory_min_dist.append(traj_min_d)
        trajectory_geojson.append(traj_geo)
        speed_evidence.append(spd_ev)
        speed_values.append(sog)
        ais_gap_evidence.append(gap_score)
        ais_gap_detected.append(gap_flag)
        ais_gap_duration.append(gap_dur)
        vessel_type_evidence.append(vtype_ev)
        suspicion_scores.append(result["suspicion_score"])
        evidence_availability.append(
            f"{result['available_count']} / {result['total_count']}"
        )
        evidence_breakdowns.append(result["weighted_components"])

    # Add columns to the dataframe
    out = ranked_df.copy()
    out["Distance_Evidence"] = distance_evidence
    out["Trajectory_Evidence"] = pd.array(trajectory_evidence, dtype=pd.Float64Dtype())
    out["Min_Trajectory_Distance_km"] = pd.array(trajectory_min_dist, dtype=pd.Float64Dtype())
    out["Speed_Evidence"] = pd.array(speed_evidence, dtype=pd.Float64Dtype())
    out["Speed_Knots"] = pd.array(speed_values, dtype=pd.Float64Dtype())
    out["AIS_Gap_Evidence"] = pd.array(ais_gap_evidence, dtype=pd.Float64Dtype())
    out["AIS_Gap_Detected"] = ais_gap_detected
    out["AIS_Gap_Duration_minutes"] = pd.array(ais_gap_duration, dtype=pd.Float64Dtype())
    out["Vessel_Type_Evidence"] = vessel_type_evidence
    out["Suspicion_Score"] = suspicion_scores
    out["Evidence_Availability"] = evidence_availability
    out["_trajectory_geojson"] = trajectory_geojson
    out["_evidence_breakdown"] = evidence_breakdowns

    # Re-sort by suspicion score descending, regenerate rank
    out = out.sort_values("Suspicion_Score", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)

    # Store capabilities as dataframe attribute
    out.attrs["ais_capabilities"] = capabilities

    return out


def build_incident_result(
    suspect_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a clean incident-result dictionary from the scored dataframe.

    This structure is suitable for serialization / API responses but is
    not currently rendered as raw JSON in the Streamlit UI.
    """
    suspects = []
    for _, row in suspect_df.iterrows():
        suspect: dict[str, Any] = {
            "rank": int(row["Rank"]),
            "mmsi": str(row.get("MMSI", "")),
            "vessel_name": str(row.get("VesselName", "")),
            "vessel_type": str(row.get("VesselType", "")),
            "suspicion_score": float(row["Suspicion_Score"]),
            "distance_km": round(float(row["Distance_Meters"]) / 1_000.0, 3),
            "distance_formatted": str(row.get("Distance_Formatted", "")),
            "speed_knots": (
                float(row["Speed_Knots"])
                if pd.notna(row.get("Speed_Knots"))
                else None
            ),
            "ais_gap_detected": row.get("AIS_Gap_Detected"),
            "trajectory_geojson": row.get("_trajectory_geojson"),
            "evidence_breakdown": row.get("_evidence_breakdown"),
            "evidence_availability": str(row.get("Evidence_Availability", "")),
        }
        suspects.append(suspect)

    return {
        "incident_id": None,  # No spill-timestamp attribution currently available
        "ranked_suspects": suspects,
    }


# ===================================================================== #
#                           Helpers                                      #
# ===================================================================== #

def _compute_reference_speed(ais_raw_df: pd.DataFrame) -> float:
    """Return the median SOG from the dataset, or a sensible fallback."""
    if "SOG" not in ais_raw_df.columns:
        return 12.0  # fallback — typical merchant vessel cruising speed
    speeds = pd.to_numeric(ais_raw_df["SOG"], errors="coerce").dropna()
    speeds = speeds[speeds >= 0]
    if speeds.empty:
        return 12.0
    return float(speeds.median())


def _get_speed(
    row: pd.Series,
    vessel_group: pd.DataFrame | None,
) -> float | None:
    """Extract the best available speed value for a vessel.

    Uses the SOG from the ranked row first (single-record case), falling
    back to the most recent SOG from the full history if available.
    """
    # Try the SOG column on the ranked row
    if "SOG" in row.index:
        sog = pd.to_numeric(row["SOG"], errors="coerce")
        if pd.notna(sog) and math.isfinite(float(sog)) and float(sog) >= 0:
            return float(sog)

    # Fall back to the latest record in the full group
    if vessel_group is not None and "SOG" in vessel_group.columns:
        group = vessel_group.copy()
        if "Timestamp" in group.columns:
            group = group.sort_values("Timestamp")
        last_sog = pd.to_numeric(group["SOG"].iloc[-1], errors="coerce")
        if pd.notna(last_sog) and math.isfinite(float(last_sog)) and float(last_sog) >= 0:
            return float(last_sog)

    return None

