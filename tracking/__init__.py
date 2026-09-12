"""Vessel-matching and multi-factor suspect-scoring for the prototype dashboard."""

from .vessel_matcher import (
    VesselMatcherError,
    classify_vessel_risk,
    find_closest_vessel,
    haversine_distance_m,
    load_ais_data,
    load_ais_data_full,
    rank_vessels,
)
from .suspect_scorer import (
    DEFAULT_WEIGHTS,
    build_incident_result,
    build_suspect_ranking,
    calculate_ais_gap_evidence,
    calculate_distance_evidence,
    calculate_speed_evidence,
    calculate_suspicion_score,
    calculate_trajectory_evidence,
    classify_vessel_type_relevance,
    inspect_ais_capabilities,
)

__all__ = [
    "VesselMatcherError",
    "classify_vessel_risk",
    "find_closest_vessel",
    "haversine_distance_m",
    "load_ais_data",
    "load_ais_data_full",
    "rank_vessels",
    "DEFAULT_WEIGHTS",
    "build_incident_result",
    "build_suspect_ranking",
    "calculate_ais_gap_evidence",
    "calculate_distance_evidence",
    "calculate_speed_evidence",
    "calculate_suspicion_score",
    "calculate_trajectory_evidence",
    "classify_vessel_type_relevance",
    "inspect_ais_capabilities",
]


