import os
import tempfile
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import xarray as xr
from rasterio.warp import transform as transform_coordinates

from detection.classifier import (
    ClassifierLoadError,
    EfficientNetOilClassifier,
    runtime_status,
)
import importlib
import tracking.vessel_matcher
import tracking.suspect_scorer

# Ensure latest module attributes if Streamlit process was already running
if not hasattr(tracking.vessel_matcher, "load_ais_data_full"):
    importlib.reload(tracking.vessel_matcher)
if not hasattr(tracking.suspect_scorer, "build_suspect_ranking"):
    importlib.reload(tracking.suspect_scorer)

from tracking.vessel_matcher import (
    VesselMatcherError,
    find_closest_vessel,
    load_ais_data,
    rank_vessels,
)

try:
    from tracking.vessel_matcher import load_ais_data_full
except ImportError:
    def load_ais_data_full(csv_path):
        return load_ais_data(csv_path, deduplicate=False)

from tracking.suspect_scorer import (
    build_suspect_ranking,
    build_incident_result,
    inspect_ais_capabilities,
    DEFAULT_WEIGHTS,
)


CLASSIFIER_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "models" / "efficientnet_b0_oil_classifier.pth"
)


@st.cache_resource(show_spinner=False)
def load_oil_spill_classifier(weights_path):
    """Load the fine-tuned classifier once per Streamlit process."""

    return EfficientNetOilClassifier.from_weights(weights_path)


EARTH_RADIUS_METERS = 6_371_000.0


def calculate_direction(east_m, north_m):
    """Return a compass direction from east/north displacement components."""

    if math.isclose(east_m, 0.0, abs_tol=1e-9) and math.isclose(
        north_m, 0.0, abs_tol=1e-9
    ):
        return "Stationary"

    angle_from_north = (math.degrees(math.atan2(east_m, north_m)) + 360) % 360
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return directions[int((angle_from_north + 22.5) // 45) % 8]


def format_geo_coordinate(value, positive_suffix, negative_suffix):
    """Format a geographic coordinate with a readable hemisphere suffix."""

    suffix = positive_suffix if value >= 0 else negative_suffix
    return f"{abs(value):.4f}° {suffix}"


def _fill_temporal_values(values, variable_name):
    """Fill limited temporal gaps without allowing NaNs into the forecast."""

    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f"NetCDF variable '{variable_name}' contains no valid values.")
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


def build_environmental_forcing(dataset, hours):
    """Prepare hourly current/wind forcing for the prototype trajectory.

    Spatial dimensions are averaged because this prototype does not yet perform
    spatial interpolation. If the NetCDF time series is static, a small smooth
    rotation of the existing combined vector is used and labelled as synthetic.
    """

    required_variables = ("u_current", "v_current", "u_wind", "v_wind")
    missing_variables = [
        name for name in required_variables if name not in dataset.data_vars
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise ValueError(f"NetCDF is missing expected variable(s): {missing}.")

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

        # The supplied dataset has a time dimension, but identical values at
        # every time slice. Keep the speed based on those values and apply only
        # a small smooth directional change for this demonstration.
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
            raise ValueError("Environmental forcing contains invalid values.")

    return {
        "forcing": forcing,
        "mode": mode,
        "synthetic_variation": synthetic_variation,
    }


def build_drift_trajectory(
    hours,
    net_u,
    net_v,
    origin_x,
    origin_y,
    crs,
    forcing_u=None,
    forcing_v=None,
):
    """Build cumulative physical displacement and display coordinates."""

    crs_obj = rasterio.crs.CRS.from_user_input(crs)
    is_geographic = crs_obj.is_geographic
    is_metric_projected = crs_obj.is_projected and getattr(
        crs_obj, "linear_units", None
    ) in ("metre", "meter", "m")

    if not is_geographic and not is_metric_projected:
        raise ValueError(
            "Tab 2 supports geographic CRS or projected CRS with metre units "
            "for displacement conversion."
        )

    if forcing_u is None:
        forcing_u = np.full(hours, net_u, dtype=float)
    if forcing_v is None:
        forcing_v = np.full(hours, net_v, dtype=float)
    if len(forcing_u) < hours or len(forcing_v) < hours:
        raise ValueError("Environmental forcing does not cover the forecast duration.")

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


st.set_page_config(page_title="Oil Spill Tracker", page_icon=None, layout="wide")

st.title("Maritime Oil Spill Detection & Vessel Tracking System")
st.caption("Production GIS Pipeline — Real GeoTIFF Spatial Extraction")

tab1, tab2, tab3 = st.tabs(["1. GeoTIFF Detection", "2. Drift Simulation", "3. Suspect Match"])

with tab1:
    st.subheader("1. Satellite GeoTIFF Analysis")
    classifier_runtime = runtime_status()
    st.markdown("### AI Classification")
    st.write("**Classifier:** EfficientNetB0 | **Task:** Oil-like vs Look-alike")
    st.write(f"**Device:** {classifier_runtime['device']}")
    if not classifier_runtime["available"]:
        st.warning(
            "EfficientNetB0 oil-spill classification is unavailable because "
            "PyTorch/torchvision is not installed. Classical candidate detection remains active."
        )
    elif not CLASSIFIER_WEIGHTS_PATH.is_file():
        st.info(
            "Classifier architecture available, but trained oil-spill weights are not loaded. "
            "Candidate classification is unavailable; the classical ranking will be used."
        )
    else:
        st.info(
            f"Trained-weight file found: `{CLASSIFIER_WEIGHTS_PATH.name}`. "
            "It will be loaded for candidate inference."
        )
    uploaded_file = st.file_uploader("Upload Real Satellite GeoTIFF (.tif)", type=["tif", "tiff"])
    
    if uploaded_file is not None:
        # Clear results from a previous upload before processing this raster.
        for key in [
            "real_x", "real_y", "crs", "spill_area", "spill_perimeter",
            "spill_length", "spill_width", "spill_aspect_ratio",
            "spill_orientation", "selected_candidate_score", "candidate_count",
        ]:
            st.session_state.pop(key, None)

        # Save uploaded buffer to a temp file so rasterio can read spatial headers
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            with rasterio.open(tmp_path) as dataset:
                # Read Band 1 for image processing
                band1 = dataset.read(1)
                
                # Robustly clip extreme pixel values before converting to 8-bit.
                # This keeps a few unusually bright/dark pixels from controlling
                # the scale used for the whole image.
                finite_band1 = band1[np.isfinite(band1)]
                normalization_available = finite_band1.size > 0
                if normalization_available:
                    lower = float(np.percentile(finite_band1, 2))
                    upper = float(np.percentile(finite_band1, 98))
                else:
                    lower = upper = 0.0

                if not normalization_available or not np.isfinite(lower) or not np.isfinite(upper):
                    band1_norm = np.zeros_like(band1, dtype=np.uint8)
                    normalization_available = False
                    st.warning("The GeoTIFF does not contain usable finite pixel values for normalization.")
                elif lower >= upper:
                    # A constant or near-constant raster has no contrast from
                    # which a dark-region candidate can be extracted.
                    band1_norm = np.zeros_like(band1, dtype=np.uint8)
                    normalization_available = False
                    st.warning("The raster has insufficient intensity variation for candidate detection.")
                else:
                    band_clipped = np.clip(
                        np.nan_to_num(band1, nan=lower, posinf=upper, neginf=lower),
                        lower,
                        upper,
                    )
                    band1_norm = np.uint8(
                        np.round((band_clipped - lower) / (upper - lower) * 255)
                    )

                # Median filtering reduces isolated SAR speckle while retaining
                # the boundaries of larger dark regions.
                filtered_image = cv2.medianBlur(band1_norm, 5)

                # Otsu chooses a threshold from this image rather than using a
                # fixed value that only works for one brightness distribution.
                if normalization_available:
                    otsu_threshold, otsu_mask = cv2.threshold(
                        filtered_image,
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                    )

                    # Keep morphology conservative for the small sample rasters.
                    morphology_kernel = np.ones((5, 5), np.uint8)
                    opened_mask = cv2.morphologyEx(
                        otsu_mask,
                        cv2.MORPH_OPEN,
                        morphology_kernel,
                    )
                    cleaned_mask = cv2.morphologyEx(
                        opened_mask,
                        cv2.MORPH_CLOSE,
                        morphology_kernel,
                    )
                    contours, _ = cv2.findContours(
                        cleaned_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                else:
                    otsu_threshold = None
                    cleaned_mask = np.zeros_like(band1_norm, dtype=np.uint8)
                    contours = []

                # Prototype parameters: relative to image size, with an
                # absolute floor so tiny speckle regions are not candidates.
                MIN_COMPONENT_AREA_RATIO = 0.0005
                MIN_COMPONENT_AREA_PIXELS = 20
                min_area = max(
                    MIN_COMPONENT_AREA_PIXELS,
                    int(dataset.height * dataset.width * MIN_COMPONENT_AREA_RATIO),
                )

                candidates = []
                image_median = float(np.median(filtered_image))
                for contour in contours:
                    try:
                        area = float(cv2.contourArea(contour))
                        if area < min_area:
                            continue

                        perimeter = float(cv2.arcLength(contour, True))
                        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
                        bbox_area = bbox_w * bbox_h
                        aspect_ratio = (
                            max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
                            if min(bbox_w, bbox_h) > 0
                            else None
                        )
                        compactness = area / bbox_area if bbox_area > 0 else 0.0

                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            centroid = (
                                int(moments["m10"] / moments["m00"]),
                                int(moments["m01"] / moments["m00"]),
                            )
                        else:
                            centroid = None

                        candidate_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
                        cv2.drawContours(candidate_mask, [contour], -1, 255, -1)
                        mean_intensity = float(
                            cv2.mean(filtered_image, mask=candidate_mask)[0]
                        )

                        # Compare the candidate with nearby pixels and the
                        # image median. The result is bounded to 0-1.
                        surrounding_mask = cv2.dilate(
                            candidate_mask,
                            morphology_kernel,
                            iterations=2,
                        )
                        surrounding_mask[candidate_mask > 0] = 0
                        surrounding_pixels = filtered_image[surrounding_mask > 0]
                        surrounding_mean = (
                            float(np.mean(surrounding_pixels))
                            if surrounding_pixels.size > 0
                            else image_median
                        )
                        reference_intensity = max(surrounding_mean, image_median)
                        relative_darkness = float(
                            np.clip(
                                (reference_intensity - mean_intensity)
                                / max(reference_intensity, 1.0),
                                0.0,
                                1.0,
                            )
                        )

                        border_touching = (
                            bbox_x <= 0
                            or bbox_y <= 0
                            or bbox_x + bbox_w >= dataset.width - 1
                            or bbox_y + bbox_h >= dataset.height - 1
                        )

                        candidates.append(
                            {
                                "contour": contour,
                                "area": area,
                                "perimeter": perimeter,
                                "bbox_x": bbox_x,
                                "bbox_y": bbox_y,
                                "bbox_w": bbox_w,
                                "bbox_h": bbox_h,
                                "aspect_ratio": aspect_ratio,
                                "centroid": centroid,
                                "mean_intensity": mean_intensity,
                                "surrounding_mean": surrounding_mean,
                                "relative_darkness": relative_darkness,
                                "border_touching": border_touching,
                                "compactness": compactness,
                            }
                        )
                    except (cv2.error, ValueError, ZeroDivisionError):
                        # Ignore malformed contours rather than stopping the app.
                        continue

                selected_candidate = None
                classifier_error = None
                ai_classification_available = False
                if candidates:
                    try:
                        max_candidate_area = max(candidate["area"] for candidate in candidates)
                        for candidate in candidates:
                            # Square-root scaling stops area from completely
                            # dominating the other normalized features.
                            area_score = float(
                                np.sqrt(candidate["area"] / max_candidate_area)
                            ) if max_candidate_area > 0 else 0.0
                            shape_score = float(np.clip(candidate["compactness"], 0.0, 1.0))
                            border_penalty = 1.0 if candidate["border_touching"] else 0.0

                            # This is a ranking score, not an oil probability.
                            candidate["area_score"] = area_score
                            candidate["darkness_score"] = candidate["relative_darkness"]
                            candidate["shape_score"] = shape_score
                            candidate["border_penalty"] = border_penalty
                            candidate["score"] = float(
                                np.clip(
                                    0.40 * area_score
                                    + 0.35 * candidate["darkness_score"]
                                    + 0.25 * shape_score
                                    - 0.20 * border_penalty,
                                    0.0,
                                    1.0,
                                )
                            )
                            candidate["classical_score"] = candidate["score"]

                        classifier = None
                        if (
                            CLASSIFIER_WEIGHTS_PATH.is_file()
                            and classifier_runtime["available"]
                        ):
                            try:
                                classifier = load_oil_spill_classifier(
                                    str(CLASSIFIER_WEIGHTS_PATH)
                                )
                            except ClassifierLoadError as exc:
                                classifier_error = str(exc)

                        if classifier is not None:
                            candidate_crop_padding = 8
                            for candidate in candidates:
                                crop_x0 = max(
                                    0,
                                    candidate["bbox_x"] - candidate_crop_padding,
                                )
                                crop_y0 = max(
                                    0,
                                    candidate["bbox_y"] - candidate_crop_padding,
                                )
                                crop_x1 = min(
                                    dataset.width,
                                    candidate["bbox_x"]
                                    + candidate["bbox_w"]
                                    + candidate_crop_padding,
                                )
                                crop_y1 = min(
                                    dataset.height,
                                    candidate["bbox_y"]
                                    + candidate["bbox_h"]
                                    + candidate_crop_padding,
                                )
                                crop = band1_norm[crop_y0:crop_y1, crop_x0:crop_x1]
                                try:
                                    prediction = classifier.predict(crop)
                                    candidate["crop"] = crop
                                    candidate.update(prediction)
                                except Exception as exc:
                                    candidate["classification_error"] = str(exc)

                        ai_classification_available = bool(
                            classifier is not None
                            and candidates
                            and all(
                                "oil_probability" in candidate
                                for candidate in candidates
                            )
                        )

                        if ai_classification_available:
                            # Keep the classical score and model probability
                            # separate. The model probability is used only as
                            # the experimental ranking signal when a trained
                            # checkpoint is actually loaded.
                            selected_candidate = max(
                                candidates,
                                key=lambda candidate: candidate["oil_probability"],
                            )
                        else:
                            selected_candidate = max(
                                candidates,
                                key=lambda candidate: candidate["classical_score"],
                            )
                    except (KeyError, ValueError, TypeError):
                        candidates = []
                        st.warning("Candidate scoring failed, so no spill candidate was selected.")

                if classifier_error:
                    st.warning(
                        "EfficientNetB0 weights could not be loaded. "
                        f"Classical candidate ranking remains active. Details: {classifier_error}"
                    )

                if selected_candidate is not None:
                    # Preserve the existing characterization interface.
                    largest_spill = selected_candidate["contour"]
                    pixel_area = float(selected_candidate["area"])

                    if pixel_area <= 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(
                                band1_norm,
                                caption=f"Raw Band 1 ({dataset.width}x{dataset.height} px)",
                                use_container_width=True,
                            )
                        with col2:
                            st.image(
                                cleaned_mask,
                                caption="Processed Oil Spill Mask (Otsu + morphology)",
                                use_container_width=True,
                            )
                        with col3:
                            st.image(
                                band1_norm,
                                caption="No valid candidate selected",
                                use_container_width=True,
                            )
                        st.warning("The largest detected contour has zero area, so it cannot be characterized.")
                    else:
                        if pixel_area < 1:
                            st.warning("The detected contour is very small; its geometric measurements may be unstable.")

                        # Basic geometry in image pixels.
                        pixel_perimeter = float(cv2.arcLength(largest_spill, True))
                        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest_spill)
                        rect = cv2.minAreaRect(largest_spill)
                        rect_center, (rect_w, rect_h), rect_angle = rect
                        length_pixels = float(max(rect_w, rect_h))
                        width_pixels = float(min(rect_w, rect_h))
                        aspect_ratio = (
                            length_pixels / width_pixels if width_pixels > 0 else None
                        )

                        # OpenCV's angle is tied to the rectangle's short/long side.
                        # Convert it to the major-axis orientation in [-90, 90) degrees.
                        orientation = float(rect_angle)
                        if rect_w < rect_h:
                            orientation += 90.0
                        if orientation >= 90.0:
                            orientation -= 180.0
                        elif orientation < -90.0:
                            orientation += 180.0

                        # Calculate the centroid, while keeping the existing pixel-to-map
                        # coordinate conversion for the other tabs.
                        moments = cv2.moments(largest_spill)
                        centroid_available = moments["m00"] != 0
                        px = py = None
                        real_x = real_y = None
                        if centroid_available:
                            px = int(moments["m10"] / moments["m00"])
                            py = int(moments["m01"] / moments["m00"])
                            real_x, real_y = dataset.xy(py, px)

                        # Determine whether raster resolution can convert pixel geometry
                        # into map units. Do not label geographic degrees as metres.
                        pixel_width, pixel_height = (abs(value) for value in dataset.res)
                        resolution_available = all(
                            np.isfinite(value) and value > 0
                            for value in (pixel_width, pixel_height)
                        )
                        crs = dataset.crs
                        crs_text = str(crs) if crs else "CRS unavailable"
                        is_projected = bool(crs and crs.is_projected)
                        linear_units = getattr(crs, "linear_units", None) if crs else None
                        is_metric_projected = is_projected and linear_units in ("metre", "meter", "m")
                        can_convert_to_map_units = resolution_available and is_projected

                        map_unit = linear_units or "map units"
                        pixel_scale = (pixel_width + pixel_height) / 2
                        area_map_units = pixel_area * pixel_width * pixel_height
                        perimeter_map_units = pixel_perimeter * pixel_scale
                        length_map_units = length_pixels * pixel_scale
                        width_map_units = width_pixels * pixel_scale

                        # Draw all retained candidates, then highlight the selected
                        # candidate with the existing characterization geometry.
                        overlay = cv2.cvtColor(band1_norm, cv2.COLOR_GRAY2RGB)
                        for candidate in candidates:
                            if candidate is not selected_candidate:
                                cv2.drawContours(
                                    overlay,
                                    [candidate["contour"]],
                                    -1,
                                    (255, 200, 0),
                                    1,
                                )
                        cv2.drawContours(overlay, [largest_spill], -1, (0, 255, 0), 2)
                        cv2.rectangle(
                            overlay,
                            (bbox_x, bbox_y),
                            (bbox_x + bbox_w, bbox_y + bbox_h),
                            (255, 0, 0),
                            1,
                        )
                        rotated_box = cv2.boxPoints(rect).astype(np.int32)
                        cv2.polylines(overlay, [rotated_box], True, (255, 165, 0), 2)
                        cv2.putText(
                            overlay,
                            "Top-ranked",
                            (bbox_x, max(15, bbox_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        if centroid_available:
                            cv2.circle(overlay, (px, py), 5, (255, 0, 255), -1)
                            cv2.putText(
                                overlay,
                                "centroid",
                                (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(
                                band1_norm,
                                caption=f"Raw Band 1 ({dataset.width}x{dataset.height} px)",
                                use_container_width=True,
                            )
                        with col2:
                            st.image(
                                cleaned_mask,
                                caption="Processed Oil Spill Mask (Otsu + morphology)",
                                use_container_width=True,
                            )
                        with col3:
                            st.image(
                                overlay,
                                caption="Candidates and selected spill geometry",
                                use_container_width=True,
                            )

                        if otsu_threshold is not None:
                            st.caption(
                                f"Otsu threshold: {otsu_threshold:.2f}/255 | "
                                f"Minimum candidate area: {min_area} px² | "
                                f"Retained candidates: {len(candidates)}"
                            )

                        st.info(
                            "Detection note: Dark regions in SAR imagery are not necessarily oil. "
                            "Low-wind areas, land/coastal effects, shadows, ships, and other SAR "
                            "look-alikes can also appear dark. This prototype uses classical "
                            "image-processing heuristics and is not a scientifically validated "
                            "oil-spill classifier."
                        )

                        st.markdown("### Detection Candidates")
                        ranking_key = (
                            "oil_probability"
                            if ai_classification_available
                            else "classical_score"
                        )
                        candidate_rows = []
                        for rank, candidate in enumerate(
                            sorted(candidates, key=lambda item: item[ranking_key], reverse=True),
                            start=1,
                        ):
                            row = {
                                "Candidate": f"C{rank}",
                                "Top Rank": rank,
                                "Top-Ranked": "Yes" if candidate is selected_candidate else "",
                                "Classical Score": round(candidate["classical_score"], 3),
                                "Area (px²)": round(candidate["area"], 2),
                                "Aspect Ratio": (
                                    round(candidate["aspect_ratio"], 2)
                                    if candidate["aspect_ratio"] is not None
                                    else None
                                ),
                                "Mean Intensity": round(candidate["mean_intensity"], 2),
                                "Compactness": round(candidate["compactness"], 3),
                                "Border Touching": candidate["border_touching"],
                            }
                            if ai_classification_available:
                                row.update(
                                    {
                                        "Oil-like Probability": round(
                                            candidate["oil_probability"],
                                            3,
                                        ),
                                        "Look-alike Probability": round(
                                            candidate["look_alike_probability"],
                                            3,
                                        ),
                                        "Classification": candidate["classification"],
                                    }
                                )
                            candidate_rows.append(row)
                        st.dataframe(
                            pd.DataFrame(candidate_rows),
                            hide_index=True,
                            use_container_width=True,
                        )
                        if ai_classification_available:
                            st.caption(
                                "Candidates are ranked by EfficientNetB0 oil-like probability "
                                "because a trained classifier is loaded. The classical score "
                                "remains separate and is not an oil probability."
                            )
                            st.markdown("### Top-Ranked Candidate")
                            top_candidate_col, top_candidate_info = st.columns([1, 2])
                            with top_candidate_col:
                                st.image(
                                    selected_candidate["crop"],
                                    caption="Candidate crop supplied to EfficientNetB0",
                                    use_container_width=True,
                                )
                            with top_candidate_info:
                                st.write(
                                    f"**Classification:** {selected_candidate['classification']}"
                                )
                                st.write(
                                    f"**Oil-like probability:** "
                                    f"{selected_candidate['oil_probability']:.3f}"
                                )
                                st.write(
                                    f"**Look-alike probability:** "
                                    f"{selected_candidate['look_alike_probability']:.3f}"
                                )
                                st.write(
                                    f"**Classical candidate score:** "
                                    f"{selected_candidate['classical_score']:.3f}"
                                )
                            st.info(
                                "EfficientNetB0 classification is meaningful for this task only "
                                "after training and validation on representative labelled SAR data."
                            )
                        else:
                            st.caption(
                                "AI Classification: Not available. The selected candidate uses "
                                "the classical candidate score."
                            )

                        st.markdown("### Spill Characterization")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            if is_metric_projected and can_convert_to_map_units:
                                st.metric("Area", f"{area_map_units:,.2f} m²")
                                st.caption(f"{area_map_units / 1_000_000:,.6f} km²")
                            elif can_convert_to_map_units:
                                st.metric("Area", f"{area_map_units:,.2f} {map_unit}²")
                                st.caption(f"Pixel area: {pixel_area:,.2f} px²")
                            else:
                                st.metric("Area", f"{pixel_area:,.2f} px²")
                                st.caption("Real-world area unavailable for this CRS")
                        with metric_col2:
                            if is_metric_projected and can_convert_to_map_units:
                                st.metric("Perimeter", f"{perimeter_map_units:,.2f} m")
                            elif can_convert_to_map_units:
                                st.metric("Perimeter", f"{perimeter_map_units:,.2f} {map_unit}")
                            else:
                                st.metric("Perimeter", f"{pixel_perimeter:,.2f} px")
                                st.caption("Approximate contour perimeter")
                        with metric_col3:
                            st.metric(
                                "Aspect Ratio",
                                f"{aspect_ratio:.2f}" if aspect_ratio is not None else "Unavailable",
                            )
                            st.caption("~1 = compact; larger = more elongated")

                        metric_col4, metric_col5, metric_col6 = st.columns(3)
                        with metric_col4:
                            if can_convert_to_map_units:
                                unit_label = "m" if is_metric_projected else map_unit
                                st.metric("Approx. Length", f"{length_map_units:,.2f} {unit_label}")
                            else:
                                st.metric("Approx. Length", f"{length_pixels:,.2f} px")
                            st.caption("Minimum-area rotated rectangle")
                        with metric_col5:
                            if can_convert_to_map_units:
                                unit_label = "m" if is_metric_projected else map_unit
                                st.metric("Approx. Width", f"{width_map_units:,.2f} {unit_label}")
                            else:
                                st.metric("Approx. Width", f"{width_pixels:,.2f} px")
                            st.caption("Minimum-area rotated rectangle")
                        with metric_col6:
                            st.metric("Orientation", f"{orientation:.1f}°")
                            st.caption("Major-axis angle, not movement direction")

                        st.markdown("### Spatial Information")
                        spatial_col1, spatial_col2 = st.columns(2)
                        with spatial_col1:
                            if centroid_available:
                                st.write(f"**Pixel centroid:** ({px}, {py})")
                                st.write(f"**Spatial centroid:** ({real_x:.2f}, {real_y:.2f})")
                            else:
                                st.warning("Centroid could not be calculated for this contour.")
                            st.write(
                                f"**Bounding box (pixels):** X={bbox_x}, Y={bbox_y}, "
                                f"Width={bbox_w}, Height={bbox_h}"
                            )
                            if can_convert_to_map_units:
                                bbox_width_map = bbox_w * pixel_width
                                bbox_height_map = bbox_h * pixel_height
                                bbox_unit = "m" if is_metric_projected else map_unit
                                st.write(
                                    f"**Bounding box (approx.):** "
                                    f"Width={bbox_width_map:,.2f} {bbox_unit}, "
                                    f"Height={bbox_height_map:,.2f} {bbox_unit}"
                                )
                        with spatial_col2:
                            st.write(f"**CRS:** {crs_text}")
                            if resolution_available:
                                st.write(
                                    f"**Raster resolution:** {pixel_width:g} × {pixel_height:g} {map_unit}/pixel"
                                )
                            else:
                                st.write("**Raster resolution:** unavailable")

                        if not resolution_available:
                            st.info("The raster resolution is unavailable or invalid, so only pixel measurements are shown.")
                        elif not crs:
                            st.info("No CRS is attached. Spatial coordinates are shown from the raster transform, but their units are unknown.")
                        elif not is_projected:
                            st.info(
                                "This raster uses a geographic CRS (usually degrees). Pixel measurements are shown; "
                                "accurate m²/meter values require geodesic area and distance calculations."
                            )
                        elif not is_metric_projected:
                            st.info(
                                f"The projected CRS uses {map_unit}, not metres. Measurements are shown in native map units."
                            )

                        # Preserve the existing cross-tab coordinate contract and expose
                        # the new measurements for later pipeline stages.
                        if centroid_available:
                            st.session_state["real_x"] = real_x
                            st.session_state["real_y"] = real_y
                        st.session_state["crs"] = crs_text
                        st.session_state["spill_area"] = area_map_units if can_convert_to_map_units else pixel_area
                        st.session_state["spill_perimeter"] = perimeter_map_units if can_convert_to_map_units else pixel_perimeter
                        st.session_state["spill_length"] = length_map_units if can_convert_to_map_units else length_pixels
                        st.session_state["spill_width"] = width_map_units if can_convert_to_map_units else width_pixels
                        st.session_state["spill_aspect_ratio"] = aspect_ratio
                        st.session_state["spill_orientation"] = orientation
                        st.session_state["selected_candidate_score"] = selected_candidate["score"]
                        st.session_state["candidate_count"] = len(candidates)

                        if centroid_available:
                            st.success(
                                f"Spatial centroid extracted! Pixel: ({px}, {py}) | "
                                f"Coordinates: ({real_x:.2f}, {real_y:.2f}) | CRS: {crs_text}"
                            )
                        else:
                            st.warning("Geometry was characterized, but no valid centroid was available for Tab 2/Tab 3.")

                else:
                    # Keep all three views visible even when no candidate survives.
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(
                            band1_norm,
                            caption=f"Raw Band 1 ({dataset.width}x{dataset.height} px)",
                            use_container_width=True,
                        )
                    with col2:
                        st.image(
                            cleaned_mask,
                            caption="Processed Oil Spill Mask (Otsu + morphology)",
                            use_container_width=True,
                        )
                    with col3:
                        st.image(
                            band1_norm,
                            caption="No valid candidate selected",
                            use_container_width=True,
                        )

                    if otsu_threshold is not None:
                        st.caption(
                            f"Otsu threshold: {otsu_threshold:.2f}/255 | "
                            f"Minimum candidate area: {min_area} px²"
                        )
                    if not contours:
                        st.warning("No contours were detected after Otsu segmentation and morphological cleanup.")
                    elif not candidates:
                        st.warning(
                            "No valid spill candidates were found after filtering. "
                            "Try another SAR image or adjust the detection parameters."
                        )
                    else:
                        st.warning("Candidate scoring failed, so no spill candidate was selected.")
                    st.info(
                        "Detection note: Dark regions in SAR imagery are not necessarily oil. "
                        "Low-wind areas, land/coastal effects, shadows, ships, and other SAR "
                        "look-alikes can also appear dark. This prototype uses classical "
                        "image-processing heuristics and is not a scientifically validated "
                        "oil-spill classifier."
                    )

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

import matplotlib.pyplot as plt

with tab2:
    st.markdown(
        """
        <style>
        /* Single light visual system shared by all three tabs. */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --app-bg: #F5F6F7;
            --card-bg: #FFFFFF;
            --text: #202124;
            --muted: #6B7075;
            --border: #E7E8EA;
            --accent: #C8FF3D;
            --control-track: #E2E4E6;
        }

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMain"],
        [data-testid="stSidebar"] {
            background-color: var(--app-bg) !important;
            color: var(--text) !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] *,
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        /* Preserve Streamlit's icon font; otherwise material icons render as text. */
        [data-testid="stIconMaterial"],
        span.material-icons,
        span.material-symbols-rounded {
            font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
        }

        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stHeading"] h1,
        [data-testid="stHeading"] h2,
        [data-testid="stHeading"] h3,
        [data-testid="stHeading"] h4 {
            color: var(--text) !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        [data-testid="stAppViewContainer"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stWidgetLabel"] * {
            color: var(--muted) !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b {
            color: var(--text);
        }

        .main .block-container {
            max-width: 1440px;
            padding: 2rem 3rem 2.5rem;
        }

        [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid var(--border);
        }
        [data-baseweb="tab"] {
            color: var(--muted) !important;
            font-size: 0.875rem;
            font-weight: 500;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: var(--text) !important;
        }
        [data-baseweb="tab-highlight"] {
            background: var(--accent) !important;
            height: 2px !important;
        }
        [data-baseweb="tab-border"] {
            background: var(--border) !important;
        }
        [data-testid="stTab"][role="tab"] {
            color: var(--muted) !important;
            font-size: 0.875rem;
            font-weight: 500;
        }
        [data-testid="stTab"][role="tab"][data-selected="true"] {
            color: var(--text) !important;
        }
        [data-testid="stTab"] .react-aria-SelectionIndicator {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px;
            box-shadow: 0 4px 16px rgba(32, 33, 36, 0.04);
        }
        [data-testid="stMetric"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 13px;
            padding: 0.8rem 0.9rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: var(--muted) !important;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: var(--text) !important;
            font-size: 1.35rem;
            font-weight: 700;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 13px;
            overflow: hidden;
        }

        [data-testid="stSlider"] label,
        [data-testid="stSlider"] label * {
            color: var(--muted) !important;
        }
        [data-baseweb="slider"] > div > div {
            background: var(--control-track) !important;
        }
        [data-baseweb="slider"] > div > div > div {
            background: var(--accent) !important;
        }
        [data-baseweb="slider"] [role="slider"] {
            background: var(--text) !important;
            border: 2px solid var(--text) !important;
            box-shadow: 0 0 0 3px var(--accent) !important;
        }
        [data-baseweb="slider"] [role="slider"]:focus {
            box-shadow: 0 0 0 3px var(--accent) !important;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {
            background: var(--card-bg) !important;
            border: 1px dashed #D5D8DB !important;
            border-radius: 13px !important;
        }
        [data-testid="stFileUploader"] button {
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
            background: var(--card-bg) !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            color: var(--muted) !important;
        }

        [data-testid="stButton"] button[kind="primary"] {
            background: var(--text) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--text) !important;
            border-radius: 11px !important;
        }
        [data-testid="stButton"] button[kind="secondary"],
        button[kind="secondary"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 11px !important;
            color: var(--text) !important;
        }

        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid #DCEAF7 !important;
            background: #F1F7FF !important;
        }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
            color: #4B5563 !important;
        }
        [data-testid="stExpander"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 13px !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("2. Spill Drift Simulation")
    st.caption("Environmental forcing → cumulative drift trajectory → forecast movement")

    if "real_x" in st.session_state and "real_y" in st.session_state:
        rx = float(st.session_state["real_x"])
        ry = float(st.session_state["real_y"])
        spill_crs = st.session_state.get("crs", "CRS unavailable")

        try:
            spill_crs_obj = rasterio.crs.CRS.from_user_input(spill_crs)
            geographic_origin = spill_crs_obj.is_geographic
        except Exception as exc:
            st.error(f"The spill CRS could not be interpreted: {exc}")
            geographic_origin = False
            spill_crs_obj = None

        try:
            with st.container(border=True):
                st.markdown("#### Forecast Controls")
                control_col1, control_col2 = st.columns(2)
                with control_col1:
                    hours = st.slider(
                        "Forecast horizon (hours)",
                        1,
                        48,
                        24,
                        key="tab2_forecast_horizon",
                    )
                with control_col2:
                    screening_radius_km = st.slider(
                        "Investigation radius (km)",
                        0.5,
                        20.0,
                        5.0,
                        0.5,
                        key="tab2_investigation_radius",
                    )

            ds = xr.open_dataset("ocean_currents.nc")
            environmental_result = build_environmental_forcing(ds, hours)
            environmental_forcing = environmental_result["forcing"]
            forcing_mode = environmental_result["mode"]
            synthetic_variation = environmental_result["synthetic_variation"]
            u_c = float(environmental_forcing["u_current"][0])
            v_c = float(environmental_forcing["v_current"][0])
            u_w = float(environmental_forcing["u_wind"][0])
            v_w = float(environmental_forcing["v_wind"][0])
            net_u = float(environmental_forcing["net_u"][0])
            net_v = float(environmental_forcing["net_v"][0])
            average_drift_speed = float(
                np.mean(
                    np.hypot(
                        environmental_forcing["net_u"],
                        environmental_forcing["net_v"],
                    )
                )
            )

            origin_col, environment_col = st.columns(2)
            with origin_col:
                with st.container(border=True):
                    st.markdown("#### Spill Origin")
                    origin_metric_col1, origin_metric_col2 = st.columns(2)
                    with origin_metric_col1:
                        if geographic_origin:
                            st.metric("Longitude", format_geo_coordinate(rx, "E", "W"))
                        else:
                            origin_units = getattr(
                                spill_crs_obj,
                                "linear_units",
                                "map units",
                            )
                            st.metric("X / Easting", f"{rx:,.2f} {origin_units}")
                    with origin_metric_col2:
                        if geographic_origin:
                            st.metric("Latitude", format_geo_coordinate(ry, "N", "S"))
                        else:
                            origin_units = getattr(
                                spill_crs_obj,
                                "linear_units",
                                "map units",
                            )
                            st.metric("Y / Northing", f"{ry:,.2f} {origin_units}")
                    st.caption(f"CRS: {spill_crs}")

            with environment_col:
                with st.container(border=True):
                    st.markdown("#### Environmental Conditions")
                    environment_metric_col1, environment_metric_col2, environment_metric_col3 = st.columns(3)
                    with environment_metric_col1:
                        st.metric(
                            "Ocean Current",
                            f"{u_c:.2f} m/s E | {v_c:.2f} m/s N",
                        )
                    with environment_metric_col2:
                        st.metric(
                            "Surface Wind",
                            f"{u_w:.1f} m/s E | {v_w:.1f} m/s N",
                        )
                    with environment_metric_col3:
                        net_speed = math.hypot(net_u, net_v)
                        st.metric("Combined Drift", f"{net_speed:.2f} m/s")
                    st.caption(
                        "Combined drift = ocean current + 3% of wind. "
                        f"Forcing mode: {forcing_mode}."
                    )

            trajectory_df = build_drift_trajectory(
                hours,
                net_u,
                net_v,
                rx,
                ry,
                spill_crs,
                forcing_u=environmental_forcing["net_u"],
                forcing_v=environmental_forcing["net_v"],
            )
            final_point = trajectory_df.iloc[-1]
            total_drift_km = float(final_point["Total_Drift_km"])
            direction = calculate_direction(
                float(final_point["East_Drift_km"]),
                float(final_point["North_Drift_km"]),
            )

            with st.container(border=True):
                st.markdown(f"#### {hours}-Hour Forecast")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric("Predicted Movement", f"{total_drift_km:.1f} km")
                with summary_col2:
                    st.metric("Direction", direction)
                with summary_col3:
                    st.metric("Average Drift Speed", f"{average_drift_speed:.2f} m/s")
                with summary_col4:
                    st.metric("Forecast Horizon", f"{hours} h")

                st.info(
                    f"Predicted spill movement: {total_drift_km:.1f} km over {hours} hours."
                )

                visualization_col, checkpoint_col = st.columns([3, 2])
                with visualization_col:
                    st.markdown("##### Predicted Spill Movement")
                    figure, axis = plt.subplots(figsize=(7, 4.6))
                    figure.patch.set_facecolor("#FFFFFF")
                    axis.set_facecolor("#FFFFFF")
                    axis.plot(
                        trajectory_df["East_Drift_km"],
                        trajectory_df["North_Drift_km"],
                        color="#202124",
                        linestyle="-",
                        linewidth=2.2,
                        label="Spill Movement",
                    )
                    axis.scatter(
                        0,
                        0,
                        color="#202124",
                        edgecolors="#C8FF3D",
                        linewidths=2,
                        s=90,
                        zorder=5,
                        label="Spill Start",
                    )
                    axis.scatter(
                        final_point["East_Drift_km"],
                        final_point["North_Drift_km"],
                        color="#C8FF3D",
                        edgecolors="#202124",
                        linewidths=1.4,
                        s=90,
                        zorder=5,
                        label="Predicted Position",
                    )
                    axis.annotate(
                        f"+{hours}h · {total_drift_km:.1f} km",
                        (final_point["East_Drift_km"], final_point["North_Drift_km"]),
                        textcoords="offset points",
                        xytext=(8, 8),
                        ha="left",
                        color="#202124",
                        fontsize=8,
                        weight="bold",
                    )
                    axis.set_xlabel("Distance East (km)", color="#777B80")
                    axis.set_ylabel("Distance North (km)", color="#777B80")
                    axis.tick_params(colors="#777B80", labelsize=8)
                    axis.margins(0.15)
                    for spine in axis.spines.values():
                        spine.set_color("#E7E8EA")
                    axis.grid(True, linestyle="-", color="#E7E8EA", alpha=0.8)
                    axis.legend(
                        facecolor="#FFFFFF",
                        edgecolor="#E7E8EA",
                        labelcolor="#202124",
                        frameon=True,
                        fontsize=8,
                    )
                    figure.tight_layout()
                    st.pyplot(figure, use_container_width=True)

                with checkpoint_col:
                    st.markdown("##### Forecast Checkpoints")
                    checkpoint_hours = [
                        checkpoint
                        for checkpoint in (0, 6, 12, 24, 36, 48)
                        if checkpoint <= hours
                    ]
                    if hours not in checkpoint_hours:
                        checkpoint_hours.append(hours)
                        checkpoint_hours.sort()
                    checkpoint_df = trajectory_df[
                        trajectory_df["Hour"].isin(checkpoint_hours)
                    ][
                        ["Hour", "East_Drift_km", "North_Drift_km", "Total_Drift_km"]
                    ].rename(
                        columns={
                            "East_Drift_km": "East (km)",
                            "North_Drift_km": "North (km)",
                            "Total_Drift_km": "Total Movement (km)",
                        }
                    )
                    st.dataframe(
                        checkpoint_df.round(2),
                        hide_index=True,
                        use_container_width=True,
                    )

            with st.expander("Detailed Forecast Data"):
                detailed_df = trajectory_df.rename(
                    columns={
                        "East_Drift_km": "East Drift (km)",
                        "North_Drift_km": "North Drift (km)",
                        "Total_Drift_km": "Total Drift (km)",
                    }
                )
                st.dataframe(
                    detailed_df.round(6),
                    height=320,
                    hide_index=True,
                    use_container_width=True,
                )

            st.info(
                "Prototype simulation using current and wind inputs from the supplied "
                "NetCDF dataset. This is not an operational hydrodynamic forecast."
                + (
                    " Temporal environmental variation is simulated for prototype demonstration."
                    if synthetic_variation
                    else ""
                )
            )

        except FileNotFoundError:
            st.error("Missing 'ocean_currents.nc'. Run python generate_metocean.py first!")
        except (KeyError, ValueError, rasterio.errors.RasterioError) as exc:
            st.error(f"Tab 2 could not build the drift forecast: {exc}")
    else:
        st.warning("Please upload a .tif file in Tab 1 first.")

with tab3:
    st.subheader("3. Vessel Tracking & Suspect Matching")
    st.info(
        "Prototype Mode: AIS positions shown here are synthetic/mock data for demonstration. "
        "They are not live Marine Cadastre observations."
    )

    if "real_x" not in st.session_state or "real_y" not in st.session_state:
        st.warning("Please upload a `.tif` file in Tab 1 first.")
    else:
        rx = float(st.session_state["real_x"])
        ry = float(st.session_state["real_y"])
        spill_crs = st.session_state.get("crs", "CRS unavailable")

        is_geographic = False
        spill_crs_obj = None
        try:
            spill_crs_obj = rasterio.crs.CRS.from_user_input(spill_crs)
            is_geographic = spill_crs_obj.is_geographic
        except Exception:
            pass

        origin_x_label = "Longitude" if is_geographic else "X / Easting"
        origin_y_label = "Latitude" if is_geographic else "Y / Northing"

        st.markdown("### Spill Origin")
        origin_col1, origin_col2, origin_col3 = st.columns(3)
        with origin_col1:
            st.metric(origin_x_label, f"{rx:.6f}" if is_geographic else f"{rx:.2f}")
        with origin_col2:
            st.metric(origin_y_label, f"{ry:.6f}" if is_geographic else f"{ry:.2f}")
        with origin_col3:
            st.metric("CRS", str(spill_crs))

        st.caption("AIS data source: Synthetic / Mock AIS Dataset")
        screening_radius_km = st.slider(
            "Screening Radius (km)",
            min_value=0.5,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Used as the reference scale for distance evidence decay; all vessels remain ranked.",
        )

        try:
            ais_df = load_ais_data("marine_cadastre_ais.csv")
            ais_raw_df = load_ais_data_full("marine_cadastre_ais.csv")
            duplicate_count = int(ais_df.attrs.get("duplicate_count", 0))
            ranked_vessels_df, distance_method, invalid_count = rank_vessels(
                ais_df,
                rx,
                ry,
                spill_crs,
                screening_radius_m=screening_radius_km * 1000,
            )
        except VesselMatcherError as exc:
            st.error(f"AIS matching unavailable: {exc}")
        else:
            if duplicate_count:
                st.warning(f"Ignored {duplicate_count} duplicate MMSI row(s).")
            if invalid_count:
                st.warning(f"Ignored {invalid_count} AIS row(s) with invalid coordinates.")

            # --- Determine spill origin in EPSG:4326 for scoring ---
            if is_geographic:
                spill_lon, spill_lat = rx, ry
            elif spill_crs_obj is not None:
                from rasterio.warp import transform as transform_coordinates_tab3
                spill_lon_list, spill_lat_list = transform_coordinates_tab3(
                    spill_crs_obj, "EPSG:4326", [rx], [ry],
                )
                spill_lon, spill_lat = spill_lon_list[0], spill_lat_list[0]
            else:
                spill_lon, spill_lat = rx, ry

            # --- Build multi-factor suspect ranking ---
            suspect_df = build_suspect_ranking(
                ranked_vessels_df,
                ais_raw_df,
                spill_lat,
                spill_lon,
                screening_radius_km,
            )
            capabilities = suspect_df.attrs.get(
                "ais_capabilities",
                inspect_ais_capabilities(ais_raw_df),
            )

            # Top-ranked vessel by suspicion score (not distance)
            top_suspect = suspect_df.iloc[0] if not suspect_df.empty else None
            # Independently closest vessel by distance
            closest_vessel = find_closest_vessel(ranked_vessels_df)
            vessels_in_radius = suspect_df[suspect_df["Within_Screening_Radius"]]

            # ---- Summary Cards ----
            st.markdown("### Summary")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("Total Vessels Analyzed", len(suspect_df))
            with summary_col2:
                st.metric(
                    "Highest-Priority Vessel",
                    str(top_suspect["VesselName"]) if top_suspect is not None else "None",
                )
            with summary_col3:
                st.metric(
                    "Suspicion Score",
                    f"{top_suspect['Suspicion_Score']:.1f} / 100"
                    if top_suspect is not None
                    else "—",
                )
            with summary_col4:
                st.metric(
                    "Closest Distance",
                    str(closest_vessel["Distance_Formatted"])
                    if closest_vessel is not None
                    else "None",
                )

            # ---- Highest-Priority Vessel Section ----
            if top_suspect is not None:
                st.markdown("### Highest-Priority Vessel")

                score_value = float(top_suspect["Suspicion_Score"])
                if score_value >= 70:
                    st.error(
                        f"Investigation Priority: HIGH — "
                        f"Suspicion Score {score_value:.1f} / 100"
                    )
                elif score_value >= 40:
                    st.warning(
                        f"Investigation Priority: MODERATE — "
                        f"Suspicion Score {score_value:.1f} / 100"
                    )
                else:
                    st.info(
                        f"Investigation Priority: LOW — "
                        f"Suspicion Score {score_value:.1f} / 100"
                    )

                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write(f"**Vessel:** {top_suspect['VesselName']}")
                    st.write(f"**MMSI:** {top_suspect['MMSI']}")
                    st.write(f"**Type:** {top_suspect['VesselType']}")
                    if "Timestamp" in suspect_df.columns and pd.notna(top_suspect.get("Timestamp")):
                        st.write(
                            f"**AIS timestamp:** "
                            f"{top_suspect['Timestamp'].strftime('%Y-%m-%d %H:%M UTC')}"
                        )
                with detail_col2:
                    st.write(f"**Distance from spill origin:** {top_suspect['Distance_Formatted']}")
                    if pd.notna(top_suspect.get("Speed_Knots")):
                        st.write(f"**Speed (SOG):** {top_suspect['Speed_Knots']:.1f} knots")
                    else:
                        st.write("**Speed (SOG):** Unavailable")
                    if top_suspect.get("AIS_Gap_Detected") is not None:
                        gap_text = "Detected" if top_suspect["AIS_Gap_Detected"] else "No gap detected"
                        st.write(f"**AIS gap:** {gap_text}")
                    else:
                        st.write("**AIS gap:** Not available (single AIS record)")
                    if pd.notna(top_suspect.get("Trajectory_Evidence")):
                        traj_level = (
                            "High" if top_suspect["Trajectory_Evidence"] >= 0.7
                            else "Moderate" if top_suspect["Trajectory_Evidence"] >= 0.3
                            else "Low"
                        )
                        st.write(f"**Trajectory relationship:** {traj_level}")
                    else:
                        st.write("**Trajectory relationship:** Not available (single AIS record)")
                    vtype_score = top_suspect.get("Vessel_Type_Evidence", 0.5)
                    vtype_level = (
                        "High" if vtype_score >= 0.8
                        else "Moderate" if vtype_score >= 0.5
                        else "Low"
                    )
                    st.write(f"**Vessel type relevance:** {vtype_level}")

                st.caption(
                    "This score prioritizes vessels for investigation based on available "
                    "AIS evidence. It does not establish responsibility for the spill."
                )

                # ---- Evidence Breakdown ----
                st.markdown("### Suspicion Score Breakdown")
                breakdown = top_suspect.get("_evidence_breakdown", {})
                evidence_label_map = {
                    "distance": "Distance from origin",
                    "trajectory": "Trajectory relationship",
                    "speed": "Speed / behaviour",
                    "ais_gap": "AIS gap",
                    "vessel_type": "Vessel type",
                }
                breakdown_rows = []
                for signal_key, label in evidence_label_map.items():
                    component = breakdown.get(signal_key, {})
                    if component.get("available", False):
                        score_display = f"{component['score']:.3f}"
                    else:
                        score_display = "Not available"
                    weight_pct = f"{DEFAULT_WEIGHTS.get(signal_key, 0) * 100:.0f}%"
                    breakdown_rows.append({
                        "Evidence": label,
                        "Score": score_display,
                        "Weight": weight_pct,
                    })
                breakdown_rows.append({
                    "Evidence": "Final Suspicion Score",
                    "Score": "",
                    "Weight": f"{score_value:.1f}",
                })
                st.dataframe(
                    pd.DataFrame(breakdown_rows),
                    hide_index=True,
                    use_container_width=True,
                )

                avail_count = sum(
                    1 for c in breakdown.values() if c.get("available", False)
                )
                total_count = len(evidence_label_map)
                st.caption(f"Evidence available: {avail_count} / {total_count}")

            # ---- Evidence Availability Section ----
            st.markdown("### Evidence Availability")
            capability_labels = {
                "distance": ("AIS geographic distance", capabilities.get("distance", False)),
                "speed": ("Speed over ground (SOG)", capabilities.get("speed", False)),
                "vessel_type": ("Vessel type classification", capabilities.get("vessel_type", False)),
                "trajectory": ("AIS historical trajectory", capabilities.get("trajectory", False)),
                "ais_gap": ("AIS gap analysis", capabilities.get("ais_gap", False)),
                "has_timestamp": ("Timestamps", capabilities.get("has_timestamp", False)),
                "has_course": ("Course / heading", capabilities.get("has_course", False)),
                "has_multi_record": ("Multiple records per vessel", capabilities.get("has_multi_record", False)),
            }
            avail_items = []
            for cap_key, (cap_label, cap_present) in capability_labels.items():
                status = "Available" if cap_present else "Unavailable"
                avail_items.append(f"**{cap_label}:** {status}")
            st.markdown("  \n".join(avail_items))

            if not capabilities.get("has_multi_record", False):
                st.caption(
                    "Trajectory and AIS-gap analysis require multiple AIS records per vessel. "
                    "The current dataset contains only one position per MMSI."
                )

            # ---- Screening Radius Summary ----
            if vessels_in_radius.empty:
                st.warning("No vessels detected within the screening radius.")
            else:
                st.success(
                    f"{len(vessels_in_radius)} vessel(s) fall within the "
                    f"{screening_radius_km:.1f} km screening radius."
                )

            st.caption(f"Distance method: {distance_method}")

            # ---- Suspect Ranking Table ----
            st.markdown("### Suspect Ranking")

            # Build display columns dynamically based on what is available
            display_columns = ["Rank", "VesselName", "MMSI", "VesselType", "Suspicion_Score"]
            rename_map = {
                "VesselName": "Vessel",
                "VesselType": "Type",
                "Suspicion_Score": "Suspicion Score",
                "Distance_Formatted": "Distance",
                "Within_Screening_Radius": "In Screening Radius",
            }
            display_columns.append("Distance_Formatted")

            if capabilities.get("speed", False) and "Speed_Knots" in suspect_df.columns:
                display_columns.append("Speed_Knots")
                rename_map["Speed_Knots"] = "SOG (knots)"

            display_columns.append("Status")

            if capabilities.get("trajectory", False) and "Trajectory_Evidence" in suspect_df.columns:
                display_columns.append("Trajectory_Evidence")
                rename_map["Trajectory_Evidence"] = "Trajectory"

            if capabilities.get("ais_gap", False) and "AIS_Gap_Detected" in suspect_df.columns:
                display_columns.append("AIS_Gap_Detected")
                rename_map["AIS_Gap_Detected"] = "AIS Gap"

            display_columns.append("Within_Screening_Radius")

            if "Timestamp" in suspect_df.columns:
                display_columns.append("Timestamp")

            # Filter to columns that actually exist
            display_columns = [c for c in display_columns if c in suspect_df.columns]
            display_df = suspect_df[display_columns].rename(columns=rename_map)

            # Format timestamp for display
            if "Timestamp" in display_df.columns:
                display_df["Timestamp"] = display_df["Timestamp"].map(
                    lambda value: value.strftime("%Y-%m-%d %H:%M UTC")
                    if pd.notna(value)
                    else "Unavailable"
                )
            # Round suspicion score in table
            if "Suspicion Score" in display_df.columns:
                display_df["Suspicion Score"] = display_df["Suspicion Score"].map(
                    lambda v: f"{v:.1f}"
                )
            # Round SOG in table
            if "SOG (knots)" in display_df.columns:
                display_df["SOG (knots)"] = display_df["SOG (knots)"].map(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                )

            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption(
                "Vessels are ranked by multi-factor Suspicion Score, not distance alone. "
                "This is an evidence-based investigative priority ranking and does not "
                "establish vessel responsibility."
            )
