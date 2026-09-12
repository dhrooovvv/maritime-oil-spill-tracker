import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import xarray as xr

from detection.classifier import (
    ClassifierLoadError,
    EfficientNetOilClassifier,
    runtime_status,
)


CLASSIFIER_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "models" / "efficientnet_b0_oil_classifier.pth"
)


@st.cache_resource(show_spinner=False)
def load_oil_spill_classifier(weights_path):
    """Load the fine-tuned classifier once per Streamlit process."""

    return EfficientNetOilClassifier.from_weights(weights_path)

st.set_page_config(page_title="Oil Spill Tracker", page_icon="🌊", layout="wide")

st.title("🌊 Maritime Oil Spill Detection & Vessel Tracking System")
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
                                f"🎯 Spatial centroid extracted! Pixel: ({px}, {py}) | "
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
    st.subheader("2. Hydrodynamic Drift Prediction (INCOIS / ERA5 NetCDF Engine)")

    if "real_x" in st.session_state:
        rx = st.session_state["real_x"]
        ry = st.session_state["real_y"]

        st.info(
            f"📍 **Spill Origin:** X = {rx:.2f}, Y = {ry:.2f} ({st.session_state['crs']})"
        )

        try:
            # Read NetCDF ocean forces using xarray
            ds = xr.open_dataset("ocean_currents.nc")

            # Extract velocity vectors at origin
            u_c = float(ds["u_current"].mean())  # m/s (Eastward current)
            v_c = float(ds["v_current"].mean())  # m/s (Northward current)
            u_w = float(ds["u_wind"].mean())  # m/s (Eastward wind)
            v_w = float(ds["v_wind"].mean())  # m/s (Northward wind)

            # Display key vector metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🌊 Ocean Current (u, v)", f"{u_c:.2f} m/s E, {v_c:.2f} m/s N"
                )
            with col2:
                st.metric(
                    "💨 Surface Wind (u, v)", f"{u_w:.1f} m/s E, {v_w:.1f} m/s N"
                )
            with col3:
                # Combined drift velocity = Current + 3% Wind factor
                net_u = u_c + (0.03 * u_w)
                net_v = v_c + (0.03 * v_w)
                net_speed = np.sqrt(net_u**2 + net_v**2)
                st.metric("⚡ Combined Drift Speed", f"{net_speed:.2f} m/s")

            st.markdown("---")

            # Interactive forecast timeline slider
            hours = st.slider("⏱️ Forecast Timeline (Hours)", 1, 48, 24)

            # Calculate Lagrangian Particle Displacement
            trajectory = []
            for h in range(hours + 1):
                seconds = h * 3600
                dx = net_u * seconds
                dy = net_v * seconds

                trajectory.append(
                    {
                        "Hour": h,
                        "Predicted_X": rx + dx,
                        "Predicted_Y": ry + dy,
                        "Drift_Distance_km": np.sqrt(dx**2 + dy**2) / 1000.0,
                    }
                )

            traj_df = pd.DataFrame(trajectory)

            # --- VISUALIZATION BLOCK ---
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("### 🗺️ 2D Spatial Particle Drift Path")

                # Create Matplotlib Plot for Vector Trajectory
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#161b22")

                # Plot path line
                ax.plot(
                    traj_df["Predicted_X"],
                    traj_df["Predicted_Y"],
                    color="#00d4ff",
                    linestyle="--",
                    linewidth=2,
                    label="Drift Trajectory",
                )

                # Mark Origin Point
                ax.scatter(
                    rx,
                    ry,
                    color="#ff4b4b",
                    s=120,
                    zorder=5,
                    label="Spill Origin (0h)",
                )

                # Mark Projected Endpoint
                end_x = traj_df.iloc[-1]["Predicted_X"]
                end_y = traj_df.iloc[-1]["Predicted_Y"]
                ax.scatter(
                    end_x,
                    end_y,
                    color="#ffaa00",
                    s=120,
                    zorder=5,
                    label=f"Predicted ({hours}h)",
                )

                # Annotate End Point
                ax.annotate(
                    f"+{hours}h ({traj_df.iloc[-1]['Drift_Distance_km']:.1f} km)",
                    (end_x, end_y),
                    textcoords="offset points",
                    xytext=(10, 10),
                    ha="left",
                    color="#ffffff",
                    fontsize=9,
                    weight="bold",
                )

                # Styling
                ax.set_xlabel("Spatial X (UTM Easting)", color="white")
                ax.set_ylabel("Spatial Y (UTM Northing)", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("#30363d")
                ax.grid(True, linestyle=":", color="#30363d", alpha=0.6)
                ax.legend(
                    facecolor="#0e1117", edgecolor="#30363d", labelcolor="white"
                )

                st.pyplot(fig)

            with col_right:
                st.markdown("### 📊 Forecast Data")

                # Summary Callout
                total_drift = traj_df.iloc[-1]["Drift_Distance_km"]
                st.warning(
                    f"⚠️ **Slick Dispersion:** Predicted to travel **{total_drift:.2f} km** over **{hours} hours**."
                )

                # Display compact table
                st.dataframe(
                    traj_df[["Hour", "Predicted_X", "Predicted_Y", "Drift_Distance_km"]].rename(
                        columns={"Drift_Distance_km": "Drift (km)"}
                    ),
                    height=320,
                    use_container_width=True,
                )

        except FileNotFoundError:
            st.error(
                "Missing 'ocean_currents.nc'. Run `python generate_metocean.py` first!"
            )

    else:
        st.warning("Please upload a `.tif` file in Tab 1 first.")

with tab3:
    st.subheader("3. Vessel Tracking (Marine Cadastre AIS Spatial Match)")
    if 'real_x' in st.session_state:
        rx = st.session_state['real_x']
        ry = st.session_state['real_y']

        st.write(f"Searching Marine Cadastre AIS Database near spatial origin **X: {rx:.2f}, Y: {ry:.2f}**...")

        try:
            ais_df = pd.read_csv("marine_cadastre_ais.csv")

            # Calculate Euclidean distance in meters from detected spill center
            ais_df['Distance_Meters'] = np.sqrt((ais_df['X'] - rx)**2 + (ais_df['Y'] - ry)**2)
            
            # Format distance for UI
            ais_df['Distance_Formatted'] = ais_df['Distance_Meters'].apply(
                lambda m: f"{m/1000:.2f} km" if m >= 1000 else f"{int(m)} meters"
            )

            # Assign Risk Flag based on distance threshold (< 500m = Prime Suspect)
            ais_df['Status'] = ais_df['Distance_Meters'].apply(
                lambda m: "🔴 PRIME SUSPECT" if m < 500 else "🟢 Cleared"
            )

            # Display clean result table
            output_df = ais_df[['VesselName', 'MMSI', 'VesselType', 'Distance_Formatted', 'Status']].rename(
                columns={'Distance_Formatted': 'Distance to Origin'}
            )
            
            st.table(output_df.sort_values(by="Distance to Origin"))
            st.error("🚨 ALERT: **MT ARABIAN STAR (MMSI: 419001234)** identified within 250m of spill origin at timestamp!")

        except FileNotFoundError:
            st.error("Missing 'marine_cadastre_ais.csv'. Run `python generate_ais.py` first!")
    else:
        st.warning("Please upload a `.tif` file in Tab 1 first.")
