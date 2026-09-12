"""Reusable GeoTIFF detection processing for the Streamlit and HTTP frontends.

The image-processing operations in this module mirror the existing Tab 1
pipeline.  The returned payload is deliberately JSON-friendly so that the
same real detection result can be consumed by the React frontend.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.warp import transform as transform_coordinates

from detection.classifier import (
    ClassifierLoadError,
    EfficientNetOilClassifier,
    runtime_status,
)


CLASSIFIER_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "efficientnet_b0_oil_classifier.pth"
)


class DetectionPipelineError(RuntimeError):
    """Raised when a GeoTIFF cannot be processed by the existing pipeline."""


def _image_data_url(image: np.ndarray) -> str:
    """Encode a rendered processing view as a browser-readable PNG data URL."""

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise DetectionPipelineError("A processed imagery view could not be encoded.")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _normalise_band(band1: np.ndarray) -> tuple[np.ndarray, bool, float | None, float | None]:
    """Apply the existing 2nd–98th percentile normalization safely."""

    finite_band1 = band1[np.isfinite(band1)]
    if finite_band1.size == 0:
        return np.zeros_like(band1, dtype=np.uint8), False, None, None

    lower = float(np.percentile(finite_band1, 2))
    upper = float(np.percentile(finite_band1, 98))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return np.zeros_like(band1, dtype=np.uint8), False, lower, upper

    band_clipped = np.clip(
        np.nan_to_num(band1, nan=lower, posinf=upper, neginf=lower),
        lower,
        upper,
    )
    band1_norm = np.uint8(
        np.round((band_clipped - lower) / (upper - lower) * 255)
    )
    return band1_norm, True, lower, upper


def _candidate_features(
    contour: np.ndarray,
    cleaned_mask: np.ndarray,
    filtered_image: np.ndarray,
    morphology_kernel: np.ndarray,
    image_median: float,
    width: int,
    height: int,
) -> dict:
    """Calculate the same transparent geometric and intensity features as Tab 1."""

    area = float(cv2.contourArea(contour))
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
    centroid = (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    ) if moments["m00"] != 0 else None

    candidate_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    cv2.drawContours(candidate_mask, [contour], -1, 255, -1)
    mean_intensity = float(cv2.mean(filtered_image, mask=candidate_mask)[0])

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
        or bbox_x + bbox_w >= width - 1
        or bbox_y + bbox_h >= height - 1
    )

    return {
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


def _serialise_candidate(candidate: dict, rank: int, selected: bool) -> dict:
    """Remove OpenCV/numpy objects before returning a JSON response."""

    return {
        "id": f"C{rank}",
        "rank": rank,
        "score": float(candidate["score"]),
        "area": float(candidate["area"]),
        "aspectRatio": (
            float(candidate["aspect_ratio"])
            if candidate["aspect_ratio"] is not None
            else None
        ),
        "meanIntensity": float(candidate["mean_intensity"]),
        "compactness": float(candidate["compactness"]),
        "borderTouching": bool(candidate["border_touching"]),
        "selected": selected,
    }


def _wgs84_coordinates(dataset, x: float, y: float) -> tuple[float, float]:
    """Return longitude/latitude while preserving native raster coordinates separately."""

    if dataset.crs is None or dataset.crs.is_geographic:
        return float(x), float(y)

    longitude, latitude = transform_coordinates(
        dataset.crs,
        "EPSG:4326",
        [x],
        [y],
    )
    return float(longitude[0]), float(latitude[0])


def _engine_metadata(classifier_available: bool, device: str, classifier_error: str | None) -> dict:
    if classifier_available:
        return {
            "name": "EfficientNetB0 + classical ranking",
            "status": "trained weights loaded",
            "note": "Candidate selection uses the loaded oil-like classifier; classical features remain available.",
            "hardware": device,
            "precision": "FP32",
            "objective": "Oil-like vs look-alike candidate classification",
        }

    note = (
        "Classifier architecture available, but trained oil-spill weights are not loaded. "
        "Classical candidate ranking is used."
    )
    if classifier_error:
        note += f" {classifier_error}"
    return {
        "name": "Classical candidate ranking",
        "status": "active",
        "note": note,
        "hardware": device,
        "precision": "UINT8",
        "objective": "Dark formation segmentation",
    }


def process_geotiff(path: str | Path) -> dict:
    """Run the real Tab 1 pipeline and return a JSON-serialisable result.

    The returned imagery is derived from the uploaded raster and the actual
    segmentation output.  No detection values are fabricated.
    """

    try:
        dataset_context = rasterio.open(path)
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise DetectionPipelineError(f"The uploaded file is not a readable GeoTIFF: {exc}") from exc

    with dataset_context as dataset:
        try:
            band1 = dataset.read(1)
        except (rasterio.errors.RasterioError, ValueError) as exc:
            raise DetectionPipelineError(f"Band 1 could not be read: {exc}") from exc

        band1_norm, normalization_available, lower, upper = _normalise_band(band1)
        filtered_image = cv2.medianBlur(band1_norm, 5)
        morphology_kernel = np.ones((5, 5), np.uint8)

        if normalization_available:
            otsu_threshold, otsu_mask = cv2.threshold(
                filtered_image,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
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

        min_area = max(
            20,
            int(dataset.height * dataset.width * 0.0005),
        )
        image_median = float(np.median(filtered_image))
        candidates = []
        for contour in contours:
            try:
                if cv2.contourArea(contour) < min_area:
                    continue
                candidates.append(
                    _candidate_features(
                        contour,
                        cleaned_mask,
                        filtered_image,
                        morphology_kernel,
                        image_median,
                        dataset.width,
                        dataset.height,
                    )
                )
            except (cv2.error, ValueError, ZeroDivisionError):
                continue

        classifier_error = None
        classifier = None
        classifier_runtime = runtime_status()
        if candidates and CLASSIFIER_WEIGHTS_PATH.is_file() and classifier_runtime["available"]:
            try:
                classifier = EfficientNetOilClassifier.from_weights(CLASSIFIER_WEIGHTS_PATH)
            except ClassifierLoadError as exc:
                classifier_error = str(exc)

        if candidates:
            max_candidate_area = max(candidate["area"] for candidate in candidates)
            for candidate in candidates:
                area_score = (
                    float(np.sqrt(candidate["area"] / max_candidate_area))
                    if max_candidate_area > 0
                    else 0.0
                )
                candidate["area_score"] = area_score
                candidate["darkness_score"] = candidate["relative_darkness"]
                candidate["shape_score"] = float(
                    np.clip(candidate["compactness"], 0.0, 1.0)
                )
                candidate["border_penalty"] = (
                    1.0 if candidate["border_touching"] else 0.0
                )
                candidate["score"] = float(
                    np.clip(
                        0.40 * candidate["area_score"]
                        + 0.35 * candidate["darkness_score"]
                        + 0.25 * candidate["shape_score"]
                        - 0.20 * candidate["border_penalty"],
                        0.0,
                        1.0,
                    )
                )
                candidate["classical_score"] = candidate["score"]

            if classifier is not None:
                for candidate in candidates:
                    padding = 8
                    crop_x0 = max(0, candidate["bbox_x"] - padding)
                    crop_y0 = max(0, candidate["bbox_y"] - padding)
                    crop_x1 = min(dataset.width, candidate["bbox_x"] + candidate["bbox_w"] + padding)
                    crop_y1 = min(dataset.height, candidate["bbox_y"] + candidate["bbox_h"] + padding)
                    try:
                        candidate.update(
                            classifier.predict(band1_norm[crop_y0:crop_y1, crop_x0:crop_x1])
                        )
                    except Exception as exc:  # classifier failure should not discard classical detection
                        candidate["classification_error"] = str(exc)

        ai_classification_available = bool(
            classifier is not None
            and candidates
            and all("oil_probability" in candidate for candidate in candidates)
        )
        ranking_key = "oil_probability" if ai_classification_available else "classical_score"
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: candidate[ranking_key],
            reverse=True,
        )
        selected_candidate = ranked_candidates[0] if ranked_candidates else None

        overlay = cv2.cvtColor(band1_norm, cv2.COLOR_GRAY2RGB)
        if selected_candidate is not None:
            for candidate in ranked_candidates:
                if candidate is not selected_candidate:
                    cv2.drawContours(overlay, [candidate["contour"]], -1, (255, 200, 0), 1)

            largest_spill = selected_candidate["contour"]
            cv2.drawContours(overlay, [largest_spill], -1, (0, 255, 0), 2)
            bbox_x = selected_candidate["bbox_x"]
            bbox_y = selected_candidate["bbox_y"]
            bbox_w = selected_candidate["bbox_w"]
            bbox_h = selected_candidate["bbox_h"]
            rect = cv2.minAreaRect(largest_spill)
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
                "Selected",
                (bbox_x, max(15, bbox_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if selected_candidate["centroid"] is not None:
                px, py = selected_candidate["centroid"]
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

        engine = _engine_metadata(
            ai_classification_available,
            classifier_runtime["device"],
            classifier_error,
        )
        response = {
            "status": "success" if selected_candidate is not None else "no_candidate",
            "engine": engine,
            "imagery": {
                "backscatter": {
                    "src": _image_data_url(band1_norm),
                    "title": "Calibrated Backscatter (Band 1)",
                    "technicalLabel": "Normalized 8-bit",
                    "caption": "Band 1 after percentile clipping and normalization.",
                },
                "adaptiveMask": {
                    "src": _image_data_url(cleaned_mask),
                    "title": "Adaptive Mask (Otsu + morphology)",
                    "technicalLabel": (
                        f"Threshold {otsu_threshold:.2f}/255"
                        if otsu_threshold is not None
                        else "Unavailable"
                    ),
                    "caption": "Dark-region candidates after median filtering and conservative cleanup.",
                },
                "polygonOverlay": {
                    "src": _image_data_url(overlay),
                    "title": "Delineated Candidate Overlay",
                    "technicalLabel": "Selected candidate highlighted",
                    "highlighted": selected_candidate is not None,
                    "caption": "Retained contours are shown in amber; the selected candidate is green with geometry overlays.",
                },
            },
            "raster": {
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bounds": str(dataset.bounds),
            },
            "metadata": {
                "otsuThreshold": float(otsu_threshold) if otsu_threshold is not None else None,
                "minComponentArea": int(min_area),
                "candidateCount": len(ranked_candidates),
                "normalizationLower": lower,
                "normalizationUpper": upper,
            },
            "candidates": [
                _serialise_candidate(candidate, rank, candidate is selected_candidate)
                for rank, candidate in enumerate(ranked_candidates, start=1)
            ],
        }

        if selected_candidate is None:
            response["message"] = (
                "No valid spill candidates were found after filtering. "
                "Try another SAR image or adjust the detection parameters."
            )
            return response

        largest_spill = selected_candidate["contour"]
        pixel_area = float(selected_candidate["area"])
        pixel_perimeter = float(cv2.arcLength(largest_spill, True))
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest_spill)
        rect_center, (rect_w, rect_h), rect_angle = cv2.minAreaRect(largest_spill)
        length_pixels = float(max(rect_w, rect_h))
        width_pixels = float(min(rect_w, rect_h))
        aspect_ratio = length_pixels / width_pixels if width_pixels > 0 else None

        orientation = float(rect_angle)
        if rect_w < rect_h:
            orientation += 90.0
        if orientation >= 90.0:
            orientation -= 180.0
        elif orientation < -90.0:
            orientation += 180.0

        moments = cv2.moments(largest_spill)
        centroid_available = moments["m00"] != 0
        px = py = None
        real_x = real_y = None
        longitude = latitude = None
        if centroid_available:
            px = int(moments["m10"] / moments["m00"])
            py = int(moments["m01"] / moments["m00"])
            real_x, real_y = dataset.xy(py, px)
            longitude, latitude = _wgs84_coordinates(dataset, real_x, real_y)

        pixel_width, pixel_height = (abs(value) for value in dataset.res)
        resolution_available = all(
            np.isfinite(value) and value > 0
            for value in (pixel_width, pixel_height)
        )
        crs = dataset.crs
        crs_text = str(crs) if crs else "CRS unavailable"
        is_projected = bool(crs and crs.is_projected)
        linear_units = getattr(crs, "linear_units", None) if crs else None
        can_convert_to_map_units = resolution_available and is_projected
        map_unit = linear_units or "map units"
        pixel_scale = (pixel_width + pixel_height) / 2
        area_map_units = pixel_area * pixel_width * pixel_height
        perimeter_map_units = pixel_perimeter * pixel_scale
        length_map_units = length_pixels * pixel_scale
        width_map_units = width_pixels * pixel_scale

        response["confidence"] = float(selected_candidate["score"])
        response["geometry"] = {
            "area": area_map_units if can_convert_to_map_units else pixel_area,
            "perimeter": perimeter_map_units if can_convert_to_map_units else pixel_perimeter,
            "length": length_map_units if can_convert_to_map_units else length_pixels,
            "width": width_map_units if can_convert_to_map_units else width_pixels,
            "areaUnit": (
                "m²"
                if linear_units in ("metre", "meter", "m")
                else f"{map_unit}²"
                if can_convert_to_map_units
                else "px²"
            ),
            "distanceUnit": (
                "m"
                if linear_units in ("metre", "meter", "m")
                else map_unit
                if can_convert_to_map_units
                else "px"
            ),
            "orientation": orientation,
            "aspectRatio": aspect_ratio,
        }
        response["spatial"] = {
            "longitude": longitude,
            "latitude": latitude,
            "x": float(real_x) if real_x is not None else None,
            "y": float(real_y) if real_y is not None else None,
            "pixelX": px,
            "pixelY": py,
            "crs": crs_text,
            "epsg": int(crs.to_epsg()) if crs and crs.to_epsg() is not None else None,
        }
        response["raster"].update(
            {
                "resolution": pixel_scale if resolution_available else None,
                "groundSamplingDistance": (
                    f"{pixel_width:g} × {pixel_height:g} {map_unit}/px"
                    if resolution_available
                    else None
                ),
            }
        )
        response["morphologyLabel"] = (
            "Elongated formation" if aspect_ratio is not None and aspect_ratio > 1.5 else "Compact formation"
        )
        response["interpretation"] = (
            "Dark-region candidate selected by transparent classical image-processing features. "
            "This result is not an oil probability and does not resolve SAR look-alikes."
        )
        return response
