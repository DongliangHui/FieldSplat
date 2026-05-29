from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from pathlib import PurePosixPath
from typing import Any


def _camera_records(cameras_json: Any) -> list[dict[str, Any]]:
    if isinstance(cameras_json, list):
        return [item for item in cameras_json if isinstance(item, dict)]
    if isinstance(cameras_json, dict):
        for key in ("cameras", "frames", "images"):
            value = cameras_json.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _image_name(record: dict[str, Any]) -> str | None:
    for key in ("img_name", "image_name", "file_name", "filename", "name"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return PurePosixPath(value.replace("\\", "/")).name
    return None


def _center(record: dict[str, Any]) -> tuple[float, float, float] | None:
    value = record.get("camera_center") or record.get("center") or record.get("position")
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"]), float(value["z"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    return None


def _canonical_expected_name(item: str | dict[str, Any]) -> str:
    if isinstance(item, dict):
        value = item.get("crop_id") or item.get("image_name") or item.get("img_name") or item.get("path")
    else:
        value = item
    value = str(value)
    if "/" in value or "\\" in value:
        value = PurePosixPath(value.replace("\\", "/")).name
    return value


def _crop_records(crop_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not crop_manifest:
        return []
    crops = crop_manifest.get("crops")
    if isinstance(crops, list):
        return [crop for crop in crops if isinstance(crop, dict)]
    images = crop_manifest.get("images")
    if isinstance(images, list):
        return [image for image in images if isinstance(image, dict)]
    return []


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def validate_camera_mapping(
    expected_images: list[str | dict[str, Any]],
    cameras_json: Any,
    crop_manifest: dict[str, Any] | None = None,
    pano_center_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Validate image-camera mapping before InstantSplat++ training.

    This is the hard gate for the documented failure where 16 camera records
    were emitted with only 4 unique img_name values.
    """

    expected_names = [_canonical_expected_name(item) for item in expected_images]
    camera_records = _camera_records(cameras_json)
    img_names = [_image_name(record) for record in camera_records]
    img_names = [name for name in img_names if name]

    expected_counter = Counter(expected_names)
    actual_counter = Counter(img_names)
    expected_set = set(expected_names)
    actual_set = set(img_names)

    missing_crop_ids = sorted(expected_set - actual_set)
    unexpected_img_names = sorted(actual_set - expected_set)
    duplicated_img_names = sorted(name for name, count in actual_counter.items() if count > 1)
    duplicated_expected_names = sorted(name for name, count in expected_counter.items() if count > 1)

    crop_records = _crop_records(crop_manifest)
    crops_missing_shared_center = [
        _canonical_expected_name(crop)
        for crop in crop_records
        if crop.get("source_pano_id") and not crop.get("shared_center_group")
    ]

    pano_group_constraints_valid = len(crops_missing_shared_center) == 0
    centers_by_name = {_image_name(record): _center(record) for record in camera_records}
    centers_by_group: dict[str, list[tuple[str, tuple[float, float, float]]]] = defaultdict(list)
    for crop in crop_records:
        group = crop.get("shared_center_group")
        if not group:
            continue
        name = _canonical_expected_name(crop)
        center = centers_by_name.get(name)
        if center is not None:
            centers_by_group[str(group)].append((name, center))

    pano_group_center_spread: dict[str, float] = {}
    for group, group_centers in centers_by_group.items():
        if len(group_centers) < 2:
            continue
        anchor = group_centers[0][1]
        spread = max(_distance(anchor, center) for _, center in group_centers[1:])
        pano_group_center_spread[group] = spread
        if spread > pano_center_tolerance:
            pano_group_constraints_valid = False

    expected_views = len(expected_names)
    actual_cameras = len(camera_records)
    unique_img_names = len(set(img_names))

    passed = (
        expected_views == actual_cameras
        and expected_views == unique_img_names
        and not missing_crop_ids
        and not duplicated_img_names
        and not duplicated_expected_names
        and pano_group_constraints_valid
    )

    hard_fail_reason = None if passed else "camera_mapping_error"
    return {
        "passed": passed,
        "expected_views": expected_views,
        "actual_cameras": actual_cameras,
        "unique_img_names": unique_img_names,
        "missing_crop_ids": missing_crop_ids,
        "unexpected_img_names": unexpected_img_names,
        "duplicated_img_names": duplicated_img_names,
        "duplicated_expected_names": duplicated_expected_names,
        "pano_group_constraints_valid": pano_group_constraints_valid,
        "crops_missing_shared_center": sorted(crops_missing_shared_center),
        "pano_group_center_spread": pano_group_center_spread,
        "hard_fail": not passed,
        "hard_fail_reason": hard_fail_reason,
    }


def quality_report_from_camera_check(workflow_id: str, check: dict[str, Any]) -> dict[str, Any]:
    passed = bool(check.get("passed"))
    return {
        "run_id": workflow_id,
        "workflow_id": workflow_id,
        "checks": {
            "input_camera_count_match": check.get("expected_views") == check.get("actual_cameras"),
            "unique_image_names_match": check.get("expected_views") == check.get("unique_img_names"),
            "missing_crop_ids": check.get("missing_crop_ids", []),
            "duplicated_img_names": check.get("duplicated_img_names", []),
            "pano_group_constraints_valid": check.get("pano_group_constraints_valid", True),
            "trajectory_reasonable": None,
            "pointcloud_fragmentation": None,
            "holdout_render_score": None,
            "reprojection_error_mean": None,
        },
        "hard_fail": not passed,
        "hard_fail_reason": None if passed else "camera_mapping_error",
        "quality_grade": "B" if passed else "D",
        "measurement_allowed": False,
        "notes": [] if passed else ["Camera-image mapping failed before training; train.py must not run."],
        "raw_camera_mapping_check": check,
    }
