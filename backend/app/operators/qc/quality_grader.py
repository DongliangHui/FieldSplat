from __future__ import annotations

from typing import Any


HARD_D_KEYS = {
    "input_camera_count_match",
    "unique_image_names_match",
    "pano_group_constraints_valid",
}


def grade_quality(checks: dict[str, Any]) -> dict[str, Any]:
    if checks.get("duplicated_img_names") or checks.get("missing_crop_ids"):
        return {"quality_grade": "D", "measurement_allowed": False, "hard_fail": True, "hard_fail_reason": "camera_mapping_error"}

    for key in HARD_D_KEYS:
        if checks.get(key) is False:
            return {"quality_grade": "D", "measurement_allowed": False, "hard_fail": True, "hard_fail_reason": "camera_mapping_error"}

    if checks.get("pointcloud_fragmentation") == "high" and checks.get("trajectory_reasonable") is False:
        return {"quality_grade": "D", "measurement_allowed": False, "hard_fail": True, "hard_fail_reason": "geometry_fragmentation"}

    if checks.get("pointcloud_fragmentation") == "high":
        return {"quality_grade": "C", "measurement_allowed": False, "hard_fail": False, "hard_fail_reason": None}

    return {"quality_grade": "B", "measurement_allowed": False, "hard_fail": False, "hard_fail_reason": None}
