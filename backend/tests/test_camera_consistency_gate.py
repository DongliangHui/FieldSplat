from __future__ import annotations

from app.operators.qc import quality_report_from_camera_check, validate_camera_mapping


def test_camera_consistency_blocks_duplicate_img_names() -> None:
    expected = [f"crop_{idx:02d}.jpg" for idx in range(16)]
    cameras = [{"img_name": f"crop_{idx % 4:02d}.jpg"} for idx in range(16)]

    result = validate_camera_mapping(expected, cameras)
    report = quality_report_from_camera_check("workflow_test", result)

    assert result["passed"] is False
    assert result["expected_views"] == 16
    assert result["actual_cameras"] == 16
    assert result["unique_img_names"] == 4
    assert result["duplicated_img_names"] == ["crop_00.jpg", "crop_01.jpg", "crop_02.jpg", "crop_03.jpg"]
    assert report["quality_grade"] == "D"
    assert report["measurement_allowed"] is False
    assert report["hard_fail"] is True
    assert report["hard_fail_reason"] == "camera_mapping_error"


def test_camera_consistency_enforces_pano_shared_center_group() -> None:
    expected = ["pano_001_yaw_000.jpg", "pano_001_yaw_060.jpg"]
    cameras = [
        {"img_name": "pano_001_yaw_000.jpg", "camera_center": [0.0, 0.0, 0.0]},
        {"img_name": "pano_001_yaw_060.jpg", "camera_center": [1.0, 0.0, 0.0]},
    ]
    crop_manifest = {
        "crops": [
            {"crop_id": "pano_001_yaw_000.jpg", "source_pano_id": "pano_001", "shared_center_group": "pano_001"},
            {"crop_id": "pano_001_yaw_060.jpg", "source_pano_id": "pano_001", "shared_center_group": "pano_001"},
        ]
    }

    result = validate_camera_mapping(expected, cameras, crop_manifest=crop_manifest, pano_center_tolerance=0.25)

    assert result["passed"] is False
    assert result["pano_group_constraints_valid"] is False
    assert result["hard_fail_reason"] == "camera_mapping_error"


def test_camera_consistency_passes_one_to_one_mapping() -> None:
    expected = ["frame_001.jpg", "frame_002.jpg"]
    cameras = [{"img_name": "frame_001.jpg"}, {"img_name": "frame_002.jpg"}]

    result = validate_camera_mapping(expected, cameras)

    assert result["passed"] is True
    assert result["hard_fail"] is False
