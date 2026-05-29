from __future__ import annotations

import json
import struct
from pathlib import Path

from app.operators.repair import apply_repair_policy
from app.workers.workflow_executor import _camera_quality_from_colmap


def _transform_at(x: float, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_sparse_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    rows = [struct.pack("<fffBBB", x, y, z, 128, 128, 128) for x, y, z in points]
    path.write_bytes(header + b"".join(rows))


def _ply_vertex_count(path: Path) -> int:
    for line in path.read_bytes().splitlines():
        if line.startswith(b"element vertex "):
            return int(line.split()[-1])
    raise AssertionError("PLY vertex count header missing")


def test_repair_policy_prunes_isolated_camera_jump_and_crops_sparse_bbox(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    frames = [
        {"file_path": "images/frame_000.jpg", "transform_matrix": _transform_at(0)},
        {"file_path": "images/frame_001.jpg", "transform_matrix": _transform_at(1)},
        {"file_path": "images/frame_002.jpg", "transform_matrix": _transform_at(2)},
        {"file_path": "images/frame_003_bad.jpg", "transform_matrix": _transform_at(50)},
        {"file_path": "images/frame_004.jpg", "transform_matrix": _transform_at(3)},
        {"file_path": "images/frame_005.jpg", "transform_matrix": _transform_at(4)},
    ]
    (dataset_dir / "transforms.json").write_text(
        json.dumps({"ply_file_path": "sparse_point_cloud.ply", "frames": frames}),
        encoding="utf-8",
    )
    _write_sparse_ply(dataset_dir / "sparse_point_cloud.ply", [(0, 0, 0), (1, 0, 0), (2, 0, 0), (100, 0, 0)])

    report = apply_repair_policy(
        dataset_dir,
        {
            "input_image_count": 6,
            "registered_camera_count": 6,
            "registration_rate": 1.0,
            "sparse_point_count": 4,
            "trajectory_continuity": {"passed": True, "median_step": 1.0, "max_step": 48.0},
        },
        {
            "repair_source_workflow_id": "workflow_source",
            "repair": {
                "enabled": True,
                "max_camera_position_jump_ratio": 6.0,
                "sparse_percentile_min": 0,
                "sparse_percentile_max": 75,
                "sparse_expand_ratio": 1.0,
            },
        },
    )

    repaired_transforms = json.loads((dataset_dir / "transforms.json").read_text(encoding="utf-8"))
    remaining_names = [Path(frame["file_path"]).name for frame in repaired_transforms["frames"]]
    assert "frame_003_bad.jpg" not in remaining_names
    assert report["quality_after"]["registered_camera_count"] == 5
    assert report["quality_after"]["registration_rate"] == 5 / 6
    assert report["quality_after"]["repair"]["camera_prune_applied"] is True
    assert report["quality_after"]["repair"]["sparse_bbox_crop_applied"] is True
    assert _ply_vertex_count(dataset_dir / "sparse_point_cloud.ply") == 3
    assert (dataset_dir / "sparse_point_cloud.ply.pre_repair").exists()


def test_camera_quality_gate_blocks_large_position_jump() -> None:
    quality = _camera_quality_from_colmap(
        {
            "input_image_count": 20,
            "registered_camera_count": 20,
            "registration_rate": 1.0,
            "mean_reprojection_error": 1.0,
            "trajectory_continuity": {"passed": True, "median_step": 1.0, "max_step": 12.5},
        },
        "standard",
        media_metadata={"input_mode": "video"},
    )

    assert quality["passed"] is False
    assert "camera_position_jump_too_large" in quality["issues"]
    assert quality["max_step_over_median"] == 12.5


def test_unordered_detail_photos_do_not_hard_fail_on_filename_adjacency_jump() -> None:
    quality = _camera_quality_from_colmap(
        {
            "input_image_count": 106,
            "registered_camera_count": 106,
            "registration_rate": 1.0,
            "mean_reprojection_error": 1.29,
            "sparse_point_count": 80436,
            "largest_component_ratio": 1.0,
            "trajectory_continuity": {"passed": True, "median_step": 0.652, "max_step": 8.128},
        },
        "standard",
        media_metadata={
            "input_mode": "images",
            "asset_type_summary": {"detail_photo": 106},
            "role_summary": {"detail_patch": 106},
            "source_files": [
                "9ebf3a3ec54158f1e3dc5ab1c356a622.jpg",
                "65fc8c5a256143ea6dd39e646ea9bf89.jpg",
                "IMG_20260519_163247.jpg",
            ],
        },
    )

    assert quality["passed"] is True
    assert quality["hard_fail"] is False
    assert quality["camera_quality_gate_mode"] == "unordered_graph_gate"
    assert quality["camera_adjacency_basis"] == "disabled_for_unordered_photos"
    assert "camera_position_jump_too_large" not in quality["issues"]
    assert "camera_position_jump_too_large" in quality["warnings"]


def test_video_camera_quality_still_hard_fails_on_sequential_jump() -> None:
    quality = _camera_quality_from_colmap(
        {
            "input_image_count": 120,
            "registered_camera_count": 118,
            "registration_rate": 118 / 120,
            "mean_reprojection_error": 1.8,
            "sparse_point_count": 12000,
            "trajectory_continuity": {"passed": True, "median_step": 0.5, "max_step": 7.0},
        },
        "standard",
        media_metadata={"input_mode": "video", "source_files": ["frame_000001.jpg", "frame_000002.jpg"]},
    )

    assert quality["passed"] is False
    assert quality["camera_quality_gate_mode"] == "sequential_trajectory_gate"
    assert quality["camera_adjacency_basis"] == "frame_index"
    assert "camera_position_jump_too_large" in quality["issues"]
