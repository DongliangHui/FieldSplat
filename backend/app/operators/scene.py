from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fieldsplat_defaults import default_at
from app.models import Workflow


class ScenePartitionOperator:
    name = "scene.partition"
    queue = "cpu"

    def run(self, workflow: Workflow, pose_quality: dict[str, Any], *, input_image_count: int) -> dict[str, Any]:
        workspace_dir = Path(get_settings().workspace_root) / "runs" / workflow.id / "scene_partition"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        enable_if = default_at("scene_partition.enable_if", {})
        enable_if = enable_if if isinstance(enable_if, dict) else {}
        cell_policy = default_at("scene_partition.cell", {})
        cell_policy = cell_policy if isinstance(cell_policy, dict) else {}
        sparse_points = int(pose_quality.get("sparse_point_count") or 0)
        registered_cameras = int(pose_quality.get("registered_camera_count") or 0)
        connected_components = int(pose_quality.get("connected_component_count") or pose_quality.get("component_count") or 1)
        image_threshold = int(enable_if.get("image_count_gt", 800))
        camera_threshold = int(enable_if.get("registered_camera_gt", 600))
        sparse_threshold = int(enable_if.get("sparse_points_gt", 1_500_000))
        component_threshold = int(enable_if.get("connected_components_gt", 1))
        max_cameras_per_cell = int(cell_policy.get("max_cameras_per_cell", 450))
        target_cameras_per_cell = int(cell_policy.get("target_cameras_per_cell", 250))
        should_partition = (
            input_image_count > image_threshold
            or registered_cameras > camera_threshold
            or sparse_points > sparse_threshold
            or connected_components > component_threshold
            or (pose_quality.get("trajectory_continuity") or {}).get("passed") is False
        )
        divisor = max(1, target_cameras_per_cell)
        cell_count = max(1, min(64, (max(input_image_count, registered_cameras) // divisor) + 1)) if should_partition else 1
        manifest = {
            "workflow_id": workflow.id,
            "operator": self.name,
            "partitioned": should_partition,
            "reason": _partition_reason(input_image_count, sparse_points, pose_quality) if should_partition else "single_cell_scene",
            "input_image_count": input_image_count,
            "registered_camera_count": registered_cameras,
            "sparse_point_count": sparse_points,
            "cells": [
                {
                    "cell_id": f"cell_{index:03d}",
                    "training_strategy": "chunked_splatfacto" if should_partition else "splatfacto",
                    "target_cameras_per_cell": target_cameras_per_cell,
                    "max_cameras_per_cell": max_cameras_per_cell,
                }
                for index in range(cell_count)
            ],
            "lod": {"enabled": should_partition, "levels": [0, 1, 2] if should_partition else [0]},
            "policy": {"forbid_single_global_ply_for_large_scene": True},
        }
        (workspace_dir / "scene_partition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def _partition_reason(input_image_count: int, sparse_points: int, pose_quality: dict[str, Any]) -> str:
    enable_if = default_at("scene_partition.enable_if", {})
    enable_if = enable_if if isinstance(enable_if, dict) else {}
    if input_image_count > int(enable_if.get("image_count_gt", 800)):
        return "input_images_exceed_single_scene_threshold"
    if int(pose_quality.get("registered_camera_count") or 0) > int(enable_if.get("registered_camera_gt", 600)):
        return "registered_cameras_exceed_single_scene_threshold"
    if sparse_points > int(enable_if.get("sparse_points_gt", 1_500_000)):
        return "sparse_points_exceed_single_scene_threshold"
    if int(pose_quality.get("connected_component_count") or pose_quality.get("component_count") or 1) > int(enable_if.get("connected_components_gt", 1)):
        return "connected_components_require_partition"
    if (pose_quality.get("trajectory_continuity") or {}).get("passed") is False:
        return "connected_areas_require_partition"
    return "partition_policy_triggered"
