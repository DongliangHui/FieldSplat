from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_at


PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def apply_repair_policy(
    dataset_dir: Path,
    colmap_quality: dict[str, Any],
    config: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    repair_config = config.get("repair") or {}
    enabled = bool(repair_config.get("enabled") or config.get("repair_source_workflow_id") or config.get("repair_from_workflow_id"))
    workspace_dir = dataset_dir.parent / "repair"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace_dir / "repair_manifest.json"
    manifest: dict[str, Any] = {
        "enabled": enabled,
        "source_workflow_id": repair_config.get("source_workflow_id") or config.get("repair_source_workflow_id") or config.get("repair_from_workflow_id"),
        "actions": [],
        "quality_before": colmap_quality,
    }
    if not enabled:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**manifest, "manifest_path": str(manifest_path)}

    transforms_path = dataset_dir / "transforms.json"
    transforms = _read_json(transforms_path)
    frames = list(transforms.get("frames") or []) if isinstance(transforms, dict) else []
    camera_report = _prune_camera_frames(frames, repair_config, settings)
    if camera_report["applied"]:
        transforms["frames"] = [frame for index, frame in enumerate(frames) if index not in set(camera_report["removed_indices"])]
        transforms.setdefault("repair", {})["camera_prune"] = camera_report
        transforms_path.write_text(json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8")
        frames = list(transforms.get("frames") or [])
    manifest["actions"].append({"name": "prune_bad_cameras", **camera_report})

    sparse_path = dataset_dir / str(transforms.get("ply_file_path") or "sparse_point_cloud.ply")
    crop_report = _crop_sparse_point_cloud(sparse_path, repair_config)
    manifest["actions"].append({"name": "crop_sparse_bbox_p1_p99", **crop_report})

    repaired_quality = dict(colmap_quality)
    input_count = int(colmap_quality.get("input_image_count") or len(frames) or 0)
    if frames:
        repaired_quality["registered_camera_count"] = len(frames)
        repaired_quality["registration_rate"] = len(frames) / max(input_count, 1)
        repaired_quality["trajectory_continuity"] = _trajectory_continuity_from_frames(frames)
    if crop_report.get("applied"):
        repaired_quality["sparse_point_count"] = int(crop_report["kept_vertices"])
    repaired_quality["repair"] = {
        "enabled": True,
        "camera_prune_applied": camera_report["applied"],
        "sparse_bbox_crop_applied": crop_report.get("applied", False),
        "manifest_path": str(manifest_path),
    }

    camera_trajectory_path = workspace_dir / "camera_trajectory_repaired.json"
    _write_camera_trajectory(camera_trajectory_path, frames)
    registration_report_path = workspace_dir / "registration_report_repaired.json"
    registration_report_path.write_text(json.dumps(repaired_quality, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest.update(
        {
            "quality_after": repaired_quality,
            "transforms_path": str(transforms_path),
            "sparse_point_cloud_path": str(sparse_path) if sparse_path.exists() else None,
            "camera_trajectory_path": str(camera_trajectory_path),
            "registration_report_path": str(registration_report_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _prune_camera_frames(frames: list[dict[str, Any]], repair_config: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if not frames:
        return {"applied": False, "reason": "no_frames", "removed_indices": [], "removed_frames": []}
    camera_gate = default_at("camera_quality_gate", {}, settings=settings)
    camera_gate = camera_gate if isinstance(camera_gate, dict) else {}
    max_jump_ratio = float(repair_config.get("max_camera_position_jump_ratio") or camera_gate.get("max_camera_position_jump_ratio") or 6.0)
    max_removed_ratio = float(repair_config.get("max_removed_camera_ratio") or 0.25)
    bbox_min = repair_config.get("camera_bbox_percentile_min")
    bbox_max = repair_config.get("camera_bbox_percentile_max")
    bbox_expand = float(repair_config.get("camera_bbox_expand_ratio") or repair_config.get("expand_ratio") or 1.15)
    centers = [_frame_center(frame) for frame in frames]
    valid_indices = [index for index, center in enumerate(centers) if center is not None]
    removed: set[int] = set()

    if len(valid_indices) >= 3 and not repair_config.get("disable_adjacency_prune"):
        ordered_centers = [centers[index] for index in range(len(frames))]
        distances = [
            _distance(ordered_centers[index - 1], ordered_centers[index])
            for index in range(1, len(ordered_centers))
            if ordered_centers[index - 1] is not None and ordered_centers[index] is not None
        ]
        median_step = _percentile(distances, 50) or 0.0
        threshold = median_step * max_jump_ratio if median_step > 0 else math.inf
        for index in range(1, len(frames) - 1):
            prev_center = centers[index - 1]
            current_center = centers[index]
            next_center = centers[index + 1]
            if prev_center is None or current_center is None or next_center is None:
                continue
            prev_step = _distance(prev_center, current_center)
            next_step = _distance(current_center, next_center)
            bridge_step = _distance(prev_center, next_center)
            if prev_step > threshold and next_step > threshold and bridge_step < threshold:
                removed.add(index)

    if bbox_min is not None and bbox_max is not None and valid_indices:
        bounds = _expanded_percentile_bounds([centers[index] for index in valid_indices if centers[index] is not None], float(bbox_min), float(bbox_max), bbox_expand)
        for index in valid_indices:
            center = centers[index]
            if center is not None and _outside_bounds(center, bounds):
                removed.add(index)
    else:
        bounds = None

    removed_ratio = len(removed) / max(len(frames), 1)
    if removed_ratio > max_removed_ratio:
        return {
            "applied": False,
            "reason": "too_many_camera_removals",
            "removed_indices": sorted(removed),
            "removed_ratio": removed_ratio,
            "max_removed_ratio": max_removed_ratio,
            "max_camera_position_jump_ratio": max_jump_ratio,
            "bounds": bounds,
        }
    return {
        "applied": bool(removed),
        "reason": None if removed else "no_bad_cameras_detected",
        "original_frame_count": len(frames),
        "kept_frame_count": len(frames) - len(removed),
        "removed_indices": sorted(removed),
        "removed_frames": [frames[index].get("file_path") for index in sorted(removed)],
        "removed_ratio": removed_ratio,
        "max_removed_ratio": max_removed_ratio,
        "max_camera_position_jump_ratio": max_jump_ratio,
        "bounds": bounds,
    }


def _crop_sparse_point_cloud(path: Path, repair_config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"applied": False, "reason": "sparse_point_cloud_missing", "path": str(path)}
    percentile_min = float(_first_config_value(repair_config, ["sparse_percentile_min", "bbox_percentile_min"], 1))
    percentile_max = float(_first_config_value(repair_config, ["sparse_percentile_max", "bbox_percentile_max"], 99))
    expand_ratio = float(_first_config_value(repair_config, ["sparse_expand_ratio", "expand_ratio"], 1.15))
    parsed = _read_ply_rows(path)
    if not parsed.get("ok"):
        return {"applied": False, "reason": parsed["reason"], "path": str(path)}
    rows: list[bytes] = parsed["rows"]
    xyz: list[tuple[float, float, float]] = parsed["xyz"]
    bounds = _expanded_percentile_bounds([list(point) for point in xyz], percentile_min, percentile_max, expand_ratio)
    kept_rows = [row for row, point in zip(rows, xyz) if not _outside_bounds(list(point), bounds)]
    if len(kept_rows) == len(rows):
        return {
            "applied": False,
            "reason": "no_sparse_points_outside_bbox",
            "path": str(path),
            "original_vertices": len(rows),
            "kept_vertices": len(kept_rows),
            "bounds": bounds,
        }
    backup_path = path.with_suffix(path.suffix + ".pre_repair")
    if not backup_path.exists():
        path.replace(backup_path)
    else:
        path.unlink()
    _write_ply_rows(path, parsed["header_lines"], kept_rows)
    return {
        "applied": True,
        "path": str(path),
        "backup_path": str(backup_path),
        "original_vertices": len(rows),
        "kept_vertices": len(kept_rows),
        "removed_vertices": len(rows) - len(kept_rows),
        "removed_ratio": (len(rows) - len(kept_rows)) / max(len(rows), 1),
        "percentile_min": percentile_min,
        "percentile_max": percentile_max,
        "expand_ratio": expand_ratio,
        "bounds": bounds,
    }


def _read_ply_rows(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        return {"ok": False, "reason": "ply_header_missing"}
    header_end += len(b"end_header\n")
    header_text = data[:header_end].decode("ascii", errors="replace")
    header_lines = header_text.splitlines()
    if "format binary_little_endian 1.0" not in header_text:
        return {"ok": False, "reason": "unsupported_sparse_ply_format"}
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            if parts[1] not in PLY_TYPES:
                return {"ok": False, "reason": f"unsupported_sparse_ply_property:{parts[1]}"}
            properties.append((parts[1], parts[2]))
    offsets: dict[str, int] = {}
    row_size = 0
    for scalar_type, name in properties:
        offsets[name] = row_size
        row_size += PLY_TYPES[scalar_type][1]
    if not {"x", "y", "z"}.issubset(offsets):
        return {"ok": False, "reason": "sparse_ply_missing_xyz"}
    rows: list[bytes] = []
    xyz: list[tuple[float, float, float]] = []
    for index in range(vertex_count):
        start = header_end + index * row_size
        row = data[start : start + row_size]
        if len(row) != row_size:
            return {"ok": False, "reason": "sparse_ply_truncated"}
        rows.append(row)
        xyz.append(
            (
                struct.unpack_from("<f", row, offsets["x"])[0],
                struct.unpack_from("<f", row, offsets["y"])[0],
                struct.unpack_from("<f", row, offsets["z"])[0],
            )
        )
    return {"ok": True, "header_lines": header_lines, "rows": rows, "xyz": xyz}


def _write_ply_rows(path: Path, header_lines: list[str], rows: list[bytes]) -> None:
    next_lines = []
    for line in header_lines:
        if line.startswith("element vertex "):
            next_lines.append(f"element vertex {len(rows)}")
        else:
            next_lines.append(line)
    path.write_bytes(("\n".join(next_lines) + "\n").encode("ascii") + b"".join(rows))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_camera_trajectory(path: Path, frames: list[dict[str, Any]]) -> None:
    cameras = []
    for frame in frames:
        center = _frame_center(frame)
        if center is None:
            continue
        cameras.append({"image_name": Path(str(frame.get("file_path") or "")).name, "camera_center": center, "transform_matrix": frame.get("transform_matrix")})
    path.write_text(json.dumps({"source": "repair.prune_bad_cameras", "camera_count": len(cameras), "cameras": cameras}, ensure_ascii=False, indent=2), encoding="utf-8")


def _frame_center(frame: dict[str, Any]) -> list[float] | None:
    matrix = frame.get("transform_matrix")
    if not (isinstance(matrix, list) and len(matrix) >= 3):
        return None
    try:
        return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])]
    except (TypeError, ValueError, IndexError):
        return None


def _trajectory_continuity_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    centers = [center for frame in frames if (center := _frame_center(frame)) is not None]
    if len(centers) < 3:
        return {"passed": len(centers) > 1, "reason": "too_few_cameras_after_repair"}
    distances = [_distance(centers[index - 1], centers[index]) for index in range(1, len(centers))]
    median = _percentile(distances, 50) or 0.0
    max_distance = max(distances) if distances else 0.0
    jump_ratio = max_distance / max(median, 1e-12)
    return {"passed": median == 0 or jump_ratio <= 6.0, "median_step": median, "max_step": max_distance, "max_step_over_median": jump_ratio}


def _expanded_percentile_bounds(points: list[list[float]], percentile_min: float, percentile_max: float, expand_ratio: float) -> dict[str, list[float]]:
    mins: list[float] = []
    maxs: list[float] = []
    for axis in range(3):
        values = [point[axis] for point in points if math.isfinite(point[axis])]
        low = _percentile(values, percentile_min) or 0.0
        high = _percentile(values, percentile_max) or 0.0
        center = (low + high) / 2.0
        half = ((high - low) / 2.0) * expand_ratio
        mins.append(center - half)
        maxs.append(center + half)
    return {"min": mins, "max": maxs}


def _outside_bounds(point: list[float], bounds: dict[str, list[float]]) -> bool:
    return any(point[axis] < bounds["min"][axis] or point[axis] > bounds["max"][axis] for axis in range(3))


def _distance(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _first_config_value(config: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return finite[lower]
    fraction = rank - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction
