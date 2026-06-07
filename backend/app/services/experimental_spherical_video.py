from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - availability is environment-specific.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class SphericalVideoExperimentConfig:
    enabled: bool = False
    max_source_keyframes: int = 60
    yaw_degrees: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    pitch_degrees: tuple[float, ...] = (0.0,)
    pose_yaw_degrees: tuple[float, ...] = ()
    fov_degrees: float = 90.0
    output_width: int = 1600
    output_height: int = 900


SPHERICAL_RIG_VIEW_SOURCE_TYPES = {"spherical_video_keyframe_view", "panorama_station_view"}


def spherical_video_config(config: dict[str, Any]) -> SphericalVideoExperimentConfig:
    raw = config.get("experimental_360_video")
    if not isinstance(raw, dict):
        raw = {}
    return SphericalVideoExperimentConfig(
        enabled=bool(raw.get("enabled", False)),
        max_source_keyframes=max(1, int(raw.get("max_source_keyframes") or 60)),
        yaw_degrees=tuple(float(value) for value in (raw.get("yaw_degrees") or [0, 90, 180, 270])),
        pitch_degrees=tuple(float(value) for value in (raw.get("pitch_degrees") or [0])),
        pose_yaw_degrees=tuple(float(value) for value in (raw.get("pose_yaw_degrees") or [])),
        fov_degrees=float(raw.get("fov_degrees") or 90.0),
        output_width=max(64, int(raw.get("output_width") or 1600)),
        output_height=max(64, int(raw.get("output_height") or 900)),
    )


def is_equirectangular_size(width: int, height: int) -> bool:
    if height <= 0:
        return False
    ratio = width / height
    return 1.85 <= ratio <= 2.15 and width >= 300 and height >= 150


def evenly_spaced_frames(frames: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    if max_count >= len(frames):
        return list(frames)
    if max_count <= 1:
        return [frames[0]] if frames else []
    indices = np.linspace(0, len(frames) - 1, num=max_count).round().astype(int).tolist()
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(frames[index])
            seen.add(index)
    return selected


def project_equirectangular_to_perspective(
    source_path: Path,
    output_path: Path,
    *,
    yaw_degrees: float,
    pitch_degrees: float,
    fov_degrees: float,
    output_width: int,
    output_height: int,
) -> None:
    if cv2 is None:
        raise RuntimeError("opencv_unavailable")
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot_read_equirectangular_frame:{source_path}")
    height, width = image.shape[:2]
    if not is_equirectangular_size(width, height):
        raise RuntimeError(f"not_equirectangular_2_1:{width}x{height}")

    fov = math.radians(fov_degrees)
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    aspect = output_width / max(1, output_height)
    x = np.linspace(-math.tan(fov / 2.0) * aspect, math.tan(fov / 2.0) * aspect, output_width, dtype=np.float32)
    y = np.linspace(math.tan(fov / 2.0), -math.tan(fov / 2.0), output_height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)
    norm = np.sqrt(xx * xx + yy * yy + zz * zz)
    xx /= norm
    yy /= norm
    zz /= norm

    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    y_pitch = yy * cos_pitch - zz * sin_pitch
    z_pitch = yy * sin_pitch + zz * cos_pitch
    x_pitch = xx

    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x_world = x_pitch * cos_yaw + z_pitch * sin_yaw
    z_world = -x_pitch * sin_yaw + z_pitch * cos_yaw
    y_world = y_pitch

    longitude = np.arctan2(x_world, z_world)
    latitude = np.arcsin(np.clip(y_world, -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * width).astype(np.float32)
    map_y = ((0.5 - latitude / math.pi) * height).astype(np.float32)
    perspective = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), perspective, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def derive_spherical_video_views(
    *,
    asset_id: str,
    frames: list[dict[str, Any]],
    output_dir: Path,
    config: SphericalVideoExperimentConfig,
) -> list[dict[str, Any]]:
    selected_frames = evenly_spaced_frames(frames, config.max_source_keyframes)
    views: list[dict[str, Any]] = []
    for frame in selected_frames:
        source = Path(str(frame.get("image_path") or ""))
        metrics = frame.get("metrics") or {}
        if not source.exists() or not is_equirectangular_size(int(metrics.get("width") or 0), int(metrics.get("height") or 0)):
            continue
        frame_token = str(frame.get("frame_id") or source.stem).replace(":", "_")
        frame_index = frame.get("frame_index")
        for yaw in config.yaw_degrees:
            for pitch in config.pitch_degrees:
                output = output_dir / f"{frame_token}_yaw_{int(yaw):03d}_pitch_{int(pitch):+03d}.jpg"
                project_equirectangular_to_perspective(
                    source,
                    output,
                    yaw_degrees=yaw,
                    pitch_degrees=pitch,
                    fov_degrees=config.fov_degrees,
                    output_width=config.output_width,
                    output_height=config.output_height,
                )
                views.append(
                    {
                        "pano_view_id": f"{asset_id}:frame:{frame.get('frame_index')}:yaw:{yaw}:pitch:{pitch}",
                        "asset_id": asset_id,
                        "image_path": str(output),
                        "source_type": "spherical_video_keyframe_view",
                        "source_frame_id": frame.get("frame_id"),
                        "source_frame_index": frame_index,
                        "source_image_path": str(source),
                        "yaw": yaw,
                        "pitch": pitch,
                        "fov": config.fov_degrees,
                        "stream_id": f"yaw_{float(yaw):.3f}_pitch_{float(pitch):.3f}",
                        "usage": "pose_candidate",
                        "mapping": "experimental_equirectangular_video_to_perspective",
                    }
                )
    return views


def spherical_stream_key(entry: dict[str, Any]) -> tuple[float, float]:
    return (round(float(entry.get("yaw") or 0.0), 6), round(float(entry.get("pitch") or 0.0), 6))


def spherical_frame_key(entry: dict[str, Any]) -> tuple[int, str]:
    raw_index = entry.get("source_frame_index")
    if raw_index is None:
        raw_frame = str(entry.get("source_frame_id") or "")
        try:
            raw_index = int(raw_frame.rsplit(":", 1)[-1])
        except ValueError:
            raw_index = 0
    return (int(raw_index or 0), str(entry.get("source_frame_id") or ""))


def group_spherical_entries_by_stream(entries: list[dict[str, Any]]) -> dict[tuple[float, float], list[dict[str, Any]]]:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("source_type") not in SPHERICAL_RIG_VIEW_SOURCE_TYPES:
            continue
        if entry.get("pose_image") is None:
            continue
        groups.setdefault(spherical_stream_key(entry), []).append(entry)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=spherical_frame_key)
    return groups


def virtual_camera_relative_rotation(base_yaw: float, target_yaw: float, base_pitch: float, target_pitch: float) -> np.ndarray:
    yaw = math.radians(float(target_yaw) - float(base_yaw))
    pitch = math.radians(float(target_pitch) - float(base_pitch))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return rot_yaw @ rot_pitch


def apply_virtual_camera_rotation(transform_matrix: list[list[float]], *, base_yaw: float, target_yaw: float, base_pitch: float, target_pitch: float) -> list[list[float]]:
    matrix = np.array(transform_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform_matrix_must_be_4x4")
    relative = virtual_camera_relative_rotation(base_yaw, target_yaw, base_pitch, target_pitch)
    lifted = matrix.copy()
    lifted[:3, :3] = matrix[:3, :3] @ relative
    return lifted.tolist()
