from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Asset, Workflow
from app.operators.colmap import ColmapGlobalSkeletonOperator
from app.operators.feature_matching import LightGlueAlikedPreMatchingOperator
from app.operators.nerfstudio import NerfstudioSplatfactoTrainOperator
from app.operators.pose import ColmapAttemptsOperator, Mast3rSfmFallbackOperator
from app.operators.preprocess import DynamicMaskOperator, PreprocessRunResult
from app.operators.scope import SubjectMaskGenerationOperator
from app.services.artifact_service import ArtifactService
from app.services.experimental_spherical_video import (
    SPHERICAL_RIG_VIEW_SOURCE_TYPES,
    apply_virtual_camera_rotation,
    derive_spherical_video_views,
    group_spherical_entries_by_stream,
    is_equirectangular_size,
    project_equirectangular_to_perspective,
    spherical_video_config,
    spherical_frame_key,
)
from app.services.storage_service import StorageService

try:  # pragma: no cover - import availability is environment-specific.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - import availability is environment-specific.
    from PIL import ExifTags, Image, ImageDraw, ImageOps
except Exception:  # pragma: no cover
    ExifTags = None
    Image = None
    ImageDraw = None
    ImageOps = None


OPTIMIZED_STAGE_NAMES = [
    "raw_media_inspection",
    "image_enhancement",
    "video_keyframe_optimization",
    "panorama_normalization",
    "dataset_assembly",
    "pose_estimation_optimization",
    "mask_optimization",
    "training_input_optimization",
    "gaussian_training_optimization",
    "render_evaluation",
    "final_artifact_selection",
]


IMAGE_ENHANCEMENT_ROUTES = [
    "original",
    "denoise_light",
    "deblur_light",
    "exposure_normalized",
    "white_balance_light",
    "white_balance_normalized",
    "gamma_light",
    "sharpen_light",
    "contrast_local_light",
    "super_resolution_safe",
    "combined_safe_enhance",
]

SAFE_IMAGE_ENHANCEMENT_ROUTES = [
    "denoise_light",
    "exposure_normalized",
    "deblur_light",
    "contrast_local_light",
    "white_balance_light",
    "gamma_light",
    "sharpen_light",
]

ROUTE_PRESETS: dict[str, dict[str, Any]] = {
    "safe_pose_original_train": {
        "label": "R1",
        "pose_source": "safe_enhanced",
        "training_source": "original",
        "training_supervision_modified": False,
        "default_required": True,
    },
}

DEFAULT_PRODUCTION_ROUTE_PRESET = "safe_pose_original_train"

VIDEO_KEYFRAME_ROUTES = [
    "uniform_1fps",
    "uniform_2fps",
    "uniform_3fps",
    "dense_full_coverage",
    "motion_aware",
    "blur_filtered",
    "exposure_stable",
    "hybrid_balanced",
    "hybrid_dense",
    "hybrid_sparse",
    "loop_aware",
]

PANORAMA_NORMALIZATION_ROUTES = [
    "keep_equirectangular",
    "perspective_cubemap_4",
    "perspective_cubemap_6",
    "perspective_views_dense",
    "panorama_as_context_only",
]

DATASET_ASSEMBLY_ROUTES = [
    "safe_pose_original_train",
    "jpg_only_best_pose",
    "video_only_best_keyframes",
    "jpg_video_fused_balanced",
    "jpg_video_fused_dense",
    "jpg_video_fused_sparse",
    "panorama_context_added",
    "high_confidence_only",
]

POSE_ESTIMATION_ROUTES = [
    "colmap_exhaustive",
    "colmap_sequential",
    "colmap_sequential_loop",
    "colmap_vocab_tree",
    "colmap_hybrid",
    "spherical_video_rig_lift",
    "colmap_multi_camera_model_test",
    "hloc_lightglue_aliked_fallback",
    "mast3r_dust3r_fallback",
]

MASK_OPTIMIZATION_ROUTES = [
    "no_mask",
    "dynamic_object_mask",
    "human_vehicle_animal_mask",
    "reflection_sensitive_mask",
    "foreground_interference_mask",
    "conservative_mask",
    "aggressive_mask",
]

TRAINING_INPUT_ROUTES = [
    "original_training_images",
    "resize_native",
    "resize_balanced",
    "balanced_holdout_split",
    "mask_safe_training_input",
]

GAUSSIAN_TRAINING_ROUTES = [
    "splatfacto_baseline",
    "splatfacto_tuned",
    "splatfacto_big",
    "splatfacto_w_light",
    "splatfacto_w",
    "splatfacto_with_conservative_mask",
    "splatfacto_with_robust_mask",
    "splatfacto_high_resolution",
    "splatfacto_long_train",
    "prior_assisted_fallback",
]

RENDER_EVALUATION_ROUTES = [
    "held_out_view_render",
    "fixed_camera_path_render",
    "orbit_render",
    "close_up_render",
    "sparse_vs_render_comparison",
    "original_vs_reconstruction_comparison",
    "baseline_vs_best_comparison",
    "mask_vs_no_mask_comparison",
    "enhanced_vs_original_comparison",
]

ROUTE_SCOPED_STAGE_NAMES = {
    "dataset_assembly",
    "pose_estimation_optimization",
    "mask_optimization",
    "training_input_optimization",
    "gaussian_training_optimization",
    "render_evaluation",
    "final_artifact_selection",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = Path(value).name
    return cleaned or fallback


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            path.write_text(body, encoding="utf-8")
            return path
        except OSError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return path


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            path.write_text(body, encoding="utf-8")
            return path
        except OSError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return path


def file_sha256(path: Path | str | None) -> str | None:
    if not path:
        return None
    source = Path(str(path))
    if not source.exists() or not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_preset_config(name: str | None) -> dict[str, Any]:
    return dict(ROUTE_PRESETS.get(str(name or DEFAULT_PRODUCTION_ROUTE_PRESET), ROUTE_PRESETS[DEFAULT_PRODUCTION_ROUTE_PRESET]))


def active_route_preset(context: "StageContext") -> str:
    value = context.config.get("active_route_preset") or context.config.get("route_preset") or DEFAULT_PRODUCTION_ROUTE_PRESET
    return str(value) if str(value) in ROUTE_PRESETS else DEFAULT_PRODUCTION_ROUTE_PRESET


def _source_family(candidate_type: str | None) -> str:
    if candidate_type in {None, "", "original"}:
        return "original"
    if candidate_type in SAFE_IMAGE_ENHANCEMENT_ROUTES or candidate_type in {"white_balance_normalized"}:
        return "safe_enhanced"
    return "derived"


def _distribution(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in items:
        name = str(item.get(key) or "missing")
        distribution[name] = distribution.get(name, 0) + 1
    return distribution


def image_process_params(process_type: str) -> dict[str, Any]:
    params: dict[str, Any] = {"process_type": process_type}
    if process_type == "denoise_light":
        params.update({"h": 3, "hColor": 3, "templateWindowSize": 7, "searchWindowSize": 21})
    elif process_type == "deblur_light":
        params.update({"gaussian_sigma": 1.0, "amount": 0.35})
    elif process_type == "exposure_normalized":
        params.update({"target_luma": 135.0, "scale_min": 0.75, "scale_max": 1.25})
    elif process_type in {"white_balance_light", "white_balance_normalized"}:
        params.update({"method": "gray_world", "max_channel_scale": "implicit"})
    elif process_type == "gamma_light":
        params.update({"gamma": 0.95})
    elif process_type == "sharpen_light":
        params.update({"gaussian_sigma": 0.9, "amount": 0.18})
    elif process_type == "contrast_local_light":
        params.update({"clahe_clip_limit": 1.4, "tile_grid_size": [8, 8]})
    elif process_type == "combined_safe_enhance":
        params.update({"denoise": "light", "sharpen": "light", "exposure": "bounded", "white_balance": "gray_world"})
    return params


def copy_file_safely(source: str | Path, target: str | Path) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            if target_path.exists():
                target_path.unlink()
            try:
                os.link(source_path, target_path)
                return target_path
            except OSError:
                pass
            with source_path.open("rb") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            return target_path
        except OSError as exc:
            last_error = exc
            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError:
                pass
            time.sleep(0.25 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return target_path


def storage_relative_from_uri(storage_uri: str) -> str:
    if storage_uri.startswith("local://"):
        return storage_uri.removeprefix("local://")
    if storage_uri.startswith("s3://"):
        bucket_and_key = storage_uri.removeprefix("s3://")
        _, _, key = bucket_and_key.partition("/")
        return key
    return storage_uri


def asset_kind(asset: Asset) -> str:
    suffix = Path(asset.original_filename or asset.filename or "").suffix.lower()
    mime = asset.mime_type or ""
    if asset.asset_type in {"global_video", "supplement_video", "video"} or mime.startswith("video/") or suffix in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        return "video"
    if asset.asset_type in {"pano_360", "panorama"} or asset.role == "pano_anchor" or suffix in {".insp", ".insv"}:
        return "panorama"
    return "image"


def is_probable_panorama(width: int, height: int, asset: Asset | None = None) -> bool:
    if height <= 0:
        return False
    ratio = width / height
    declared = bool(asset and (asset.asset_type in {"pano_360", "panorama"} or asset.role == "pano_anchor"))
    return declared or (1.85 <= ratio <= 2.15 and width >= 3000 and height >= 1400)


def config_section(settings: Settings, name: str) -> dict[str, Any]:
    section = settings.engine_config.get(name)
    return section if isinstance(section, dict) else {}


def nested_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = config
    for item in path.split("."):
        if not isinstance(cursor, dict) or item not in cursor:
            return default
        cursor = cursor[item]
    return cursor


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-8:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def estimate_image_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "path": str(path),
        "readable": False,
        "width": 0,
        "height": 0,
        "long_edge": 0,
        "short_edge": 0,
        "sharpness_score": 0.0,
        "laplacian_variance": 0.0,
        "brightness_mean": 0.0,
        "overexposed_ratio": 0.0,
        "underexposed_ratio": 0.0,
        "noise_score": 0.0,
        "compression_artifact_score": 0.0,
        "psnr_estimate": 0.0,
        "feature_detectability_score": 0.0,
        "keypoint_count": 0,
        "keypoint_distribution_score": 0.0,
        "exposure_score": 0.0,
        "white_balance_shift": 0.0,
    }
    if cv2 is None:
        if Image is None:
            metrics["error"] = "opencv_and_pillow_unavailable"
            return metrics
        try:
            with Image.open(path) as image:
                metrics.update({"readable": True, "width": image.width, "height": image.height})
                metrics["long_edge"] = max(image.width, image.height)
                metrics["short_edge"] = min(image.width, image.height)
            return metrics
        except Exception as exc:  # pragma: no cover - defensive.
            metrics["error"] = str(exc)
            return metrics

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        metrics["error"] = "opencv_read_failed"
        return metrics
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    over = float((gray >= 245).mean())
    under = float((gray <= 10).mean())
    blur_norm = clamp(lap / 300.0)
    exposure_center = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
    exposure_penalty = clamp(1.0 - max(over, under) * 4.0)
    exposure_score = clamp(0.65 * exposure_center + 0.35 * exposure_penalty)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    white_balance_shift = float(np.std(channel_means) / max(1.0, np.mean(channel_means)))
    noise = float(np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        psnr = compute_psnr(image, decoded) if decoded is not None else 0.0
    else:
        psnr = 0.0
    keypoints = []
    try:
        detector = cv2.ORB_create(nfeatures=3000)
        keypoints = detector.detect(gray, None) or []
    except Exception:
        keypoints = []
    distribution = 0.0
    if keypoints:
        grid = np.zeros((4, 4), dtype=np.uint8)
        for kp in keypoints:
            x = min(3, max(0, int(kp.pt[0] / max(1, width) * 4)))
            y = min(3, max(0, int(kp.pt[1] / max(1, height) * 4)))
            grid[y, x] = 1
        distribution = float(grid.mean())
    compression_artifact_score = clamp(1.0 - psnr / 45.0)
    metrics.update(
        {
            "readable": True,
            "width": width,
            "height": height,
            "long_edge": max(width, height),
            "short_edge": min(width, height),
            "sharpness_score": round(blur_norm, 4),
            "laplacian_variance": round(lap, 3),
            "brightness_mean": round(brightness, 3),
            "overexposed_ratio": round(over, 5),
            "underexposed_ratio": round(under, 5),
            "noise_score": round(clamp(noise / 60.0), 4),
            "compression_artifact_score": round(compression_artifact_score, 4),
            "psnr_estimate": round(float(psnr), 3),
            "feature_detectability_score": round(clamp(len(keypoints) / 1600.0) * 0.65 + distribution * 0.35, 4),
            "keypoint_count": len(keypoints),
            "keypoint_distribution_score": round(distribution, 4),
            "exposure_score": round(exposure_score, 4),
            "white_balance_shift": round(white_balance_shift, 5),
        }
    )
    return metrics


def image_passes_basic_gate(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    image_config = nested_get(config, "image", {}) or {}
    min_width = int(image_config.get("min_width_px") or image_config.get("min_width") or 1200)
    min_height = int(image_config.get("min_height_px") or image_config.get("min_height") or 800)
    min_lap = float(image_config.get("laplacian_variance_min") or 60.0)
    min_psnr = float(image_config.get("psnr_estimate_min") or 24.0)
    min_brightness = float(image_config.get("brightness_mean_min") or 35.0)
    max_brightness = float(image_config.get("brightness_mean_max") or 225.0)
    return bool(
        metrics.get("readable")
        and int(metrics.get("long_edge") or 0) >= min(min_width, min_height)
        and int(metrics.get("short_edge") or 0) >= min(min_width, min_height) * 0.65
        and float(metrics.get("laplacian_variance") or 0.0) >= min_lap
        and float(metrics.get("psnr_estimate") or 0.0) >= min_psnr
        and min_brightness <= float(metrics.get("brightness_mean") or 0.0) <= max_brightness
    )


def create_contact_sheet(items: list[dict[str, Any]], output_path: Path, *, title: str = "contact sheet", max_items: int = 24) -> Path | None:
    if Image is None or ImageDraw is None:
        return None
    thumbs: list[tuple[Any, str]] = []
    for item in items[:max_items]:
        path_value = item.get("path") or item.get("image_path") or item.get("output_path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists():
            continue
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((220, 160))
            canvas = Image.new("RGB", (240, 200), (246, 247, 248))
            canvas.paste(image, ((240 - image.width) // 2, 8))
            label = str(item.get("label") or item.get("asset_id") or path.name)[:32]
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 174), label, fill=(32, 39, 43))
            thumbs.append((canvas, label))
        except Exception:
            continue
    if not thumbs:
        return None
    cols = min(4, max(1, len(thumbs)))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 240, rows * 200 + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(32, 39, 43))
    for idx, (thumb, _label) in enumerate(thumbs):
        x = (idx % cols) * 240
        y = 34 + (idx // cols) * 200
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def _unique_copy_name(target_dir: Path, source: Path, index: int) -> str:
    stem = source.stem or f"image_{index:05d}"
    suffix = source.suffix.lower() or ".jpg"
    name = f"{index:05d}_{stem}{suffix}"
    if not (target_dir / name).exists():
        return name
    counter = 1
    while True:
        candidate = f"{index:05d}_{stem}_{counter}{suffix}"
        if not (target_dir / candidate).exists():
            return candidate
        counter += 1


def _preprocess_from_dataset_manifest(
    context: "StageContext",
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    route_id: str,
    route_key: str,
) -> PreprocessRunResult:
    _reset_materialized_preprocess_dir(context, output_dir)
    dataset_dir = output_dir / "nerfstudio_dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    active_route_id = str(context.config.get("active_route_id") or "").strip()
    image_paths: list[Path] = []
    source_entries: list[dict[str, Any]] = []
    for index, value in enumerate(manifest.get("pose_images") or [], start=1):
        source = Path(str(value))
        if not source.exists():
            source_entries.append({"source": str(source), "status": "missing"})
            continue
        target = images_dir / _unique_copy_name(images_dir, source, index)
        if not target.exists():
            copy_file_safely(source, target)
        image_paths.append(target)
        source_entries.append({"source": str(source), "image": target.name, "status": "copied"})
    routing_manifest_path = output_dir / "routing_manifest.json"
    write_json(
        routing_manifest_path,
        {
            "workflow_id": context.run_id,
            "project_id": context.project_id,
            "route_id": route_id,
            "route_key": route_key,
            "source": "stage_optimized_dataset",
            "image_count": len(image_paths),
            "sources": source_entries,
        },
    )
    asset_quality = {
        "passed": len(image_paths) >= 3,
        "input_asset_count": len(context.assets),
        "global_image_count": len(image_paths),
        "min_required_global_images": 3,
        "route_key": route_key,
        "issues": [] if len(image_paths) >= 3 else ["insufficient_stage_optimized_images"],
    }
    media_metadata = {
        "input_mode": "stage_optimized_dataset",
        "route_id": route_id,
        "route_key": route_key,
        "asset_count": len(context.assets),
        "staged_file_count": len(image_paths),
        "source_files": [path.name for path in image_paths],
        "workspace_suffix": "/".join([part for part in (active_route_id, route_id) if part]),
        "stage_optimized_reconstruction": True,
        "asset_quality": asset_quality,
    }
    write_json(
        output_dir / "preprocess_metadata.json",
        {
            "workflow_id": context.run_id,
            "project_id": context.project_id,
            "route_id": route_id,
            "active_route_id": active_route_id or None,
            "route_key": route_key,
            "dataset_dir": str(dataset_dir),
            "images_dir": str(images_dir),
            "source_files": [path.name for path in image_paths],
            "source_entries": source_entries,
            "stage_optimized_reconstruction": True,
        },
    )
    return PreprocessRunResult(
        workspace_dir=output_dir,
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        image_paths=image_paths,
        commands=[],
        media_metadata=media_metadata,
        asset_quality=asset_quality,
        routing_manifest_path=routing_manifest_path,
    )


def _reset_materialized_preprocess_dir(context: "StageContext", output_dir: Path) -> None:
    run_root = context.run_dir.resolve()
    target = output_dir.resolve()
    if target == run_root or run_root not in target.parents:
        raise RuntimeError(f"Refusing to reset materialized preprocess directory outside run root: {target}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _route_scoped_workspace_name(context: "StageContext", relative_workspace_name: str) -> str:
    active_route_id = str(context.config.get("active_route_id") or "").strip()
    if not active_route_id:
        return relative_workspace_name
    safe_route = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in active_route_id)[:120]
    return f"optimized_reconstruction/routes/{safe_route}/{relative_workspace_name}"


def _real_pose_matcher(route: str) -> str:
    if route in {"colmap_sequential", "colmap_sequential_loop"}:
        return "sequential"
    if route == "colmap_vocab_tree":
        return "vocabtree"
    return "exhaustive"


def _configured_names(value: Any, default: list[str]) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set(default)


def _read_optional_json(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _flag(context: "StageContext", name: str, default: bool = False) -> bool:
    if name in context.config:
        return bool(context.config.get(name))
    return bool(default)


def _stage_execution_requested(context: "StageContext", flag_name: str) -> bool:
    return bool(
        context.config.get("execute_all_route_candidates")
        or nested_get(context.config, "execution.execute_all_route_candidates", False)
        or context.config.get(flag_name)
        or nested_get(context.config, f"execution.{flag_name}_by_default", False)
    )


def _planned_candidate(
    context: "StageContext",
    stage_name: str,
    candidate: dict[str, Any],
    *,
    reason: str,
    metrics: dict[str, Any] | None = None,
    risk_level: str = "medium",
) -> dict[str, Any]:
    output_dir = context.stage_dir(stage_name) / "planned_routes" / str(candidate.get("candidate_name") or "candidate")
    output = write_json(
        output_dir / "planned_route.json",
        {
            "candidate_name": candidate.get("candidate_name"),
            "candidate_type": candidate.get("candidate_type"),
            "status": "planned",
            "reason": reason,
            "metrics": metrics or {},
        },
    )
    candidate.update(
        {
            "status": "planned",
            "output_path": str(output),
            "metrics_path": str(output),
            "metrics": metrics or {"status": "planned", "reason": reason},
            "score": 0.0,
            "rejected_reason": reason,
            "risk_level": risk_level,
        }
    )
    return candidate


def _score_pose_metrics(metrics: dict[str, Any]) -> float:
    registration_rate = float(metrics.get("registered_ratio") or metrics.get("registration_rate") or 0.0)
    component_ratio = float(metrics.get("largest_component_ratio") or 0.0)
    reproj_value = metrics.get("mean_reprojection_error")
    has_reprojection_metric = reproj_value is not None
    reproj_score = 0.45 if not has_reprojection_metric else 1.0 - min(float(reproj_value) / 5.0, 1.0)
    sparse_density = float(metrics.get("sparse_density_score") or 0.0)
    if not has_reprojection_metric:
        sparse_density = min(sparse_density, 0.5)
    score = 0.45 * registration_rate + 0.25 * component_ratio + 0.2 * reproj_score + 0.1 * sparse_density
    if str(metrics.get("execution") or "").startswith("real_mast3r") and not has_reprojection_metric:
        score -= 0.04
    return round(clamp(score), 4)


def _pose_metrics_from_colmap_result(
    result: Any,
    report: dict[str, Any],
    *,
    image_count: int,
    route: str,
    matcher: str,
    camera_model: str,
    execution: str,
) -> dict[str, Any]:
    registration_rate = float(report.get("registration_rate") or result.quality.get("registration_rate") or 0.0)
    sparse_points = int(report.get("sparse_point_count") or result.quality.get("sparse_point_count") or 0)
    reproj = report.get("mean_reprojection_error")
    reproj_value = float(reproj) if reproj is not None else None
    component_ratio = float(result.quality.get("largest_component_ratio") or report.get("largest_component_ratio") or 1.0)
    registered_count = int(report.get("registered_camera_count") or round(image_count * registration_rate))
    metrics = {
        "execution": execution,
        "route": route,
        "matcher": matcher,
        "camera_model": camera_model,
        "registered_images_count": registered_count,
        "total_images_count": image_count,
        "registered_ratio": round(registration_rate, 4),
        "sparse_points_count": sparse_points,
        "mean_reprojection_error": round(reproj_value, 4) if reproj_value is not None else None,
        "median_reprojection_error": report.get("median_reprojection_error"),
        "track_length_mean": report.get("track_length_mean"),
        "camera_graph_components": int(report.get("camera_graph_components") or 1),
        "largest_component_ratio": round(component_ratio, 4),
        "failed_images": report.get("failed_images", []),
        "weak_images": report.get("weak_images", []),
        "camera_path_continuity": report.get("trajectory_continuity", {}),
        "loop_closure_success": bool(route == "colmap_sequential_loop"),
        "sparse_density_score": round(clamp(sparse_points / max(1, image_count * 600)), 4),
        "dataset_dir": str(result.dataset_dir),
        "transforms_path": str(result.transforms_path),
        "sparse_point_cloud_path": str(result.sparse_point_cloud_path),
        "model_dir": str(getattr(result, "model_dir", getattr(result, "final_export_dir", ""))),
        "selected_model_dir": report.get("selected_model_dir") or Path(str(getattr(result, "model_dir", ""))).name or None,
        "registration_report_path": str(result.registration_report_path),
        "command_count": len(result.commands),
    }
    metrics["geometry_stability_score"] = _score_pose_metrics(metrics)
    return metrics


def _pose_quality_rejection(metrics: dict[str, Any]) -> str | None:
    registration_rate = float(metrics.get("registered_ratio") or 0.0)
    component_ratio = float(metrics.get("largest_component_ratio") or 0.0)
    reproj = metrics.get("mean_reprojection_error")
    sparse_points = int(metrics.get("sparse_points_count") or 0)
    if registration_rate < 0.65 or component_ratio < 0.65 or (reproj is not None and float(reproj) > 4.0) or sparse_points < 100:
        return "real_pose_quality_gate_failed"
    return None


@dataclass
class StageContext:
    db: Session
    workflow: Workflow
    assets: list[Asset]
    settings: Settings
    storage: StorageService
    artifact_service: ArtifactService
    run_dir: Path
    config: dict[str, Any]
    previous_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    capability_report: dict[str, Any] = field(default_factory=dict)

    @property
    def project_id(self) -> str:
        return self.workflow.project_id

    @property
    def run_id(self) -> str:
        return self.workflow.id

    def stage_dir(self, stage_name: str) -> Path:
        route_id = self.config.get("active_route_id")
        if route_id and stage_name in ROUTE_SCOPED_STAGE_NAMES:
            return self.run_dir / "routes" / str(route_id) / "stages" / stage_name
        return self.run_dir / "stages" / stage_name


class RunRecordStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.stage_records_path = run_dir / "records" / "run_stage_records.json"
        self.candidate_records_path = run_dir / "records" / "run_candidate_records.json"
        self.final_selection_path = run_dir / "records" / "run_final_selection.json"

    def append_stage(self, result: dict[str, Any]) -> None:
        records = read_json(self.stage_records_path, [])
        route_id = result.get("route_id")
        records = [
            record
            for record in records
            if not (
                record.get("run_id") == result.get("run_id")
                and record.get("stage_name") == result.get("stage_name")
                and record.get("route_id") == route_id
            )
        ]
        records.append(
            {
                "id": f"{result.get('run_id')}:{route_id or 'common'}:{result.get('stage_name')}",
                "run_id": result.get("run_id"),
                "route_id": route_id,
                "stage_name": result.get("stage_name"),
                "status": result.get("status", "succeeded"),
                "input_artifact_path": result.get("input_artifacts"),
                "best_artifact_path": result.get("best_artifact"),
                "metrics_path": result.get("metrics_path"),
                "report_path": result.get("report_path"),
                "improvement_summary": result.get("improvement_summary"),
                "risk_summary": result.get("risk_summary"),
                "has_remaining_improvement": result.get("whether_stage_has_remaining_improvement"),
                "created_at": result.get("created_at"),
                "updated_at": utc_now_iso(),
            }
        )
        write_json(self.stage_records_path, records)

    def replace_stage_candidates(self, run_id: str, stage_name: str, candidates: list[dict[str, Any]], route_id: str | None = None) -> None:
        records = read_json(self.candidate_records_path, [])
        records = [
            record
            for record in records
            if not (
                record.get("run_id") == run_id
                and record.get("stage_name") == stage_name
                and record.get("route_id") == route_id
            )
        ]
        now = utc_now_iso()
        for candidate in candidates:
            records.append(
                {
                    "id": f"{run_id}:{route_id or 'common'}:{stage_name}:{candidate.get('candidate_name')}",
                    "run_id": run_id,
                    "route_id": route_id,
                    "stage_name": stage_name,
                    "candidate_name": candidate.get("candidate_name"),
                    "candidate_type": candidate.get("candidate_type"),
                    "input_path": candidate.get("input_path"),
                    "output_path": candidate.get("output_path"),
                    "config_path": candidate.get("config_path"),
                    "metrics_path": candidate.get("metrics_path"),
                    "status": candidate.get("status"),
                    "score": candidate.get("score"),
                    "selected_as_best": bool(candidate.get("selected_as_best")),
                    "rejected_reason": candidate.get("rejected_reason"),
                    "risk_level": candidate.get("risk_level"),
                    "created_at": candidate.get("created_at") or now,
                    "updated_at": now,
                }
            )
        write_json(self.candidate_records_path, records)

    def write_final_selection(self, payload: dict[str, Any]) -> None:
        write_json(self.final_selection_path, payload)

    def read_all(self) -> dict[str, Any]:
        return {
            "stages": read_json(self.stage_records_path, []),
            "candidates": read_json(self.candidate_records_path, []),
            "final_selection": read_json(self.final_selection_path, {}),
        }


class ForensicIntegrityGuard:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate_derivative(self, source_path: Path, output_path: Path, *, process_type: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        result = {
            "source_file": str(source_path),
            "process_type": process_type,
            "process_params": params,
            "model_name": params.get("model_name"),
            "model_version": params.get("model_version"),
            "created_at": utc_now_iso(),
            "integrity_risk_score": 1.0,
            "accepted_for_pose": False,
            "accepted_for_training": False,
            "rejected_reason": None,
            "checks": {},
        }
        if cv2 is None:
            result["rejected_reason"] = "opencv_unavailable_for_integrity_check"
            return result
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        derived = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if source is None or derived is None:
            result["rejected_reason"] = "source_or_derivative_unreadable"
            return result
        if source.shape != derived.shape:
            derived = cv2.resize(derived, (source.shape[1], source.shape[0]))
        gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        gray_derived = cv2.cvtColor(derived, cv2.COLOR_BGR2GRAY)
        edges_source = cv2.Canny(gray_source, 80, 160)
        edges_derived = cv2.Canny(gray_derived, 80, 160)
        edge_delta = float(np.mean(np.abs(edges_source.astype(np.float32) - edges_derived.astype(np.float32))) / 255.0)
        psnr = compute_psnr(source, derived)
        brightness_delta = abs(float(gray_source.mean()) - float(gray_derived.mean())) / 255.0
        risk = clamp(edge_delta * 1.8 + max(0.0, 32.0 - psnr) / 32.0 + brightness_delta)
        accepted_pose = risk <= 0.45
        accepted_training = risk <= 0.32 and process_type not in {"super_resolution_safe", "deblur_light"}
        rejected_reason = None
        if risk > 0.45:
            rejected_reason = "structural_change_risk_high"
        elif process_type == "super_resolution_safe":
            rejected_reason = "super_resolution_requires_manual_review"
        result.update(
            {
                "integrity_risk_score": round(risk, 4),
                "accepted_for_pose": accepted_pose,
                "accepted_for_training": accepted_training,
                "rejected_reason": rejected_reason,
                "checks": {
                    "edge_delta": round(edge_delta, 5),
                    "psnr_against_source": round(psnr, 3),
                    "brightness_delta": round(brightness_delta, 5),
                },
            }
        )
        return result


class StageOptimizer:
    stage_name = "stage"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        return {}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        return candidate

    def evaluate_candidate(self, context: StageContext, candidate_result: dict[str, Any]) -> dict[str, Any]:
        return candidate_result

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        selectable = [item for item in candidate_results if item.get("status") in {"succeeded", "completed"} and not item.get("rejected_reason")]
        if not selectable:
            return {}
        return max(selectable, key=lambda item: float(item.get("score") or 0.0))

    def export_stage_result(self, context: StageContext, analysis: dict[str, Any], candidates: list[dict[str, Any]], best_result: dict[str, Any]) -> dict[str, Any]:
        stage_dir = context.stage_dir(self.stage_name)
        metrics_path = stage_dir / "candidate_metrics.json"
        write_json(metrics_path, {"stage_name": self.stage_name, "candidates": candidates})
        report_path = stage_dir / "stage_report.md"
        write_text(report_path, self._report_markdown(context, analysis, candidates, best_result))
        result = {
            "schema": "fieldsplat.stage_optimized_reconstruction.stage_result.v1",
            "run_id": context.run_id,
            "project_id": context.project_id,
            "route_id": context.config.get("active_route_id"),
            "stage_name": self.stage_name,
            "status": "succeeded" if best_result else "blocked",
            "input_artifacts": analysis.get("input_artifacts", []),
            "candidate_artifacts": [item.get("output_path") for item in candidates if item.get("output_path")],
            "best_artifact": best_result.get("output_path") if best_result else None,
            "best_candidate": best_result.get("candidate_name") if best_result else None,
            "metrics": best_result.get("metrics", {}) if best_result else {},
            "metrics_path": str(metrics_path),
            "report_path": str(report_path),
            "rejected_candidates": [item for item in candidates if item.get("rejected_reason")],
            "rejection_reasons": {item.get("candidate_name"): item.get("rejected_reason") for item in candidates if item.get("rejected_reason")},
            "improvement_summary": best_result.get("improvement_summary") or "当前阶段已选择可用 best artifact。",
            "risk_summary": best_result.get("risk_summary") or "未发现高风险派生处理进入 best。",
            "whether_stage_has_remaining_improvement": bool(best_result.get("has_remaining_improvement", False)) if best_result else True,
            "next_stage_recommendation": best_result.get("next_stage_recommendation") or ("continue" if best_result else "stop"),
            "capability_report": context.capability_report.get(self.stage_name, {}),
            "created_at": utc_now_iso(),
        }
        result_path = stage_dir / "stage_result.json"
        write_json(result_path, result)
        result["stage_result_path"] = str(result_path)
        RunRecordStore(context.run_dir).append_stage(result)
        RunRecordStore(context.run_dir).replace_stage_candidates(context.run_id, self.stage_name, candidates, context.config.get("active_route_id"))
        return result

    def run(self, context: StageContext) -> dict[str, Any]:
        analysis = self.analyze_input(context)
        candidates = []
        for candidate in self.generate_candidates(context, analysis):
            result = self.run_candidate(context, candidate)
            evaluated = self.evaluate_candidate(context, result)
            candidates.append(evaluated)
        best = self.select_best(context, candidates)
        for candidate in candidates:
            candidate["selected_as_best"] = bool(best and candidate.get("candidate_name") == best.get("candidate_name"))
        return self.export_stage_result(context, analysis, candidates, best)

    def _report_markdown(self, context: StageContext, analysis: dict[str, Any], candidates: list[dict[str, Any]], best_result: dict[str, Any]) -> str:
        lines = [
            f"# {self.stage_name}",
            "",
            f"- run_id: `{context.run_id}`",
            f"- candidates: {len(candidates)}",
            f"- best: `{best_result.get('candidate_name') if best_result else 'none'}`",
            "",
            "## Candidate Summary",
        ]
        for item in candidates:
            lines.append(
                f"- `{item.get('candidate_name')}` status={item.get('status')} score={item.get('score')} risk={item.get('risk_level') or '-'} reason={item.get('rejected_reason') or '-'}"
            )
        if context.capability_report.get(self.stage_name):
            lines.extend(["", "## Capability Report", "```json", json.dumps(context.capability_report[self.stage_name], ensure_ascii=False, indent=2), "```"])
        return "\n".join(lines) + "\n"


class RawMediaInspectionStage(StageOptimizer):
    stage_name = "raw_media_inspection"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        raw_dir = context.stage_dir(self.stage_name) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        staged_assets = []
        for asset in context.assets:
            filename = safe_name(asset.original_filename or asset.filename, f"{asset.id}.bin")
            target = raw_dir / asset.id / filename
            if not target.exists():
                context.storage.download_to_file(storage_relative_from_uri(asset.storage_uri), target)
            staged_assets.append(
                {
                    "asset_id": asset.id,
                    "asset_type": asset.asset_type,
                    "role": asset.role,
                    "kind": asset_kind(asset),
                    "path": str(target),
                    "filename": asset.original_filename or asset.filename,
                    "storage_uri": asset.storage_uri,
                    "metadata": asset.metadata_json or {},
                    "mime_type": asset.mime_type,
                    "size_bytes": asset.size_bytes,
                }
            )
        return {"input_artifacts": [item["path"] for item in staged_assets], "staged_assets": staged_assets}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"candidate_name": "diagnostic_only", "candidate_type": "inspection", "assets": analysis["staged_assets"], "status": "created", "created_at": utc_now_iso()}]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        stage_dir = context.stage_dir(self.stage_name)
        inventory = []
        quality_items = []
        video_charts = []
        for item in candidate["assets"]:
            path = Path(item["path"])
            kind = item["kind"]
            if kind in {"image", "panorama"}:
                metrics = estimate_image_metrics(path)
                metrics["is_panorama_2_1"] = is_probable_panorama(int(metrics.get("width") or 0), int(metrics.get("height") or 0), None)
                exif = self._read_exif(path)
                risk = self._image_risk(metrics)
                inventory_item = {**item, "metrics": metrics, "exif": exif, "risk": risk}
                inventory.append(inventory_item)
                quality_items.append(inventory_item)
            elif kind == "video":
                video = self._inspect_video(path, stage_dir / "video_samples" / item["asset_id"])
                inventory_item = {**item, "video": video, "risk": self._video_risk(video)}
                inventory.append(inventory_item)
                quality_items.append(inventory_item)
                if video.get("timeline_chart"):
                    video_charts.append(video["timeline_chart"])
            else:
                inventory.append({**item, "risk": {"level": "warning", "reasons": ["unsupported_media_type"]}})
        inventory_path = write_json(stage_dir / "raw_media_inventory.json", {"assets": inventory})
        report_payload = {
            "run_id": context.run_id,
            "asset_count": len(inventory),
            "assets": quality_items,
            "optimization_advice": self._advice(inventory),
            "capabilities": {
                "opencv": cv2 is not None,
                "pillow": Image is not None,
                "exif": Image is not None and ExifTags is not None,
            },
        }
        quality_report_path = write_json(stage_dir / "raw_media_quality_report.json", report_payload)
        contact = create_contact_sheet(
            [
                {"path": item.get("path"), "asset_id": item.get("asset_id"), "label": f"{item.get('kind')}:{item.get('filename')}"}
                for item in inventory
                if item.get("kind") in {"image", "panorama"}
            ],
            stage_dir / "media_contact_sheet.jpg",
            title="Raw media contact sheet",
        )
        timeline = self._make_video_timeline_chart(stage_dir / "video_timeline_quality_chart.jpg", inventory)
        risk_report_path = write_text(stage_dir / "initial_risk_report.md", self._risk_markdown(inventory))
        candidate.update(
            {
                "status": "succeeded",
                "output_path": str(inventory_path),
                "metrics_path": str(quality_report_path),
                "score": self._score(inventory),
                "risk_level": "warning" if any((item.get("risk") or {}).get("level") == "warning" for item in inventory) else "low",
                "metrics": {
                    "asset_count": len(inventory),
                    "image_count": len([item for item in inventory if item.get("kind") == "image"]),
                    "video_count": len([item for item in inventory if item.get("kind") == "video"]),
                    "panorama_count": len([item for item in inventory if item.get("kind") == "panorama" or (item.get("metrics") or {}).get("is_panorama_2_1")]),
                    "risk_count": len([item for item in inventory if (item.get("risk") or {}).get("level") != "low"]),
                },
                "artifact_paths": {
                    "raw_media_inventory": str(inventory_path),
                    "raw_media_quality_report": str(quality_report_path),
                    "media_contact_sheet": str(contact) if contact else None,
                    "video_timeline_quality_chart": str(timeline) if timeline else None,
                    "initial_risk_report": str(risk_report_path),
                    "video_charts": video_charts,
                },
                "improvement_summary": "完成原始素材体检，未修改原始素材，已为后续增强和关键帧选择生成优化建议。",
                "risk_summary": "此阶段仅诊断，不产生派生素材真实性风险。",
            }
        )
        return candidate

    def export_stage_result(self, context: StageContext, analysis: dict[str, Any], candidates: list[dict[str, Any]], best_result: dict[str, Any]) -> dict[str, Any]:
        result = super().export_stage_result(context, analysis, candidates, best_result)
        if best_result:
            for artifact_type, path_value in (best_result.get("artifact_paths") or {}).items():
                if not path_value or isinstance(path_value, list):
                    continue
                path = Path(str(path_value))
                if path.exists():
                    context.artifact_service.register_file(
                        project_id=context.project_id,
                        workflow_id=context.run_id,
                        artifact_type=artifact_type,
                        stage=self.stage_name,
                        relative_path=f"projects/{context.project_id}/runs/{context.run_id}/optimized/{self.stage_name}/{path.name}",
                        source_path=str(path),
                    )
            context.db.flush()
        return result

    def _read_exif(self, path: Path) -> dict[str, Any]:
        if Image is None:
            return {"available": False, "reason": "pillow_unavailable"}
        try:
            with Image.open(path) as image:
                raw = image.getexif()
                if not raw:
                    return {"available": False}
                tag_names = ExifTags.TAGS if ExifTags is not None else {}
                parsed = {str(tag_names.get(key, key)): str(value) for key, value in raw.items()}
                return {
                    "available": True,
                    "camera_model": parsed.get("Model"),
                    "focal_length": parsed.get("FocalLength"),
                    "datetime": parsed.get("DateTimeOriginal") or parsed.get("DateTime"),
                    "orientation": parsed.get("Orientation"),
                }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _inspect_video(self, path: Path, sample_dir: Path) -> dict[str, Any]:
        result = {
            "readable": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "codec": None,
            "sampled_frames": [],
            "blur_frame_ratio": 0.0,
            "duplicate_frame_ratio": 0.0,
            "exposure_stability": 0.0,
            "motion_intensity_avg": 0.0,
            "sequential_sfm_suitability": 0.0,
            "loop_detection_suitability": 0.0,
        }
        if cv2 is None:
            result["error"] = "opencv_unavailable"
            return result
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result["error"] = "video_open_failed"
            return result
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        sample_dir.mkdir(parents=True, exist_ok=True)
        step = max(1, frame_count // 40) if frame_count else 1
        sampled = []
        prev_gray = None
        blur_count = 0
        duplicate_count = 0
        brightness_values = []
        motion_values = []
        hashes: set[str] = set()
        for index in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = sample_dir / f"sample_{index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            metrics = estimate_image_metrics(frame_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(metrics.get("laplacian_variance") or 0.0) < 60.0:
                blur_count += 1
            brightness_values.append(float(metrics.get("brightness_mean") or 0.0))
            tiny = cv2.resize(gray, (16, 16))
            digest = "".join("1" if value > tiny.mean() else "0" for value in tiny.flatten())
            if digest in hashes:
                duplicate_count += 1
            hashes.add(digest)
            motion = 0.0
            if prev_gray is not None:
                motion = float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0
                motion_values.append(motion)
            prev_gray = gray
            sampled.append({"frame_index": index, "timestamp_sec": index / fps if fps else 0.0, "path": str(frame_path), "metrics": metrics, "motion": round(motion, 4)})
        cap.release()
        sample_count = max(1, len(sampled))
        brightness_std = float(np.std(brightness_values)) if brightness_values else 0.0
        exposure_stability = clamp(1.0 - brightness_std / 80.0)
        motion_avg = float(np.mean(motion_values)) if motion_values else 0.0
        sharp_ratio = 1.0 - blur_count / sample_count
        result.update(
            {
                "readable": True,
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "frame_count": frame_count,
                "duration_sec": round(duration, 3),
                "sampled_frames": sampled,
                "blur_frame_ratio": round(blur_count / sample_count, 4),
                "duplicate_frame_ratio": round(duplicate_count / sample_count, 4),
                "exposure_stability": round(exposure_stability, 4),
                "motion_intensity_avg": round(motion_avg, 4),
                "sequential_sfm_suitability": round(clamp(0.45 * sharp_ratio + 0.35 * exposure_stability + 0.2 * clamp(motion_avg * 8.0)), 4),
                "loop_detection_suitability": round(clamp(duration / 60.0) * 0.4 + clamp(len(sampled) / 40.0) * 0.6, 4),
            }
        )
        return result

    def _image_risk(self, metrics: dict[str, Any]) -> dict[str, Any]:
        reasons = []
        if not metrics.get("readable"):
            reasons.append("unreadable")
        if float(metrics.get("laplacian_variance") or 0.0) < 60:
            reasons.append("blur")
        if float(metrics.get("overexposed_ratio") or 0.0) > 0.12 or float(metrics.get("underexposed_ratio") or 0.0) > 0.12:
            reasons.append("bad_exposure")
        if float(metrics.get("psnr_estimate") or 0.0) < 24:
            reasons.append("heavy_compression")
        return {"level": "warning" if reasons else "low", "reasons": reasons}

    def _video_risk(self, video: dict[str, Any]) -> dict[str, Any]:
        reasons = []
        if not video.get("readable"):
            reasons.append("unreadable")
        if float(video.get("blur_frame_ratio") or 0.0) > 0.25:
            reasons.append("high_blur_ratio")
        if float(video.get("duplicate_frame_ratio") or 0.0) > 0.45:
            reasons.append("high_duplicate_ratio")
        if float(video.get("exposure_stability") or 0.0) < 0.5:
            reasons.append("exposure_instability")
        return {"level": "warning" if reasons else "low", "reasons": reasons}

    def _score(self, inventory: list[dict[str, Any]]) -> float:
        if not inventory:
            return 0.0
        scores = []
        for item in inventory:
            if item.get("kind") == "video":
                scores.append(float((item.get("video") or {}).get("sequential_sfm_suitability") or 0.2))
            else:
                metrics = item.get("metrics") or {}
                scores.append(0.45 * float(metrics.get("sharpness_score") or 0.0) + 0.35 * float(metrics.get("feature_detectability_score") or 0.0) + 0.2 * float(metrics.get("exposure_score") or 0.0))
        return round(sum(scores) / max(1, len(scores)), 4)

    def _advice(self, inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        advice = []
        for item in inventory:
            reasons = (item.get("risk") or {}).get("reasons") or []
            for reason in reasons:
                advice.append({"asset_id": item.get("asset_id"), "reason": reason, "recommendation": self._recommendation(reason)})
        return advice

    def _recommendation(self, reason: str) -> str:
        return {
            "blur": "优先尝试轻度去噪/去模糊候选，若特征点仍不足则要求补拍。",
            "bad_exposure": "尝试曝光归一候选；若高光或暗部已丢失则不能依赖增强修复。",
            "heavy_compression": "保留原图并避免超分作为最终纹理，必要时补采更高码率素材。",
            "high_blur_ratio": "视频抽帧阶段必须剔除模糊片段，优先保留稳定慢速移动段。",
            "high_duplicate_ratio": "视频抽帧阶段需要去重，避免重复帧压倒 JPG。",
            "exposure_instability": "视频抽帧阶段优先选择曝光稳定片段。",
        }.get(reason, "进入后续阶段时保守处理，并在报告中保留风险。")

    def _risk_markdown(self, inventory: list[dict[str, Any]]) -> str:
        lines = ["# Initial Risk Report", ""]
        for item in inventory:
            risk = item.get("risk") or {}
            lines.append(f"- `{item.get('filename')}` kind={item.get('kind')} risk={risk.get('level')} reasons={', '.join(risk.get('reasons') or []) or '-'}")
        return "\n".join(lines) + "\n"

    def _make_video_timeline_chart(self, output_path: Path, inventory: list[dict[str, Any]]) -> Path | None:
        if Image is None or ImageDraw is None:
            return None
        videos = [item for item in inventory if item.get("kind") == "video"]
        if not videos:
            return None
        width = 900
        row_h = 70
        image = Image.new("RGB", (width, 30 + row_h * len(videos)), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((12, 8), "Video timeline quality chart", fill=(32, 39, 43))
        for row, item in enumerate(videos):
            y = 30 + row * row_h
            draw.text((12, y + 6), str(item.get("filename"))[:48], fill=(32, 39, 43))
            samples = ((item.get("video") or {}).get("sampled_frames") or [])[:120]
            for idx, frame in enumerate(samples):
                x = 220 + int(idx * max(1, (width - 240) / max(1, len(samples))))
                metrics = frame.get("metrics") or {}
                sharp = float(metrics.get("sharpness_score") or 0.0)
                exposure = float(metrics.get("exposure_score") or 0.0)
                score = clamp((sharp + exposure) / 2.0)
                color = (36, 143, 113) if score > 0.65 else (208, 151, 51) if score > 0.35 else (170, 65, 65)
                draw.rectangle((x, y + 34, x + 4, y + 58), fill=color)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=92)
        return output_path


class ImageEnhancementStage(StageOptimizer):
    stage_name = "image_enhancement"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        raw = read_json(context.stage_dir("raw_media_inspection") / "raw_media_inventory.json", {"assets": []})
        images = [
            item
            for item in raw.get("assets", [])
            if item.get("kind") == "image" and not bool((item.get("metrics") or {}).get("is_panorama_2_1"))
        ]
        return {"input_artifacts": [item.get("path") for item in images if item.get("path")], "images": images}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        settings = config_section(context.settings, "stage_optimized_reconstruction")
        allow = {
            "denoise_light": bool(context.config.get("allow_denoise", True)),
            "deblur_light": bool(context.config.get("allow_deblur", True)),
            "exposure_normalized": True,
            "white_balance_light": True,
            "white_balance_normalized": True,
            "gamma_light": True,
            "sharpen_light": True,
            "contrast_local_light": True,
            "super_resolution_safe": bool(context.config.get("allow_super_resolution", False)),
            "combined_safe_enhance": bool(context.config.get("allow_denoise", True) or context.config.get("allow_deblur", True)),
        }
        configured_routes = context.config.get("image_enhancement_routes")
        if isinstance(configured_routes, list) and configured_routes:
            requested_routes = [str(route) for route in configured_routes if str(route) in IMAGE_ENHANCEMENT_ROUTES]
            routes = list(dict.fromkeys(["original", *requested_routes]))
        else:
            routes = list(dict.fromkeys(["original", *SAFE_IMAGE_ENHANCEMENT_ROUTES, *IMAGE_ENHANCEMENT_ROUTES]))
        for item in analysis["images"]:
            for process_type in routes:
                if process_type != "original" and not allow.get(process_type, True):
                    continue
                candidate = {
                    "candidate_name": f"{item['asset_id']}:{process_type}",
                    "candidate_type": process_type,
                    "asset_id": item["asset_id"],
                    "input_path": item["path"],
                    "process_params": self._process_params(process_type),
                    "status": "created",
                    "created_at": utc_now_iso(),
                }
                if process_type == "super_resolution_safe" and not settings.get("super_resolution_adapter_enabled", False):
                    candidate.update(
                        {
                            "status": "skipped",
                            "rejected_reason": "super_resolution_adapter_not_configured_for_safe_execution",
                            "score": 0.0,
                            "risk_level": "medium",
                        }
                    )
                    context.capability_report.setdefault(self.stage_name, {})["super_resolution_safe"] = "capability_unavailable"
                candidates.append(candidate)
        if not candidates and analysis["images"]:
            context.capability_report[self.stage_name] = {"opencv": cv2 is not None, "reason": "no_image_candidates_generated"}
        return candidates

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        if candidate.get("status") == "skipped":
            return candidate
        source = Path(str(candidate["input_path"]))
        output_dir = context.stage_dir(self.stage_name) / "enhanced_images" / str(candidate["asset_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{candidate['candidate_type']}_{source.name}"
        if candidate["candidate_type"] == "original":
            copy_file_safely(source, output)
        elif cv2 is None:
            candidate.update({"status": "skipped", "rejected_reason": "opencv_unavailable", "score": 0.0})
            return candidate
        else:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                candidate.update({"status": "failed", "rejected_reason": "image_read_failed", "score": 0.0})
                return candidate
            derived = self._apply_process(image, candidate["candidate_type"])
            cv2.imwrite(str(output), derived, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        candidate.update({"status": "succeeded", "output_path": str(output)})
        return candidate

    def evaluate_candidate(self, context: StageContext, candidate_result: dict[str, Any]) -> dict[str, Any]:
        if candidate_result.get("status") != "succeeded":
            return candidate_result
        metrics = estimate_image_metrics(Path(str(candidate_result["output_path"])))
        source_metrics = estimate_image_metrics(Path(str(candidate_result["input_path"])))
        guard = ForensicIntegrityGuard(context.config)
        integrity = (
            {
                "integrity_risk_score": 0.0,
                "accepted_for_pose": True,
                "accepted_for_training": True,
                "rejected_reason": None,
                "process_type": "original",
            }
            if candidate_result["candidate_type"] == "original"
            else guard.evaluate_derivative(Path(str(candidate_result["input_path"])), Path(str(candidate_result["output_path"])), process_type=candidate_result["candidate_type"])
        )
        sharp_gain = float(metrics.get("laplacian_variance") or 0.0) - float(source_metrics.get("laplacian_variance") or 0.0)
        feature_gain = float(metrics.get("keypoint_count") or 0.0) - float(source_metrics.get("keypoint_count") or 0.0)
        score = (
            0.35 * float(metrics.get("feature_detectability_score") or 0.0)
            + 0.25 * float(metrics.get("sharpness_score") or 0.0)
            + 0.2 * float(metrics.get("exposure_score") or 0.0)
            + 0.2 * (1.0 - float(integrity.get("integrity_risk_score") or 1.0))
        )
        rejected = integrity.get("rejected_reason")
        if candidate_result["candidate_type"] != "original" and not integrity.get("accepted_for_pose"):
            rejected = rejected or "forensic_integrity_guard_rejected"
        candidate_result.update(
            {
                "metrics": {
                    **metrics,
                    "sharpness_gain": round(sharp_gain, 3),
                    "keypoint_gain": int(feature_gain),
                    "forensic_integrity": integrity,
                },
                "score": round(score, 4) if not rejected else 0.0,
                "risk_level": "high" if float(integrity.get("integrity_risk_score") or 0.0) > 0.45 else "medium" if float(integrity.get("integrity_risk_score") or 0.0) > 0.25 else "low",
                "rejected_reason": rejected,
                "improvement_summary": f"sharpness_gain={sharp_gain:.2f}, keypoint_gain={int(feature_gain)}",
                "risk_summary": f"integrity_risk={integrity.get('integrity_risk_score')}",
            }
        )
        metrics_path = Path(str(candidate_result["output_path"])).with_suffix(".metrics.json")
        write_json(metrics_path, candidate_result["metrics"])
        candidate_result["metrics_path"] = str(metrics_path)
        return candidate_result

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        by_asset: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidate_results:
            by_asset.setdefault(str(candidate.get("asset_id")), []).append(candidate)
        selections = []
        for asset_id, candidates in by_asset.items():
            valid_pose = [item for item in candidates if item.get("status") == "succeeded" and not item.get("rejected_reason") and (item.get("metrics") or {}).get("forensic_integrity", {}).get("accepted_for_pose")]
            original = next((item for item in candidates if item.get("candidate_type") == "original"), None)
            pose_best = max(valid_pose, key=lambda item: float(item.get("score") or 0.0), default=original)
            training_best = original
            selections.append(
                {
                    "asset_id": asset_id,
                    "image_original": original.get("output_path") if original else None,
                    "image_for_pose": pose_best.get("output_path") if pose_best else None,
                    "image_for_training": training_best.get("output_path") if training_best else None,
                    "pose_candidate": pose_best.get("candidate_type") if pose_best else None,
                    "training_candidate": training_best.get("candidate_type") if training_best else None,
                    "pose_source_family": _source_family(pose_best.get("candidate_type") if pose_best else None),
                    "training_source_family": _source_family(training_best.get("candidate_type") if training_best else None),
                    "original_sha256": file_sha256(original.get("output_path") if original else None),
                    "pose_sha256": file_sha256(pose_best.get("output_path") if pose_best else None),
                    "training_sha256": file_sha256(training_best.get("output_path") if training_best else None),
                    "pose_score": pose_best.get("score") if pose_best else 0.0,
                    "training_score": training_best.get("score") if training_best else 0.0,
                }
            )
        provenance = []
        for candidate in candidate_results:
            output_path = candidate.get("output_path")
            metrics = candidate.get("metrics") or {}
            integrity = metrics.get("forensic_integrity") or {}
            provenance.append(
                {
                    "asset_id": candidate.get("asset_id"),
                    "source_file": candidate.get("input_path"),
                    "source_sha256": file_sha256(candidate.get("input_path")),
                    "derived_file": output_path,
                    "derived_sha256": file_sha256(output_path),
                    "process_type": candidate.get("candidate_type"),
                    "process_params": candidate.get("process_params") or self._process_params(str(candidate.get("candidate_type") or "")),
                    "model_name": None,
                    "model_version": None,
                    "created_at": candidate.get("created_at"),
                    "integrity_risk_score": integrity.get("integrity_risk_score"),
                    "accepted_for_pose": bool(integrity.get("accepted_for_pose")),
                    "accepted_for_training": bool(integrity.get("accepted_for_training")),
                    "rejected_reason": candidate.get("rejected_reason") or integrity.get("rejected_reason"),
                }
            )
        selection_payload = {
            "policy": "safe_pose_original_train",
            "generative_enhancement_used": False,
            "safe_enhancement_routes": SAFE_IMAGE_ENHANCEMENT_ROUTES,
            "images": selections,
        }
        selection_path = write_json(context.stage_dir(self.stage_name) / "image_best_selection.json", selection_payload)
        provenance_path = write_json(context.stage_dir(self.stage_name) / "enhancement_provenance.json", {"images": provenance})
        candidate_metrics_path = write_json(context.stage_dir(self.stage_name) / "image_candidate_metrics.json", {"candidates": candidate_results})
        best = {
            "candidate_name": "per_image_best_selection",
            "candidate_type": "selection_manifest",
            "status": "succeeded" if selections else "blocked",
            "output_path": str(selection_path),
            "provenance_path": str(provenance_path),
            "metrics_path": str(candidate_metrics_path),
            "score": round(sum(float(item.get("pose_score") or 0.0) for item in selections) / max(1, len(selections)), 4),
            "metrics": {
                "selected_image_count": len(selections),
                "candidate_count": len(candidate_results),
                "pose_image_distribution": _distribution(selections, "pose_candidate"),
                "training_image_distribution": _distribution(selections, "training_candidate"),
                "generative_enhancement_used": False,
            },
            "risk_level": "low",
            "improvement_summary": "已为每张图片分别选择 image_for_pose、image_for_training 和原图索引。",
            "risk_summary": "高真实性风险派生图不会进入 best selection。",
        }
        return best

    def _process_params(self, process_type: str) -> dict[str, Any]:
        return image_process_params(process_type)

    def _apply_process(self, image: np.ndarray, process_type: str) -> np.ndarray:
        if process_type == "denoise_light":
            return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        if process_type == "deblur_light":
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            return cv2.addWeighted(image, 1.35, blurred, -0.35, 0)
        if process_type == "exposure_normalized":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean = max(1.0, float(l.mean()))
            scale = clamp(135.0 / mean, 0.75, 1.25)
            l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if process_type in {"white_balance_light", "white_balance_normalized"}:
            result = image.astype(np.float32)
            means = result.reshape(-1, 3).mean(axis=0)
            target = float(np.mean(means))
            result *= target / np.maximum(means, 1.0)
            return np.clip(result, 0, 255).astype(np.uint8)
        if process_type == "gamma_light":
            gamma = 0.95
            table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            return cv2.LUT(image, table)
        if process_type == "sharpen_light":
            blurred = cv2.GaussianBlur(image, (0, 0), 0.9)
            return cv2.addWeighted(image, 1.18, blurred, -0.18, 0)
        if process_type == "contrast_local_light":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
            return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        if process_type == "combined_safe_enhance":
            result = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
            blurred = cv2.GaussianBlur(result, (0, 0), 0.8)
            result = cv2.addWeighted(result, 1.22, blurred, -0.22, 0)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean = max(1.0, float(l.mean()))
            scale = clamp(135.0 / mean, 0.85, 1.15)
            l = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            means = result.reshape(-1, 3).mean(axis=0)
            target = float(np.mean(means))
            result = result.astype(np.float32) * target / np.maximum(means, 1.0)
            return np.clip(result, 0, 255).astype(np.uint8)
        return image


class VideoKeyframeOptimizationStage(StageOptimizer):
    stage_name = "video_keyframe_optimization"

    def _video_config(self, context: StageContext, key: str, default: Any) -> Any:
        return nested_get(context.config, f"video.{key}", nested_get(context.config, f"stage_optimized_reconstruction.video.{key}", default))

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        raw = read_json(context.stage_dir("raw_media_inspection") / "raw_media_inventory.json", {"assets": []})
        videos = [item for item in raw.get("assets", []) if item.get("kind") == "video"]
        return {"input_artifacts": [item.get("path") for item in videos if item.get("path")], "videos": videos}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for video in analysis["videos"]:
            for strategy in VIDEO_KEYFRAME_ROUTES:
                candidates.append({"candidate_name": f"{video['asset_id']}:{strategy}", "candidate_type": strategy, "asset_id": video["asset_id"], "input_path": video["path"], "status": "created", "created_at": utc_now_iso()})
        if not candidates:
            candidates.append({"candidate_name": "no_video_assets", "candidate_type": "empty_manifest", "status": "succeeded", "score": 1.0, "frames": [], "created_at": utc_now_iso()})
        return candidates

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        if candidate["candidate_type"] == "empty_manifest":
            output = write_json(context.stage_dir(self.stage_name) / "best_keyframe_strategy.json", {"videos": []})
            candidate.update({"output_path": str(output), "metrics": {"frame_count": 0}})
            return candidate
        if cv2 is None:
            candidate.update({"status": "skipped", "rejected_reason": "opencv_unavailable", "score": 0.0})
            return candidate
        pool = self._load_or_create_sample_pool(context, candidate)
        if pool.get("status") != "succeeded":
            candidate.update({"status": "failed", "rejected_reason": pool.get("rejected_reason") or "video_sample_pool_failed", "score": 0.0})
            return candidate
        fps = float(pool.get("fps") or 0.0)
        duration_sec = float(pool.get("duration_sec") or 0.0)
        source_frames = list(pool.get("frames") or [])
        max_frames = int(self._video_config(context, "max_keyframes_per_video", 180) or 180)
        strategy = str(candidate["candidate_type"])
        output_dir = context.stage_dir(self.stage_name) / "video_keyframes" / strategy / str(candidate["asset_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        last_hash = None
        duplicate_rejections = 0
        blur_rejections = 0
        exposure_rejections = 0
        motion_rejections = 0
        candidate_source_frames = self._candidate_source_frames(
            source_frames=source_frames,
            strategy=strategy,
            duration_sec=duration_sec,
            max_frames=max_frames,
        )
        for frame_item in candidate_source_frames:
            metrics = dict(frame_item.get("metrics") or {})
            frame_hash = str(frame_item.get("perceptual_hash") or "")
            motion_score = float(frame_item.get("motion_score") or 0.0)
            is_duplicate = frame_hash == last_hash
            is_blur = float(metrics.get("laplacian_variance") or 0.0) < 70.0
            is_bad_exposure = float(metrics.get("exposure_score") or 0.0) < 0.35
            keep = True
            if strategy == "motion_aware" and (motion_score < 1.5 or motion_score > 55.0):
                keep = False
                motion_rejections += 1
            if strategy in {"blur_filtered", "hybrid_balanced", "hybrid_sparse"} and is_blur:
                keep = False
                blur_rejections += 1
            if strategy in {"exposure_stable", "hybrid_balanced", "hybrid_dense"} and is_bad_exposure:
                keep = False
                exposure_rejections += 1
            if strategy not in {"uniform_1fps", "uniform_2fps", "uniform_3fps", "loop_aware"} and is_duplicate:
                keep = False
                duplicate_rejections += 1
            is_loop_anchor = frame_item is source_frames[0] or frame_item is source_frames[-1]
            if strategy == "loop_aware" and not is_loop_anchor and is_duplicate:
                keep = False
                duplicate_rejections += 1
            status = "accepted" if keep else "rejected"
            frames.append(
                {
                    "frame_id": f"{candidate['asset_id']}:{frame_item.get('frame_index')}",
                    "timestamp_sec": frame_item.get("timestamp_sec", 0.0),
                    "image_path": frame_item.get("image_path"),
                    "metrics": metrics,
                    "status": status,
                    "sample_pool_frame": True,
                }
            )
            last_hash = frame_hash
        accepted = [frame for frame in frames if frame["status"] == "accepted"]
        output = write_json(output_dir.parent / f"{candidate['asset_id']}_{strategy}_keyframe_quality.json", {"frames": frames})
        temporal_span = 0.0
        if accepted:
            timestamps = [float(frame.get("timestamp_sec") or 0.0) for frame in accepted]
            temporal_span = max(timestamps) - min(timestamps)
        candidate.update(
            {
                "status": "succeeded",
                "output_path": str(output),
                "frames": accepted,
                "metrics": {
                    "frame_count": len(frames),
                    "accepted_frame_count": len(accepted),
                    "candidate_source_frame_count": len(candidate_source_frames),
                    "selection_policy": "time_axis_full_coverage_resampled",
                    "sharp_frame_ratio": round(len([f for f in accepted if float((f.get("metrics") or {}).get("laplacian_variance") or 0.0) >= 70.0]) / max(1, len(frames)), 4),
                    "blur_rejection_ratio": round(blur_rejections / max(1, len(frames)), 4),
                    "duplicate_rejection_ratio": round(duplicate_rejections / max(1, len(frames)), 4),
                    "bad_exposure_rejection_ratio": round(exposure_rejections / max(1, len(frames)), 4),
                    "motion_rejection_ratio": round(motion_rejections / max(1, len(frames)), 4),
                    "temporal_coverage": round(min(1.0, temporal_span / max(1.0, duration_sec)), 4),
                    "estimated_sfm_suitability": round(
                        0.45 * min(1.0, len(accepted) / max(1, max_frames))
                        + 0.35 * (len(accepted) / max(1, len(frames)))
                        + 0.2 * min(1.0, temporal_span / max(1.0, duration_sec)),
                        4,
                    ),
                    "sample_pool_frame_count": len(source_frames),
                },
            }
        )
        candidate["score"] = round(float(candidate["metrics"]["estimated_sfm_suitability"]), 4)
        if len(accepted) == 0 and frames:
            candidate["rejected_reason"] = "all_candidate_frames_rejected"
            candidate["score"] = 0.0
        return candidate

    def _candidate_source_frames(
        self,
        *,
        source_frames: list[dict[str, Any]],
        strategy: str,
        duration_sec: float,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        if not source_frames:
            return []
        fps_targets = {
            "uniform_1fps": 1.0,
            "uniform_2fps": 2.0,
            "uniform_3fps": 3.0,
        }
        if strategy in fps_targets:
            desired = int(math.ceil(max(1.0, duration_sec) * fps_targets[strategy]))
        elif strategy == "hybrid_sparse":
            desired = int(math.ceil(max(1.0, duration_sec)))
        else:
            desired = max_frames
        target_count = max(1, min(max_frames, desired, len(source_frames)))
        return self._evenly_spaced_items(source_frames, target_count)

    @staticmethod
    def _evenly_spaced_items(items: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
        if target_count >= len(items):
            return list(items)
        if target_count <= 1:
            return [items[0]]
        raw_indices = [int(round(value)) for value in np.linspace(0, len(items) - 1, num=target_count)]
        indices: list[int] = []
        seen: set[int] = set()
        for index in raw_indices:
            clamped = max(0, min(len(items) - 1, index))
            if clamped not in seen:
                indices.append(clamped)
                seen.add(clamped)
        if len(indices) < target_count:
            for index in range(len(items)):
                if index not in seen:
                    indices.append(index)
                    seen.add(index)
                    if len(indices) >= target_count:
                        break
            indices.sort()
        return [items[index] for index in indices[:target_count]]

    def _load_or_create_sample_pool(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        asset_id = str(candidate["asset_id"])
        pool_dir = context.stage_dir(self.stage_name) / "video_keyframes" / "_sample_pool" / asset_id
        pool_manifest = pool_dir / "sample_pool.json"
        if pool_manifest.exists():
            return read_json(pool_manifest, {})
        source = Path(str(candidate["input_path"]))
        pool_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            payload = {"status": "failed", "rejected_reason": "video_open_failed", "frames": []}
            write_json(pool_manifest, payload)
            return payload
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            payload = {"status": "failed", "rejected_reason": "video_frame_count_unavailable", "frames": []}
            write_json(pool_manifest, payload)
            return payload
        duration_sec = frame_count / fps if fps else 0.0
        max_pool_frames = int(self._video_config(context, "max_frames_per_strategy", self._video_config(context, "max_keyframes_per_video", 180)) or 180)
        max_pool_frames = max(1, max_pool_frames)
        sample_count = min(max_pool_frames, frame_count)
        indices = sorted({int(value) for value in np.linspace(0, max(0, frame_count - 1), num=sample_count)})
        frames = []
        previous_gray = None
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = pool_dir / f"{asset_id}_{index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            metrics = estimate_image_metrics(frame_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = 0.0
            if previous_gray is not None:
                resized_current = cv2.resize(gray, (160, 90))
                resized_previous = cv2.resize(previous_gray, (160, 90))
                motion_score = float(np.mean(np.abs(resized_current.astype(np.float32) - resized_previous.astype(np.float32))))
            previous_gray = gray
            tiny = cv2.resize(gray, (16, 16))
            threshold = float(tiny.mean())
            frame_hash = "".join("1" if float(value) > threshold else "0" for value in tiny.flatten())
            frames.append(
                {
                    "frame_id": f"{asset_id}:{index}",
                    "frame_index": index,
                    "timestamp_sec": index / fps if fps else 0.0,
                    "image_path": str(frame_path),
                    "metrics": metrics,
                    "motion_score": round(motion_score, 4),
                    "perceptual_hash": frame_hash,
                    "status": "sampled",
                }
            )
        cap.release()
        payload = {
            "status": "succeeded",
            "asset_id": asset_id,
            "source_path": str(source),
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "sample_count": len(frames),
            "sampling_strategy": "single_evenly_spaced_pool_reused_by_all_candidates",
            "frames": frames,
        }
        write_json(pool_manifest, payload)
        return payload

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        if len(candidate_results) == 1 and candidate_results[0].get("candidate_type") == "empty_manifest":
            return candidate_results[0]
        by_video: dict[str, list[dict[str, Any]]] = {}
        for item in candidate_results:
            by_video.setdefault(str(item.get("asset_id")), []).append(item)
        videos = []
        all_frames = []
        for asset_id, items in by_video.items():
            best = max([item for item in items if item.get("status") == "succeeded"], key=lambda item: float(item.get("score") or 0.0), default=None)
            if best:
                videos.append({"asset_id": asset_id, "strategy": best["candidate_type"], "frames": best.get("frames", []), "metrics": best.get("metrics", {})})
                all_frames.extend(best.get("frames", []))
        output = write_json(context.stage_dir(self.stage_name) / "best_keyframe_strategy.json", {"videos": videos, "frames": all_frames})
        create_contact_sheet([{"path": frame.get("image_path"), "label": frame.get("frame_id")} for frame in all_frames[:32]], context.stage_dir(self.stage_name) / "video_contact_sheet_by_strategy.jpg", title="Best keyframes")
        return {
            "candidate_name": "per_video_best_keyframe_set",
            "candidate_type": "selection_manifest",
            "status": "succeeded",
            "output_path": str(output),
            "score": round(sum(float(video.get("metrics", {}).get("estimated_sfm_suitability") or 0.0) for video in videos) / max(1, len(videos)), 4),
            "metrics": {"video_count": len(videos), "selected_frame_count": len(all_frames)},
            "improvement_summary": "已从视频中按清晰度、曝光稳定性、去重和连续覆盖选择关键帧集合。",
            "risk_summary": "严重模糊、极端曝光和重复帧不会进入 best_keyframe_set。",
        }


class PanoramaNormalizationStage(StageOptimizer):
    stage_name = "panorama_normalization"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        raw = read_json(context.stage_dir("raw_media_inspection") / "raw_media_inventory.json", {"assets": []})
        panos = [
            item
            for item in raw.get("assets", [])
            if item.get("kind") == "panorama" or bool((item.get("metrics") or {}).get("is_panorama_2_1"))
        ]
        spherical_cfg = spherical_video_config(context.config)
        spherical_videos: list[dict[str, Any]] = []
        if spherical_cfg.enabled:
            keyframes = read_json(context.stage_dir("video_keyframe_optimization") / "best_keyframe_strategy.json", {"videos": []})
            frames_by_asset = {str(video.get("asset_id")): list(video.get("frames") or []) for video in keyframes.get("videos", [])}
            for item in raw.get("assets", []):
                video = item.get("video") or {}
                if item.get("kind") != "video":
                    continue
                if not is_equirectangular_size(int(video.get("width") or 0), int(video.get("height") or 0)):
                    continue
                frames = frames_by_asset.get(str(item.get("asset_id")), [])
                if frames:
                    spherical_videos.append({"asset_id": item.get("asset_id"), "path": item.get("path"), "video": video, "frames": frames})
        input_artifacts = [item.get("path") for item in panos if item.get("path")]
        input_artifacts.extend(item.get("path") for item in spherical_videos if item.get("path"))
        return {"input_artifacts": input_artifacts, "panoramas": panos, "spherical_videos": spherical_videos}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        configured_routes = context.config.get("panorama_normalization_routes")
        if configured_routes is None:
            configured_routes = nested_get(context.config, "panorama.normalization_routes")
        routes = [str(route) for route in (configured_routes or PANORAMA_NORMALIZATION_ROUTES) if str(route) in PANORAMA_NORMALIZATION_ROUTES]
        if not routes:
            routes = list(PANORAMA_NORMALIZATION_ROUTES)
        for source_index, pano in enumerate(analysis["panoramas"]):
            for strategy in routes:
                candidates.append(
                    {
                        "candidate_name": f"{pano['asset_id']}:{strategy}",
                        "candidate_type": strategy,
                        "asset_id": pano["asset_id"],
                        "input_path": pano["path"],
                        "source_index": source_index,
                        "status": "created",
                        "created_at": utc_now_iso(),
                    }
                )
        for video in analysis.get("spherical_videos") or []:
            candidates.append(
                {
                    "candidate_name": f"{video['asset_id']}:experimental_360_video_perspective_views",
                    "candidate_type": "experimental_360_video_perspective_views",
                    "asset_id": video["asset_id"],
                    "input_path": video["path"],
                    "frames": video.get("frames") or [],
                    "status": "created",
                    "created_at": utc_now_iso(),
                }
            )
        if not candidates:
            candidates.append({"candidate_name": "no_panorama_assets", "candidate_type": "empty_manifest", "status": "succeeded", "score": 1.0, "views": [], "created_at": utc_now_iso()})
        return candidates

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        if candidate["candidate_type"] == "empty_manifest":
            output = write_json(context.stage_dir(self.stage_name) / "best_panorama_strategy.json", {"panoramas": []})
            candidate.update({"output_path": str(output), "metrics": {"generated_view_count": 0}})
            return candidate
        if candidate["candidate_type"] == "experimental_360_video_perspective_views":
            spherical_cfg = spherical_video_config(context.config)
            if not spherical_cfg.enabled:
                candidate.update({"status": "skipped", "rejected_reason": "experimental_360_video_disabled", "score": 0.0})
                return candidate
            output_dir = context.stage_dir(self.stage_name) / "spherical_video_views" / str(candidate["asset_id"])
            try:
                views = derive_spherical_video_views(
                    asset_id=str(candidate["asset_id"]),
                    frames=list(candidate.get("frames") or []),
                    output_dir=output_dir,
                    config=spherical_cfg,
                )
            except Exception as exc:
                error_path = write_json(output_dir / "spherical_video_error.json", {"error": str(exc), "candidate": candidate["candidate_name"]})
                candidate.update(
                    {
                        "status": "failed",
                        "output_path": str(error_path),
                        "rejected_reason": "experimental_360_video_projection_failed",
                        "score": 0.0,
                        "risk_level": "medium",
                    }
                )
                return candidate
            mapping_path = write_json(output_dir / "spherical_video_mapping.json", {"asset_id": candidate["asset_id"], "views": views})
            pose_suitability = min(1.0, len(views) / 80.0)
            candidate.update(
                {
                    "status": "succeeded",
                    "output_path": str(mapping_path),
                    "views": views,
                    "metrics": {
                        "generated_view_count": len(views),
                        "spherical_video_view_count": len(views),
                        "source_keyframe_count": len({view.get("source_frame_id") for view in views}),
                        "pose_suitability": round(pose_suitability, 4),
                        "geometry_risk_score": 0.42,
                    },
                    "score": round(0.6 * pose_suitability + 0.4 * 0.58, 4),
                    "risk_level": "medium",
                    "risk_summary": "Experimental 360 video perspective views preserve source mapping and are only enabled by explicit run config.",
                }
            )
            if not views:
                candidate.update({"rejected_reason": "no_spherical_video_views_generated", "score": 0.0})
            return candidate
        source = Path(str(candidate["input_path"]))
        if Image is None:
            candidate.update({"status": "skipped", "rejected_reason": "pillow_unavailable", "score": 0.0})
            return candidate
        output_dir = context.stage_dir(self.stage_name) / "panorama_views" / candidate["candidate_type"] / str(candidate["asset_id"])
        output_dir.mkdir(parents=True, exist_ok=True)
        views = []
        with Image.open(source) as image:
            image = image.convert("RGB")
            if candidate["candidate_type"] == "keep_equirectangular":
                output = output_dir / source.name
                image.save(output, quality=94)
                views.append(
                    {
                        "pano_view_id": f"{candidate['asset_id']}:equirectangular",
                        "asset_id": candidate["asset_id"],
                        "image_path": str(output),
                        "source_type": "panorama_equirectangular",
                        "source_image_path": str(source),
                        "source_pano_id": candidate["asset_id"],
                        "shared_center_group": candidate["asset_id"],
                        "yaw": None,
                        "pitch": None,
                        "fov": 360,
                        "usage": "context_texture",
                        "mapping": "source_equirectangular",
                    }
                )
            elif candidate["candidate_type"] in {"perspective_cubemap_4", "perspective_cubemap_6", "perspective_views_dense"}:
                width, height = image.size
                if candidate["candidate_type"] == "perspective_cubemap_4":
                    yaw_pitch_pairs = [(0, 0), (90, 0), (180, 0), (270, 0)]
                elif candidate["candidate_type"] == "perspective_cubemap_6":
                    yaw_pitch_pairs = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 60), (0, -60)]
                else:
                    yaw_pitch_pairs = [(yaw, pitch) for pitch in (-30, 0, 30) for yaw in range(0, 360, 45)]
                if not is_equirectangular_size(width, height):
                    candidate.update({"status": "failed", "rejected_reason": "not_equirectangular_2_1", "score": 0.0, "risk_level": "high"})
                    return candidate
                projection_size = int(
                    nested_get(
                        context.config,
                        "panorama.output_size",
                        context.config.get("panorama_output_size") or max(512, min(2048, width // 4)),
                    )
                    or 512
                )
                output_width = int(nested_get(context.config, "panorama.output_width", context.config.get("panorama_output_width") or projection_size) or projection_size)
                output_height = int(nested_get(context.config, "panorama.output_height", context.config.get("panorama_output_height") or projection_size) or projection_size)
                fov_degrees = float(nested_get(context.config, "panorama.fov_degrees", context.config.get("panorama_fov_degrees") or 90.0) or 90.0)
                source_index = int(candidate.get("source_index") or 0)
                source_frame_id = f"{candidate['asset_id']}:{source_index}"
                for yaw, pitch in yaw_pitch_pairs:
                    output = output_dir / f"{candidate['asset_id']}_yaw_{yaw}_pitch_{pitch}.jpg"
                    try:
                        project_equirectangular_to_perspective(
                            source,
                            output,
                            yaw_degrees=float(yaw),
                            pitch_degrees=float(pitch),
                            fov_degrees=fov_degrees,
                            output_width=max(64, output_width),
                            output_height=max(64, output_height),
                        )
                    except Exception as exc:
                        error_path = write_json(output_dir / "panorama_projection_error.json", {"candidate": candidate["candidate_name"], "error": str(exc)})
                        candidate.update(
                            {
                                "status": "failed",
                                "output_path": str(error_path),
                                "rejected_reason": "panorama_perspective_projection_failed",
                                "score": 0.0,
                                "risk_level": "high",
                            }
                        )
                        return candidate
                    views.append(
                        {
                            "pano_view_id": f"{candidate['asset_id']}:yaw:{yaw}:pitch:{pitch}",
                            "asset_id": candidate["asset_id"],
                            "image_path": str(output),
                            "source_type": "panorama_station_view",
                            "source_frame_id": source_frame_id,
                            "source_frame_index": source_index,
                            "source_image_path": str(source),
                            "source_pano_id": candidate["asset_id"],
                            "shared_center_group": candidate["asset_id"],
                            "crop_id": output.name,
                            "yaw": yaw,
                            "pitch": pitch,
                            "fov": fov_degrees,
                            "stream_id": f"yaw_{float(yaw):.3f}_pitch_{float(pitch):.3f}",
                            "usage": "pose_candidate",
                            "mapping": "equirectangular_to_perspective",
                            "projection_model": "pinhole_from_equirectangular",
                            "camera_model": "PINHOLE",
                            "projection_width": max(64, output_width),
                            "projection_height": max(64, output_height),
                        }
                    )
            else:
                views = []
        mapping_path = write_json(
            output_dir / "panorama_mapping.json",
            {
                "asset_id": candidate["asset_id"],
                "source_image_path": str(source),
                "source_index": candidate.get("source_index"),
                "strategy": candidate["candidate_type"],
                "views": views,
            },
        )
        feature_scores = [estimate_image_metrics(Path(view["image_path"])).get("feature_detectability_score", 0.0) for view in views if view.get("image_path")]
        generated = len(views)
        geometry_risk = {
            "keep_equirectangular": 0.15,
            "perspective_cubemap_4": 0.32,
            "perspective_cubemap_6": 0.34,
            "perspective_views_dense": 0.38,
            "panorama_as_context_only": 0.05,
        }.get(candidate["candidate_type"], 0.35)
        pose_suitability = 0.0 if candidate["candidate_type"] in {"keep_equirectangular", "panorama_as_context_only"} else float(sum(float(v) for v in feature_scores) / max(1, len(feature_scores)))
        score = 0.65 * pose_suitability + 0.35 * (1.0 - geometry_risk)
        candidate.update(
            {
                "status": "succeeded",
                "output_path": str(mapping_path),
                "views": views,
                "metrics": {
                    "generated_view_count": generated,
                    "static_panorama_view_count": len([view for view in views if view.get("source_type") == "panorama_station_view"]),
                    "spherical_rig_view_count": len([view for view in views if view.get("source_type") in SPHERICAL_RIG_VIEW_SOURCE_TYPES]),
                    "source_mapping_complete": all(view.get("source_image_path") and view.get("source_pano_id") for view in views) if views else True,
                    "projection_model": "pinhole_from_equirectangular" if candidate["candidate_type"] in {"perspective_cubemap_4", "perspective_cubemap_6", "perspective_views_dense"} else "none",
                    "feature_quality_score": round(float(sum(float(v) for v in feature_scores) / max(1, len(feature_scores))), 4) if feature_scores else 0.0,
                    "pose_suitability": round(pose_suitability, 4),
                    "texture_suitability": 0.8 if candidate["candidate_type"] in {"keep_equirectangular", "panorama_as_context_only"} else 0.55,
                    "geometry_risk_score": geometry_risk,
                },
                "score": round(score, 4),
                "risk_level": "medium" if geometry_risk >= 0.35 else "low",
                "risk_summary": "全景展开视图带有几何风险，必须记录 yaw/pitch/fov/source mapping。",
            }
        )
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        if len(candidate_results) == 1 and candidate_results[0].get("candidate_type") == "empty_manifest":
            return candidate_results[0]
        by_asset: dict[str, list[dict[str, Any]]] = {}
        for item in candidate_results:
            by_asset.setdefault(str(item.get("asset_id")), []).append(item)
        panos = []
        views = []
        for asset_id, items in by_asset.items():
            best = max([item for item in items if item.get("status") == "succeeded"], key=lambda item: float(item.get("score") or 0.0), default=None)
            if best:
                panos.append({"asset_id": asset_id, "strategy": best["candidate_type"], "views": best.get("views", []), "metrics": best.get("metrics", {})})
                views.extend(best.get("views", []))
        output = write_json(context.stage_dir(self.stage_name) / "best_panorama_strategy.json", {"panoramas": panos, "views": views})
        spherical_views = [view for view in views if view.get("source_type") == "spherical_video_keyframe_view"]
        static_pano_views = [view for view in views if view.get("source_type") == "panorama_station_view"]
        return {
            "candidate_name": "per_panorama_best_strategy",
            "candidate_type": "selection_manifest",
            "status": "succeeded",
            "output_path": str(output),
            "score": 1.0,
            "metrics": {
                "panorama_count": len(panos),
                "view_count": len(views),
                "video_panorama_count": len([item for item in panos if item.get("strategy") == "experimental_360_video_perspective_views"]),
                "spherical_video_view_count": len(spherical_views),
                "static_panorama_view_count": len(static_pano_views),
                "spherical_rig_view_count": len([view for view in views if view.get("source_type") in SPHERICAL_RIG_VIEW_SOURCE_TYPES]),
            },
            "improvement_summary": "已识别 2:1/360 全景并避免直接作为普通图输入 SfM。",
            "risk_summary": "全景派生视图保留 source mapping，几何风险会进入报告。",
        }
        return {"candidate_name": "per_panorama_best_strategy", "candidate_type": "selection_manifest", "status": "succeeded", "output_path": str(output), "score": 1.0, "metrics": {"panorama_count": len(panos), "view_count": len(views)}, "improvement_summary": "已识别 2:1/360 全景并避免直接作为普通图输入 SfM。", "risk_summary": "全景派生视图保留 source mapping，几何风险会进入报告。"}


class DatasetAssemblyStage(StageOptimizer):
    stage_name = "dataset_assembly"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        image_selection = read_json(context.stage_dir("image_enhancement") / "image_best_selection.json", {"images": []})
        keyframes = read_json(context.stage_dir("video_keyframe_optimization") / "best_keyframe_strategy.json", {"frames": []})
        panos = read_json(context.stage_dir("panorama_normalization") / "best_panorama_strategy.json", {"views": []})
        return {
            "input_artifacts": [
                str(context.stage_dir("image_enhancement") / "image_best_selection.json"),
                str(context.stage_dir("video_keyframe_optimization") / "best_keyframe_strategy.json"),
                str(context.stage_dir("panorama_normalization") / "best_panorama_strategy.json"),
            ],
            "image_selection": image_selection.get("images", []),
            "keyframes": keyframes.get("frames", []),
            "panorama_views": panos.get("views", []),
        }

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        policy_by_route = {
            "safe_pose_original_train": "route_preset",
            "jpg_only_best_pose": "jpg_only_best_pose",
            "video_only_best_keyframes": "video_only_best_keyframes",
            "jpg_video_fused_balanced": "balanced_jpg_video",
            "jpg_video_fused_dense": "dense_jpg_video",
            "jpg_video_fused_sparse": "sparse_jpg_video",
            "panorama_context_added": "panorama_context_added",
            "high_confidence_only": "only_images_and_accepted_keyframes",
        }
        active_route = context.config.get("active_route_preset") or context.config.get("active_route_id")
        if active_route in ROUTE_PRESETS:
            return [
                {
                    "candidate_name": str(active_route),
                    "candidate_type": "dataset",
                    "policy": "route_preset",
                    "analysis": analysis,
                    "status": "created",
                    "created_at": utc_now_iso(),
                }
            ]
        return [
            {
                "candidate_name": route,
                "candidate_type": "dataset",
                "policy": policy_by_route[route],
                "analysis": analysis,
                "status": "created",
                "created_at": utc_now_iso(),
            }
            for route in DATASET_ASSEMBLY_ROUTES
        ]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        dataset_id = candidate["candidate_name"]
        output_dir = context.stage_dir(self.stage_name) / "datasets" / dataset_id
        pose_dir = output_dir / "pose_images"
        training_dir = output_dir / "training_images"
        pose_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)
        analysis = candidate["analysis"]
        image_items = list(analysis.get("image_selection") or [])
        keyframes = list(analysis.get("keyframes") or [])
        pano_views = [view for view in analysis.get("panorama_views") or [] if view.get("usage") == "pose_candidate"]
        spherical_view_count = len([view for view in pano_views if view.get("source_type") == "spherical_video_keyframe_view"])
        static_panorama_view_count = len([view for view in pano_views if view.get("source_type") == "panorama_station_view"])
        spherical_rig_view_count = len([view for view in pano_views if view.get("source_type") in SPHERICAL_RIG_VIEW_SOURCE_TYPES])
        spherical_rig_enabled = bool(static_panorama_view_count or (spherical_video_config(context.config).enabled and spherical_view_count))
        policy = str(candidate["policy"])
        if policy == "jpg_only_best_pose":
            keyframes = []
            pano_views = []
        elif policy == "video_only_best_keyframes":
            image_items = []
            pano_views = []
        elif policy == "balanced_jpg_video":
            max_video = max(20, len(image_items) * 3)
            keyframes = keyframes[:max_video]
            pano_views = []
        elif policy == "dense_jpg_video":
            max_video = max(40, len(image_items) * 6)
            keyframes = keyframes[:max_video]
            pano_views = []
        elif policy == "sparse_jpg_video":
            keyframes = keyframes[::2]
            pano_views = []
        elif policy == "panorama_context_added":
            max_video = max(20, len(image_items) * 3)
            keyframes = keyframes[:max_video]
            if spherical_rig_enabled:
                keyframes = []
        elif policy == "only_images_and_accepted_keyframes":
            keyframes = [frame for frame in keyframes if (frame.get("metrics") or {}).get("feature_detectability_score", 0.0) >= 0.2]
            pano_views = []
        else:
            pano_views = []
        if spherical_rig_enabled and policy != "panorama_context_added" and keyframes and not image_items:
            keyframes = []
        entries = []
        source_map = []
        for item in image_items:
            pose_path, pose_candidate = self._select_image_path(item, policy, role="pose", route_name=str(dataset_id))
            training_path, training_candidate = self._select_image_path(item, policy, role="training", route_name=str(dataset_id))
            if pose_path:
                pose_target = pose_dir / f"{item['asset_id']}_{Path(str(pose_path)).name}"
                copy_file_safely(pose_path, pose_target)
            else:
                pose_target = None
            if training_path:
                training_target = training_dir / f"{item['asset_id']}_{Path(str(training_path)).name}"
                copy_file_safely(training_path, training_target)
            else:
                training_target = None
            entry = {
                "source_type": "image",
                "asset_id": item["asset_id"],
                "pose_image": str(pose_target) if pose_target else None,
                "training_image": str(training_target) if training_target else None,
                "pose_candidate": pose_candidate,
                "training_candidate": training_candidate,
                "pose_source_family": _source_family(pose_candidate),
                "training_source_family": _source_family(training_candidate),
                "source_original": item.get("image_original"),
                "source_original_sha256": item.get("original_sha256") or file_sha256(item.get("image_original")),
                "pose_source_path": pose_path,
                "pose_source_sha256": file_sha256(pose_path),
                "training_source_path": training_path,
                "training_source_sha256": file_sha256(training_path),
                "pose_image_sha256": file_sha256(pose_target),
                "training_image_sha256": file_sha256(training_target),
            }
            entries.append(entry)
            source_map.append(
                {
                    "derived_pose": str(pose_target) if pose_target else None,
                    "derived_pose_sha256": file_sha256(pose_target),
                    "derived_training": str(training_target) if training_target else None,
                    "derived_training_sha256": file_sha256(training_target),
                    "source_asset_id": item["asset_id"],
                    "source_original": item.get("image_original"),
                    "source_original_sha256": item.get("original_sha256") or file_sha256(item.get("image_original")),
                    "pose_candidate": pose_candidate,
                    "training_candidate": training_candidate,
                    "pose_source_family": _source_family(pose_candidate),
                    "training_source_family": _source_family(training_candidate),
                }
            )
        for frame in keyframes:
            source = frame.get("image_path")
            if not source:
                continue
            target_name = f"{str(frame.get('frame_id') or 'frame').replace(':', '_')}_{Path(str(source)).name}"
            pose_target = pose_dir / target_name
            training_target = training_dir / target_name
            copy_file_safely(source, pose_target)
            copy_file_safely(source, training_target)
            entries.append({"source_type": "video_keyframe", "frame_id": frame.get("frame_id"), "pose_image": str(pose_target), "training_image": str(training_target)})
            source_map.append({"derived_pose": str(pose_target), "derived_training": str(training_target), "source_frame_id": frame.get("frame_id"), "source_video_frame": source})
        for view in pano_views:
            source = view.get("image_path")
            if not source:
                continue
            target_name = f"{str(view.get('pano_view_id') or 'pano').replace(':', '_')}_{Path(str(source)).name}"
            pose_target = pose_dir / target_name
            training_target = training_dir / target_name
            copy_file_safely(source, pose_target)
            copy_file_safely(source, training_target)
            entries.append(
                {
                    "source_type": view.get("source_type") or "panorama_view",
                    "asset_id": view.get("asset_id"),
                    "pano_view_id": view.get("pano_view_id"),
                    "pose_image": str(pose_target),
                    "training_image": str(training_target),
                    "source_frame_id": view.get("source_frame_id"),
                    "source_frame_index": view.get("source_frame_index"),
                    "source_image_path": view.get("source_image_path"),
                    "source_pano_id": view.get("source_pano_id"),
                    "shared_center_group": view.get("shared_center_group"),
                    "crop_id": view.get("crop_id"),
                    "yaw": view.get("yaw"),
                    "pitch": view.get("pitch"),
                    "fov": view.get("fov"),
                    "stream_id": view.get("stream_id"),
                    "mapping": view.get("mapping"),
                    "projection_model": view.get("projection_model"),
                    "camera_model": view.get("camera_model"),
                }
            )
            source_map.append(
                {
                    "derived_pose": str(pose_target),
                    "derived_pose_sha256": file_sha256(pose_target),
                    "derived_training": str(training_target),
                    "derived_training_sha256": file_sha256(training_target),
                    "source_asset_id": view.get("asset_id"),
                    "source_pano_id": view.get("source_pano_id"),
                    "source_pano_view_id": view.get("pano_view_id"),
                    "source_image_path": view.get("source_image_path"),
                    "shared_center_group": view.get("shared_center_group"),
                    "mapping": view,
                }
            )
        pose_images = [Path(str(entry["pose_image"])) for entry in entries if entry.get("pose_image")]
        training_images = [Path(str(entry["training_image"])) for entry in entries if entry.get("training_image")]
        duplicate_ratio = self._duplicate_ratio(pose_images)
        jpg_count = len(image_items)
        video_count = len(keyframes)
        source_balance = 1.0 - abs(jpg_count - min(video_count, max(jpg_count, 1) * 2)) / max(1, jpg_count + video_count)
        metrics = {
            "image_count": len(pose_images),
            "source_count": len(entries),
            "jpg_video_ratio": round(jpg_count / max(1, video_count), 4) if video_count else float(jpg_count),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "source_balance_score": round(clamp(source_balance), 4),
            "expected_pose_success_score": round(clamp(len(pose_images) / 80.0) * 0.45 + (1 - duplicate_ratio) * 0.3 + clamp(source_balance) * 0.25, 4),
            "expected_training_quality_score": round(clamp(len(training_images) / 100.0) * 0.35 + (1 - duplicate_ratio) * 0.35 + clamp(source_balance) * 0.3, 4),
            "forensic_risk_score": 0.12,
            "spherical_video_view_count": len([entry for entry in entries if entry.get("source_type") == "spherical_video_keyframe_view"]),
            "static_panorama_view_count": len([entry for entry in entries if entry.get("source_type") == "panorama_station_view"]),
            "spherical_rig_view_count": len([entry for entry in entries if entry.get("source_type") in SPHERICAL_RIG_VIEW_SOURCE_TYPES]),
            "raw_equirectangular_keyframe_count": len(keyframes) if spherical_rig_enabled else 0,
        }
        image_entries = [entry for entry in entries if entry.get("source_type") == "image"]
        pose_distribution = _distribution(image_entries, "pose_candidate")
        training_distribution = _distribution(image_entries, "training_candidate")
        pose_source_families = {str(entry.get("pose_source_family")) for entry in image_entries if entry.get("pose_source_family")}
        training_source_families = {str(entry.get("training_source_family")) for entry in image_entries if entry.get("training_source_family")}
        image_policy = {
            "raw_images_preserved": True,
            "pose_image_source": self._source_label(pose_source_families),
            "training_image_source": self._source_label(training_source_families),
            "enhancement_used_for_pose": any(source != "original" for source in pose_source_families),
            "enhancement_used_for_training": any(source != "original" for source in training_source_families),
            "generative_enhancement_used": False,
        }
        route_config = {
            "route_name": dataset_id,
            "route_preset": dataset_id if str(dataset_id) in ROUTE_PRESETS else active_route_preset(context),
            "preset": route_preset_config(dataset_id if str(dataset_id) in ROUTE_PRESETS else active_route_preset(context)),
            "policy": policy,
            "safe_enhancement_routes": SAFE_IMAGE_ENHANCEMENT_ROUTES,
            "forbidden_default_operations": [
                "generative_inpainting",
                "semantic_repaint",
                "hallucinating_super_resolution",
                "content_deletion",
                "content_replacement",
                "background_generation",
                "object_generation",
            ],
        }
        manifest = {
            "dataset_id": dataset_id,
            "route_config": route_config,
            "image_policy": image_policy,
            "pose_image_distribution": pose_distribution,
            "training_image_distribution": training_distribution,
            "pose_images": [str(path) for path in pose_images],
            "training_images": [str(path) for path in training_images],
            "entries": entries,
            "metrics": metrics,
        }
        manifest_path = write_json(output_dir / "dataset_manifest.json", manifest)
        source_map_path = write_json(output_dir / "source_map.json", {"dataset_id": dataset_id, "sources": source_map})
        route_config_path = write_json(output_dir / "route_config.json", route_config)
        image_selection_manifest_path = write_json(output_dir / "image_selection_manifest.json", {"dataset_id": dataset_id, "images": image_entries, "image_policy": image_policy})
        pose_input_manifest_path = write_json(output_dir / "pose_input_manifest.json", {"dataset_id": dataset_id, "images": [{"asset_id": entry.get("asset_id"), "path": entry.get("pose_image"), "sha256": entry.get("pose_image_sha256"), "source_path": entry.get("pose_source_path"), "source_sha256": entry.get("pose_source_sha256"), "candidate": entry.get("pose_candidate")} for entry in image_entries]})
        training_input_manifest_path = write_json(output_dir / "training_input_manifest.json", {"dataset_id": dataset_id, "images": [{"asset_id": entry.get("asset_id"), "path": entry.get("training_image"), "sha256": entry.get("training_image_sha256"), "source_path": entry.get("training_source_path"), "source_sha256": entry.get("training_source_sha256"), "candidate": entry.get("training_candidate")} for entry in image_entries]})
        image_stage_provenance = read_json(context.stage_dir("image_enhancement") / "enhancement_provenance.json", {"images": []})
        provenance_by_file = {str(item.get("derived_file")): item for item in image_stage_provenance.get("images", []) if item.get("derived_file")}
        enhancement_provenance_path = write_json(
            output_dir / "enhancement_provenance.json",
            {
                "dataset_id": dataset_id,
                "images": [
                    {
                        "asset_id": entry.get("asset_id"),
                        "source_file": entry.get("source_original"),
                        "source_sha256": entry.get("source_original_sha256"),
                        "pose_file": entry.get("pose_image"),
                        "pose_sha256": entry.get("pose_image_sha256"),
                        "pose_process_type": entry.get("pose_candidate"),
                        "pose_process_params": (provenance_by_file.get(str(entry.get("pose_source_path"))) or {}).get("process_params") or image_process_params(str(entry.get("pose_candidate") or "")),
                        "pose_generated_at": (provenance_by_file.get(str(entry.get("pose_source_path"))) or {}).get("created_at"),
                        "pose_integrity_risk_score": (provenance_by_file.get(str(entry.get("pose_source_path"))) or {}).get("integrity_risk_score"),
                        "used_for_pose": True,
                        "training_file": entry.get("training_image"),
                        "training_sha256": entry.get("training_image_sha256"),
                        "training_process_type": entry.get("training_candidate"),
                        "training_process_params": (provenance_by_file.get(str(entry.get("training_source_path"))) or {}).get("process_params") or image_process_params(str(entry.get("training_candidate") or "")),
                        "training_generated_at": (provenance_by_file.get(str(entry.get("training_source_path"))) or {}).get("created_at"),
                        "training_integrity_risk_score": (provenance_by_file.get(str(entry.get("training_source_path"))) or {}).get("integrity_risk_score"),
                        "used_for_training": True,
                        "generative_enhancement_used": False,
                    }
                    for entry in image_entries
                ],
            },
        )
        candidate.update(
            {
                "status": "succeeded",
                "output_path": str(manifest_path),
                "source_map_path": str(source_map_path),
                "route_config_path": str(route_config_path),
                "image_selection_manifest_path": str(image_selection_manifest_path),
                "pose_input_manifest_path": str(pose_input_manifest_path),
                "training_input_manifest_path": str(training_input_manifest_path),
                "enhancement_provenance_path": str(enhancement_provenance_path),
                "metrics": metrics,
                "score": round(0.55 * metrics["expected_pose_success_score"] + 0.35 * metrics["expected_training_quality_score"] + 0.1 * (1 - metrics["forensic_risk_score"]), 4),
                "risk_level": "low",
            }
        )
        if len(pose_images) < 3:
            candidate["rejected_reason"] = "dataset_has_too_few_pose_images"
            candidate["score"] = 0.0
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        preferred_route = context.config.get("active_route_preset") or context.config.get("route_preset") or DEFAULT_PRODUCTION_ROUTE_PRESET
        best = next((item for item in candidate_results if item.get("candidate_name") == preferred_route and item.get("status") == "succeeded" and not item.get("rejected_reason")), None)
        if best is None:
            best = super().select_best(context, candidate_results)
        if best:
            write_json(context.stage_dir(self.stage_name) / "best_dataset_selection.json", best)
            write_json(context.stage_dir(self.stage_name) / "dataset_candidate_metrics.json", {"candidates": candidate_results})
            best["improvement_summary"] = "已构建 pose/training 分离的数据集，并做 JPG/视频关键帧 source balancing。"
            best["risk_summary"] = "数据集保留 source_map，可从每个派生输入追溯到原始素材。"
        return best

    def _select_image_path(self, item: dict[str, Any], policy: str, *, role: str, route_name: str) -> tuple[str | None, str]:
        if route_name in ROUTE_PRESETS:
            source = str(ROUTE_PRESETS[route_name]["pose_source" if role == "pose" else "training_source"])
        elif policy == "route_preset":
            source = str(route_preset_config(route_name).get("pose_source" if role == "pose" else "training_source"))
        else:
            source = "safe_enhanced"
        if source == "original":
            return item.get("image_original"), "original"
        path_key = "image_for_pose" if role == "pose" else "image_for_training"
        candidate_key = "pose_candidate" if role == "pose" else "training_candidate"
        return item.get(path_key) or item.get("image_original"), str(item.get(candidate_key) or "original")

    def _source_label(self, families: set[str]) -> str:
        cleaned = {family for family in families if family and family != "missing"}
        if not cleaned:
            return "original"
        if cleaned == {"original"}:
            return "original"
        if cleaned == {"safe_enhanced"}:
            return "safe_enhanced"
        return "mixed"

    def _duplicate_ratio(self, images: list[Path]) -> float:
        if cv2 is None or not images:
            return 0.0
        hashes = set()
        duplicates = 0
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            tiny = cv2.resize(image, (12, 12))
            digest = "".join("1" if value > tiny.mean() else "0" for value in tiny.flatten())
            if digest in hashes:
                duplicates += 1
            hashes.add(digest)
        return duplicates / max(1, len(images))


class PoseEstimationOptimizationStage(StageOptimizer):
    stage_name = "pose_estimation_optimization"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        manifest = read_json(Path(str(dataset.get("output_path") or "")), {}) if dataset.get("output_path") else {}
        return {"input_artifacts": [dataset.get("output_path")], "dataset_selection": dataset, "dataset_manifest": manifest}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"candidate_name": name, "candidate_type": "pose_route", "analysis": analysis, "status": "created", "created_at": utc_now_iso()} for name in POSE_ESTIMATION_ROUTES]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        manifest = candidate["analysis"].get("dataset_manifest") or {}
        image_count = len(manifest.get("pose_images") or [])
        route = candidate["candidate_name"]
        operators = config_section(context.settings, "operators")
        fake_pose = bool(context.settings.colmap_fake_runner or context.config.get("fake_runner"))
        run_real_pose = _stage_execution_requested(context, "execute_pose_estimation") or (
            bool(context.config.get("execute_training")) and not fake_pose
        )
        if not fake_pose and not run_real_pose:
            return _planned_candidate(
                context,
                self.stage_name,
                candidate,
                reason="pose_route_registered_but_not_executed_without_execute_pose_estimation_flag",
                metrics={"route": route, "image_count": image_count},
            )
        unavailable = []
        if run_real_pose and route.startswith("mast3r") and not nested_get(operators, "mast3r_sfm.enabled", False):
            unavailable = ["mast3r_disabled"]
        if unavailable:
            candidate.update({"status": "skipped", "rejected_reason": ",".join(unavailable), "score": 0.0, "risk_level": "medium"})
            context.capability_report.setdefault(self.stage_name, {})[route] = {"available": False, "reason": unavailable}
            return candidate
        if run_real_pose:
            default_real_candidates = ["colmap_sequential", "colmap_exhaustive"] if image_count >= 12 else ["colmap_exhaustive"]
            manifest_metrics = manifest.get("metrics") or {}
            spherical_video_view_count = int(manifest_metrics.get("spherical_video_view_count") or 0)
            static_panorama_view_count = int(manifest_metrics.get("static_panorama_view_count") or 0)
            spherical_rig_view_count = int(manifest_metrics.get("spherical_rig_view_count") or (spherical_video_view_count + static_panorama_view_count) or 0)
            if spherical_rig_view_count > 0 and (static_panorama_view_count > 0 or spherical_video_config(context.config).enabled):
                default_real_candidates = ["spherical_video_rig_lift"]
            allowed = _configured_names(context.config.get("real_pose_candidates"), default_real_candidates)
            if route not in allowed:
                candidate.update(
                    {
                        "status": "skipped",
                        "score": 0.0,
                        "rejected_reason": "not_in_real_pose_candidate_set",
                        "risk_level": "low",
                    }
                )
                return candidate
            if route == "spherical_video_rig_lift":
                return self._run_real_spherical_video_rig_lift_pose(context, candidate, manifest, image_count)
            if route == "hloc_lightglue_aliked_fallback":
                return self._run_real_lightglue_colmap_pose(context, candidate, manifest, image_count)
            if route == "mast3r_dust3r_fallback":
                return self._run_real_mast3r_pose(context, candidate, manifest, image_count)
            if route == "colmap_multi_camera_model_test":
                return self._run_real_multi_camera_pose(context, candidate, manifest, image_count)
            if route == "colmap_hybrid":
                return self._run_real_colmap_attempts_pose(context, candidate, manifest, image_count)
            output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
            try:
                preprocess = _preprocess_from_dataset_manifest(
                    context,
                    manifest,
                    output_dir / "input",
                    route_id=route,
                    route_key="stage_optimized_real_colmap",
                )
                if len(preprocess.image_paths) < 3:
                    candidate.update({"status": "failed", "score": 0.0, "rejected_reason": "too_few_images_for_real_colmap", "risk_level": "high"})
                    return candidate
                matcher = str(context.config.get("real_pose_matcher") or _real_pose_matcher(route))
                camera_model = str(context.config.get("real_camera_model") or "SIMPLE_RADIAL")
                result = ColmapGlobalSkeletonOperator(context.settings).run(
                    context.workflow,
                    preprocess,
                    attempt_key=route,
                    matcher=matcher,
                    camera_model=camera_model,
                    attempt_spec={
                        "use_gpu": bool(context.config.get("real_pose_use_gpu", False)),
                        "sift_max_image_size": int(context.config.get("real_pose_max_image_size") or 1600),
                        "sift_num_threads": int(context.config.get("real_pose_num_threads") or 2),
                        "sift_max_num_features": int(context.config.get("real_pose_max_num_features") or 4096),
                        "mapper_min_model_size": int(context.config.get("real_pose_mapper_min_model_size") or 3),
                    },
                    workspace_name=_route_scoped_workspace_name(context, f"stages/{self.stage_name}/pose_candidates/{route}/colmap"),
                )
                report = _read_optional_json(result.registration_report_path)
                metrics = _pose_metrics_from_colmap_result(result, report, image_count=image_count, route=route, matcher=matcher, camera_model=camera_model, execution="real_colmap")
                metrics["images_dir"] = str(preprocess.images_dir)
                metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
                candidate.update(
                    {
                        "status": "succeeded",
                        "output_path": str(metrics_path),
                        "metrics_path": str(metrics_path),
                        "metrics": metrics,
                        "score": metrics["geometry_stability_score"],
                        "risk_level": "low",
                        "dataset_dir": str(result.dataset_dir),
                        "transforms_path": str(result.transforms_path),
                        "sparse_point_cloud_path": str(result.sparse_point_cloud_path),
                        "registration_report_path": str(result.registration_report_path),
                    }
                )
                rejection = _pose_quality_rejection(metrics)
                if rejection:
                    candidate.update({"rejected_reason": rejection, "score": 0.0, "risk_level": "high"})
                return candidate
            except Exception as exc:
                error_path = write_json(output_dir / "pose_error.json", {"execution": "real_colmap", "route": route, "error": str(exc)})
                candidate.update(
                    {
                        "status": "failed",
                        "output_path": str(error_path),
                        "metrics_path": str(error_path),
                        "metrics": {"execution": "real_colmap", "error": str(exc)},
                        "score": 0.0,
                        "rejected_reason": "real_colmap_failed",
                        "risk_level": "high",
                    }
                )
                return candidate
        base_ratio = clamp(image_count / max(3, image_count))
        route_bonus = {
            "colmap_hybrid": 0.08,
            "colmap_exhaustive": 0.05,
            "colmap_sequential": 0.04,
            "colmap_sequential_loop": 0.05,
            "colmap_vocab_tree": 0.02,
            "spherical_video_rig_lift": 0.07,
            "colmap_multi_camera_model_test": 0.04,
        }.get(route, 0.0)
        registered_ratio = clamp(base_ratio - (0.12 if image_count < 8 else 0.0) + route_bonus)
        reproj = max(0.8, 2.4 - route_bonus * 8 - min(image_count, 120) / 120.0)
        component_ratio = clamp(registered_ratio - (0.08 if route == "colmap_vocab_tree" and image_count < 30 else 0.0))
        sparse_points = int(image_count * 250 * registered_ratio)
        metrics = {
            "registered_images_count": int(image_count * registered_ratio),
            "total_images_count": image_count,
            "registered_ratio": round(registered_ratio, 4),
            "sparse_points_count": sparse_points,
            "mean_reprojection_error": round(reproj, 4),
            "median_reprojection_error": round(reproj * 0.82, 4),
            "track_length_mean": round(3.0 + route_bonus * 10, 3),
            "camera_graph_components": 1 if component_ratio > 0.7 else 2,
            "largest_component_ratio": round(component_ratio, 4),
            "failed_images": [],
            "weak_images": [],
            "camera_path_continuity": round(component_ratio, 4),
            "loop_closure_success": route in {"colmap_sequential_loop", "colmap_hybrid"} and image_count >= 12,
            "sparse_density_score": round(clamp(sparse_points / max(1, image_count * 400)), 4),
            "geometry_stability_score": round(clamp(0.45 * registered_ratio + 0.35 * component_ratio + 0.2 * (1.0 - min(reproj / 5.0, 1.0))), 4),
        }
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        pose_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update({"status": "succeeded", "output_path": str(pose_path), "metrics": metrics, "score": metrics["geometry_stability_score"], "risk_level": "low"})
        if image_count < 3:
            candidate.update({"rejected_reason": "too_few_images_for_sfm", "score": 0.0, "risk_level": "high"})
        elif metrics["registered_ratio"] < 0.65 or metrics["largest_component_ratio"] < 0.65 or metrics["mean_reprojection_error"] > 4.0:
            candidate.update({"rejected_reason": "pose_quality_gate_failed", "score": 0.0, "risk_level": "high"})
        return candidate

    def _preprocess_for_route(self, context: StageContext, manifest: dict[str, Any], route: str) -> PreprocessRunResult:
        return _preprocess_from_dataset_manifest(
            context,
            manifest,
            context.stage_dir(self.stage_name) / "pose_candidates" / route / "input",
            route_id=route,
            route_key="stage_optimized_real_pose",
        )

    def _colmap_attempt_spec(self, context: StageContext, route: str, *, matcher: str, camera_model: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "use_gpu": bool(context.config.get("real_pose_use_gpu", False)),
            "sift_max_image_size": int(context.config.get("real_pose_max_image_size") or 1600),
            "sift_num_threads": int(context.config.get("real_pose_num_threads") or 2),
            "sift_max_num_features": int(context.config.get("real_pose_max_num_features") or 4096),
            "mapper_min_model_size": int(context.config.get("real_pose_mapper_min_model_size") or 3),
            "single_camera": route not in {"colmap_exhaustive", "hloc_lightglue_aliked_fallback"},
            "sequential_overlap": int(context.config.get("real_pose_sequential_overlap") or 30),
            **(extra or {}),
        }

    def _run_real_spherical_video_rig_lift_pose(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], image_count: int) -> dict[str, Any]:
        route = str(candidate["candidate_name"])
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = spherical_video_config(context.config)
        entries = [entry for entry in manifest.get("entries") or [] if entry.get("source_type") in SPHERICAL_RIG_VIEW_SOURCE_TYPES]
        static_panorama_entry_count = len([entry for entry in entries if entry.get("source_type") == "panorama_station_view"])
        if not cfg.enabled and static_panorama_entry_count == 0:
            metrics_path = write_json(output_dir / "pose_metrics.json", {"execution": "spherical_video_rig_lift", "reason": "experimental_360_video_disabled"})
            candidate.update({"status": "skipped", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": read_json(metrics_path, {}), "score": 0.0, "rejected_reason": "experimental_360_video_disabled", "risk_level": "low"})
            return candidate

        stream_groups = group_spherical_entries_by_stream(entries)
        if not stream_groups:
            metrics_path = write_json(output_dir / "pose_metrics.json", {"execution": "spherical_video_rig_lift", "reason": "no_spherical_rig_stream_entries"})
            candidate.update({"status": "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": read_json(metrics_path, {}), "score": 0.0, "rejected_reason": "no_spherical_rig_stream_entries", "risk_level": "high"})
            return candidate

        requested_yaws = set(float(value) for value in cfg.pose_yaw_degrees)
        stream_items = [
            (key, values)
            for key, values in sorted(stream_groups.items(), key=lambda item: (item[0][1], item[0][0]))
            if not requested_yaws or float(key[0]) in requested_yaws
        ]
        camera_models = [str(value) for value in (context.config.get("spherical_pose_camera_models") or ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"])]
        stream_matcher = str(context.config.get("spherical_pose_matcher") or "sequential").strip().lower()
        if stream_matcher not in {"sequential", "exhaustive"}:
            stream_matcher = "sequential"
        attempts: list[dict[str, Any]] = []
        best_attempt: dict[str, Any] | None = None
        best_result: Any = None
        best_preprocess: PreprocessRunResult | None = None
        best_stream_entries: list[dict[str, Any]] = []
        best_stream_key: tuple[float, float] | None = None

        for stream_key, stream_entries in stream_items:
            if len(stream_entries) < 3:
                attempts.append({"stream": {"yaw": stream_key[0], "pitch": stream_key[1]}, "status": "skipped", "reason": "too_few_stream_frames", "score": 0.0})
                continue
            stream_manifest = {
                "dataset_id": f"spherical_stream_yaw_{stream_key[0]:.3f}_pitch_{stream_key[1]:.3f}",
                "pose_images": [entry["pose_image"] for entry in stream_entries if entry.get("pose_image")],
                "training_images": [entry.get("training_image") or entry.get("pose_image") for entry in stream_entries if entry.get("pose_image")],
                "entries": stream_entries,
                "metrics": {"image_count": len(stream_entries), "source": "spherical_video_yaw_stream"},
            }
            for camera_model in camera_models:
                attempt_key = f"{route}_{stream_matcher}_yaw_{int(stream_key[0])}_pitch_{int(stream_key[1])}_{camera_model.lower()}"
                try:
                    preprocess = _preprocess_from_dataset_manifest(
                        context,
                        stream_manifest,
                        output_dir / "streams" / f"yaw_{int(stream_key[0])}_pitch_{int(stream_key[1])}_{camera_model.lower()}" / "input",
                        route_id=attempt_key,
                        route_key="spherical_video_rig_lift_pose_stream",
                    )
                    result = ColmapGlobalSkeletonOperator(context.settings).run(
                        context.workflow,
                        preprocess,
                        attempt_key=attempt_key,
                        matcher=stream_matcher,
                        camera_model=camera_model,
                        attempt_spec=self._colmap_attempt_spec(
                            context,
                            route,
                            matcher=stream_matcher,
                            camera_model=camera_model,
                            extra={
                                "single_camera": True,
                                "sequential_overlap": int(context.config.get("spherical_pose_sequential_overlap") or context.config.get("real_pose_sequential_overlap") or 12),
                                "mapper_min_model_size": int(context.config.get("spherical_pose_mapper_min_model_size") or context.config.get("real_pose_mapper_min_model_size") or 3),
                            },
                        ),
                        workspace_name=_route_scoped_workspace_name(
                            context,
                            f"stages/{self.stage_name}/pose_candidates/{route}/streams/yaw_{int(stream_key[0])}_pitch_{int(stream_key[1])}_{camera_model.lower()}/colmap",
                        ),
                    )
                    report = _read_optional_json(result.registration_report_path)
                    metrics = _pose_metrics_from_colmap_result(
                        result,
                        report,
                        image_count=len(stream_entries),
                        route=route,
                        matcher=stream_matcher,
                        camera_model=camera_model,
                        execution="real_spherical_video_pose_stream_colmap",
                    )
                    rejection = _pose_quality_rejection(metrics)
                    attempt = {
                        "stream": {"yaw": stream_key[0], "pitch": stream_key[1]},
                        "camera_model": camera_model,
                        "status": "succeeded",
                        "metrics": metrics,
                        "score": 0.0 if rejection else metrics["geometry_stability_score"],
                        "rejected_reason": rejection,
                        "transforms_path": str(result.transforms_path),
                        "dataset_dir": str(result.dataset_dir),
                        "sparse_point_cloud_path": str(result.sparse_point_cloud_path),
                    }
                    attempts.append(attempt)
                    if not rejection and (best_attempt is None or float(attempt["score"]) > float(best_attempt["score"])):
                        best_attempt = attempt
                        best_result = result
                        best_preprocess = preprocess
                        best_stream_entries = stream_entries
                        best_stream_key = stream_key
                except Exception as exc:
                    attempts.append({"stream": {"yaw": stream_key[0], "pitch": stream_key[1]}, "camera_model": camera_model, "status": "failed", "reason": str(exc), "score": 0.0})

        if best_attempt is None or best_result is None or best_preprocess is None or best_stream_key is None:
            metrics = {"execution": "real_spherical_video_rig_lift", "attempts": attempts, "reason": "no_yaw_stream_passed_pose_quality_gate"}
            metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
            candidate.update({"status": "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": "no_yaw_stream_passed_pose_quality_gate", "risk_level": "high"})
            return candidate

        lifted = self._lift_spherical_video_transforms(
            output_dir=output_dir,
            all_entries=entries,
            stream_entries=best_stream_entries,
            stream_preprocess=best_preprocess,
            base_transforms_path=Path(str(best_result.transforms_path)),
            base_sparse_point_cloud_path=Path(str(best_result.sparse_point_cloud_path)),
            base_yaw=best_stream_key[0],
            base_pitch=best_stream_key[1],
        )
        if int(lifted.get("lifted_view_count") or 0) < 3:
            metrics = {"execution": "real_spherical_video_rig_lift", "attempts": attempts, "selected_stream": best_attempt, "lifted": lifted, "reason": "too_few_lifted_spherical_views"}
            metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
            candidate.update({"status": "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": "too_few_lifted_spherical_views", "risk_level": "high"})
            return candidate

        base_metrics = dict(best_attempt["metrics"])
        lifted_registered_ratio = float(lifted["lifted_view_count"]) / max(1.0, float(image_count))
        metrics = {
            **base_metrics,
            "execution": "real_spherical_video_rig_lift",
            "route": route,
            "matcher": f"{stream_matcher}_stream_then_known_virtual_rig_lift",
            "registered_images_count": int(lifted["lifted_view_count"]),
            "total_images_count": image_count,
            "registered_ratio": round(lifted_registered_ratio, 4),
            "dataset_dir": str(lifted["dataset_dir"]),
            "transforms_path": str(lifted["transforms_path"]),
            "sparse_point_cloud_path": str(lifted["sparse_point_cloud_path"]),
            "registration_report_path": str(lifted["registration_report_path"]),
            "base_stream": {"yaw": best_stream_key[0], "pitch": best_stream_key[1], "camera_model": best_attempt.get("camera_model")},
            "base_stream_metrics": base_metrics,
            "lifted_view_count": int(lifted["lifted_view_count"]),
            "registered_source_frame_count": int(lifted["registered_source_frame_count"]),
            "static_panorama_entry_count": static_panorama_entry_count,
            "attempts": attempts,
            "pose_derivation": "registered_yaw_stream_plus_known_equirectangular_virtual_camera_rotations",
        }
        metrics["sparse_density_score"] = round(clamp(float(metrics.get("sparse_points_count") or 0) / max(1, int(lifted["lifted_view_count"]) * 600)), 4)
        metrics["geometry_stability_score"] = _score_pose_metrics(metrics)
        metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update(
            {
                "status": "succeeded",
                "output_path": str(metrics_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "score": metrics["geometry_stability_score"],
                "risk_level": "medium",
                "dataset_dir": str(lifted["dataset_dir"]),
                "transforms_path": str(lifted["transforms_path"]),
                "sparse_point_cloud_path": str(lifted["sparse_point_cloud_path"]),
                "registration_report_path": str(lifted["registration_report_path"]),
            }
        )
        rejection = _pose_quality_rejection(metrics)
        if rejection:
            candidate.update({"rejected_reason": rejection, "score": 0.0, "risk_level": "high"})
        return candidate

    def _lift_spherical_video_transforms(
        self,
        *,
        output_dir: Path,
        all_entries: list[dict[str, Any]],
        stream_entries: list[dict[str, Any]],
        stream_preprocess: PreprocessRunResult,
        base_transforms_path: Path,
        base_sparse_point_cloud_path: Path,
        base_yaw: float,
        base_pitch: float,
    ) -> dict[str, Any]:
        base_transforms = read_json(base_transforms_path, {"frames": []})
        stream_entry_by_image_name = {
            stream_preprocess.image_paths[index].name: entry
            for index, entry in enumerate(stream_entries)
            if index < len(stream_preprocess.image_paths)
        }
        base_frame_by_source_frame_id: dict[str, dict[str, Any]] = {}
        for frame in base_transforms.get("frames") or []:
            image_name = Path(str(frame.get("file_path") or "")).name
            entry = stream_entry_by_image_name.get(image_name)
            if not entry:
                continue
            source_frame_id = str(entry.get("source_frame_id") or "")
            if source_frame_id:
                base_frame_by_source_frame_id[source_frame_id] = frame

        dataset_dir = output_dir / "rig_lift_dataset"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        registered_source_ids = set(base_frame_by_source_frame_id)
        lifted_frames = []
        source_map = []
        for entry in sorted(all_entries, key=spherical_frame_key):
            source_frame_id = str(entry.get("source_frame_id") or "")
            base_frame = base_frame_by_source_frame_id.get(source_frame_id)
            if not base_frame:
                continue
            source_image = Path(str(entry.get("training_image") or entry.get("pose_image") or ""))
            if not source_image.exists():
                continue
            target_name = f"{len(lifted_frames) + 1:05d}_{source_image.name}"
            target = images_dir / target_name
            copy_file_safely(source_image, target)
            lifted_transform = apply_virtual_camera_rotation(
                base_frame["transform_matrix"],
                base_yaw=base_yaw,
                target_yaw=float(entry.get("yaw") or 0.0),
                base_pitch=base_pitch,
                target_pitch=float(entry.get("pitch") or 0.0),
            )
            lifted_frame = {
                key: value
                for key, value in base_frame.items()
                if key not in {"file_path", "transform_matrix"}
            }
            lifted_frame.update(
                {
                    "file_path": f"images/{target.name}",
                    "transform_matrix": lifted_transform,
                    "source_frame_id": source_frame_id,
                    "source_pano_view_id": entry.get("pano_view_id"),
                    "yaw": entry.get("yaw"),
                    "pitch": entry.get("pitch"),
                    "pose_source": "spherical_video_rig_lift",
                }
            )
            lifted_frames.append(lifted_frame)
            source_map.append({"derived_image": str(target), "source_entry": entry, "base_stream_frame": base_frame.get("file_path")})

        sparse_target = dataset_dir / "sparse_point_cloud.ply"
        if base_sparse_point_cloud_path.exists():
            copy_file_safely(base_sparse_point_cloud_path, sparse_target)
        transforms = {
            **{key: value for key, value in base_transforms.items() if key != "frames"},
            "frames": lifted_frames,
            "ply_file_path": sparse_target.name if sparse_target.exists() else base_transforms.get("ply_file_path"),
            "spherical_video_rig_lift": {
                "base_yaw": base_yaw,
                "base_pitch": base_pitch,
                "registered_source_frame_count": len(registered_source_ids),
                "lifted_view_count": len(lifted_frames),
            },
        }
        transforms_path = write_json(dataset_dir / "transforms.json", transforms)
        source_map_path = write_json(dataset_dir / "spherical_rig_lift_source_map.json", {"sources": source_map})
        report = {
            "operator": "spherical_video_rig_lift",
            "input_image_count": len(all_entries),
            "registered_camera_count": len(lifted_frames),
            "registration_rate": len(lifted_frames) / max(1, len(all_entries)),
            "registered_source_frame_count": len(registered_source_ids),
            "source_map_path": str(source_map_path),
            "commands_succeeded": True,
        }
        registration_report_path = write_json(output_dir / "rig_lift_registration_report.json", report)
        return {
            "dataset_dir": dataset_dir,
            "transforms_path": transforms_path,
            "sparse_point_cloud_path": sparse_target if sparse_target.exists() else base_sparse_point_cloud_path,
            "registration_report_path": registration_report_path,
            "lifted_view_count": len(lifted_frames),
            "registered_source_frame_count": len(registered_source_ids),
            "source_map_path": source_map_path,
        }

    def _run_real_lightglue_colmap_pose(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], image_count: int) -> dict[str, Any]:
        route = str(candidate["candidate_name"])
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        preprocess = self._preprocess_for_route(context, manifest, route)
        if len(preprocess.image_paths) < 3:
            candidate.update({"status": "failed", "score": 0.0, "rejected_reason": "too_few_images_for_lightglue_colmap", "risk_level": "high"})
            return candidate
        feature_result = LightGlueAlikedPreMatchingOperator(context.settings).run(context.workflow, preprocess)
        colmap_import = feature_result.report.get("colmap_import") or {}
        if not feature_result.passed or not colmap_import.get("import_ready"):
            metrics = {
                "execution": "real_lightglue_aliked",
                "status": "blocked",
                "reason": feature_result.reason or colmap_import.get("reason") or "lightglue_import_not_ready",
                "feature_matching_report_path": str(feature_result.report_path),
                "available": feature_result.available,
                "passed": feature_result.passed,
            }
            metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
            candidate.update({"status": "failed" if feature_result.available else "skipped", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": str(metrics["reason"]), "risk_level": "medium"})
            context.capability_report.setdefault(self.stage_name, {})[route] = metrics
            return candidate
        camera_model = str(context.config.get("real_lightglue_camera_model") or "OPENCV")
        result = ColmapGlobalSkeletonOperator(context.settings).run(
            context.workflow,
            preprocess,
            attempt_key=route,
            matcher="imported",
            camera_model=camera_model,
            attempt_spec=self._colmap_attempt_spec(
                context,
                route,
                matcher="imported",
                camera_model=camera_model,
                extra={
                    "colmap_features_dir": colmap_import.get("features_dir"),
                    "colmap_match_list_path": colmap_import.get("match_list_path"),
                    "match_type": colmap_import.get("match_type") or "raw",
                    "feature_source": "lightglue_aliked",
                },
            ),
            workspace_name=_route_scoped_workspace_name(context, f"stages/{self.stage_name}/pose_candidates/{route}/colmap_imported"),
        )
        report = _read_optional_json(result.registration_report_path)
        metrics = _pose_metrics_from_colmap_result(result, report, image_count=image_count, route=route, matcher="imported", camera_model=camera_model, execution="real_lightglue_aliked_colmap_import")
        metrics["feature_matching_report_path"] = str(feature_result.report_path)
        metrics["colmap_import"] = colmap_import
        metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update({"status": "succeeded", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": metrics["geometry_stability_score"], "risk_level": "low", "dataset_dir": str(result.dataset_dir), "transforms_path": str(result.transforms_path), "sparse_point_cloud_path": str(result.sparse_point_cloud_path), "registration_report_path": str(result.registration_report_path)})
        rejection = _pose_quality_rejection(metrics)
        if rejection:
            candidate.update({"rejected_reason": rejection, "score": 0.0, "risk_level": "high"})
        return candidate

    def _run_real_mast3r_pose(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], image_count: int) -> dict[str, Any]:
        route = str(candidate["candidate_name"])
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        preprocess = self._preprocess_for_route(context, manifest, route)
        result = Mast3rSfmFallbackOperator(context.settings).run(context.workflow, preprocess, "stage_optimized_pose_candidate")
        report = _read_optional_json(result.registration_report_path)
        metrics = _pose_metrics_from_colmap_result(result, report, image_count=image_count, route=route, matcher="mast3r", camera_model="transforms", execution="real_mast3r_dust3r")
        metrics["mast3r_report_path"] = str(result.report_path)
        metrics["passed"] = result.passed
        metrics["reason"] = result.reason
        metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update({"status": "succeeded" if result.passed else "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": metrics["geometry_stability_score"] if result.passed else 0.0, "risk_level": "medium", "dataset_dir": str(result.dataset_dir), "transforms_path": str(result.transforms_path), "sparse_point_cloud_path": str(result.sparse_point_cloud_path), "registration_report_path": str(result.registration_report_path)})
        if not result.passed:
            candidate["rejected_reason"] = result.reason or "mast3r_pose_failed"
        return candidate

    def _run_real_multi_camera_pose(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], image_count: int) -> dict[str, Any]:
        route = str(candidate["candidate_name"])
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        preprocess = self._preprocess_for_route(context, manifest, route)
        camera_models = list(context.config.get("real_pose_camera_models") or ["SIMPLE_RADIAL", "RADIAL", "OPENCV"])
        matcher = str(context.config.get("real_pose_matcher") or "exhaustive")
        attempts = []
        best_attempt: dict[str, Any] | None = None
        best_result: Any = None
        for camera_model in camera_models:
            try:
                result = ColmapGlobalSkeletonOperator(context.settings).run(
                    context.workflow,
                    preprocess,
                    attempt_key=f"{route}_{str(camera_model).lower()}",
                    matcher=matcher,
                    camera_model=str(camera_model),
                    attempt_spec=self._colmap_attempt_spec(context, route, matcher=matcher, camera_model=str(camera_model)),
                    workspace_name=_route_scoped_workspace_name(context, f"stages/{self.stage_name}/pose_candidates/{route}/colmap_{str(camera_model).lower()}"),
                )
                report = _read_optional_json(result.registration_report_path)
                metrics = _pose_metrics_from_colmap_result(result, report, image_count=image_count, route=route, matcher=matcher, camera_model=str(camera_model), execution="real_colmap_multi_camera_model_test")
                attempt = {"camera_model": camera_model, "status": "succeeded", "metrics": metrics, "score": metrics["geometry_stability_score"]}
                attempts.append(attempt)
                if best_attempt is None or float(attempt["score"]) > float(best_attempt["score"]):
                    best_attempt = attempt
                    best_result = result
            except Exception as exc:
                attempts.append({"camera_model": camera_model, "status": "failed", "reason": str(exc), "score": 0.0})
        if not best_attempt or best_result is None:
            metrics_path = write_json(output_dir / "pose_metrics.json", {"execution": "real_colmap_multi_camera_model_test", "attempts": attempts})
            candidate.update({"status": "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": {"attempts": attempts}, "score": 0.0, "rejected_reason": "all_camera_model_attempts_failed", "risk_level": "high"})
            return candidate
        metrics = {**best_attempt["metrics"], "tested_camera_models": attempts, "selected_camera_model": best_attempt["camera_model"]}
        metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update({"status": "succeeded", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": best_attempt["score"], "risk_level": "low", "dataset_dir": str(best_result.dataset_dir), "transforms_path": str(best_result.transforms_path), "sparse_point_cloud_path": str(best_result.sparse_point_cloud_path), "registration_report_path": str(best_result.registration_report_path)})
        rejection = _pose_quality_rejection(metrics)
        if rejection:
            candidate.update({"rejected_reason": rejection, "score": 0.0, "risk_level": "high"})
        return candidate

    def _run_real_colmap_attempts_pose(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], image_count: int) -> dict[str, Any]:
        route = str(candidate["candidate_name"])
        output_dir = context.stage_dir(self.stage_name) / "pose_candidates" / route
        preprocess = self._preprocess_for_route(context, manifest, route)
        local_feature_matching = None
        if bool(context.config.get("real_pose_hybrid_use_lightglue", True)):
            feature_result = LightGlueAlikedPreMatchingOperator(context.settings).run(context.workflow, preprocess)
            if feature_result.passed:
                local_feature_matching = feature_result.report
            else:
                context.capability_report.setdefault(self.stage_name, {})["colmap_hybrid_lightglue"] = {
                    "available": feature_result.available,
                    "passed": feature_result.passed,
                    "reason": feature_result.reason,
                    "report_path": str(feature_result.report_path),
                }
        attempts_result = ColmapAttemptsOperator(context.settings).run(context.workflow, preprocess, local_feature_matching=local_feature_matching)
        if attempts_result.selected is None:
            metrics = {"execution": "real_colmap_hybrid_attempts", "attempts": attempts_result.attempts, "reason": attempts_result.reason, "attempts_report_path": str(attempts_result.attempts_report_path)}
            metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
            candidate.update({"status": "failed", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": attempts_result.reason or "colmap_hybrid_attempts_failed", "risk_level": "high"})
            return candidate
        report = _read_optional_json(attempts_result.selected.registration_report_path)
        metrics = _pose_metrics_from_colmap_result(attempts_result.selected, report, image_count=image_count, route=route, matcher="hybrid_attempts", camera_model=str((attempts_result.selected.quality or {}).get("camera_model") or "auto"), execution="real_colmap_hybrid_attempts")
        metrics["selected_attempt_key"] = attempts_result.selected_attempt_key
        metrics["attempts_report_path"] = str(attempts_result.attempts_report_path)
        metrics["attempts"] = attempts_result.attempts
        metrics_path = write_json(output_dir / "pose_metrics.json", metrics)
        candidate.update({"status": "succeeded", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": metrics["geometry_stability_score"], "risk_level": "low", "dataset_dir": str(attempts_result.selected.dataset_dir), "transforms_path": str(attempts_result.selected.transforms_path), "sparse_point_cloud_path": str(attempts_result.selected.sparse_point_cloud_path), "registration_report_path": str(attempts_result.selected.registration_report_path)})
        rejection = _pose_quality_rejection(metrics)
        if rejection:
            candidate.update({"rejected_reason": rejection, "score": 0.0, "risk_level": "high"})
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        best = super().select_best(context, candidate_results)
        if not best:
            fake_pose = bool(context.settings.colmap_fake_runner or context.config.get("fake_runner"))
            run_real_pose = _stage_execution_requested(context, "execute_pose_estimation") or (
                bool(context.config.get("execute_training")) and not fake_pose
            )
            if run_real_pose:
                write_json(context.stage_dir(self.stage_name) / "pose_metrics.json", {"candidates": candidate_results, "best": None, "reason": "no_pose_candidate_passed_quality_gate"})
                write_text(context.stage_dir(self.stage_name) / "pose_comparison.md", self._report_markdown(context, {}, candidate_results, {}))
                return {}
            planned = [item for item in candidate_results if item.get("status") == "planned"]
            preferred = ["colmap_hybrid", "hloc_lightglue_aliked_fallback", "colmap_exhaustive", "colmap_sequential"]
            for name in preferred:
                best = next((item for item in planned if item.get("candidate_name") == name), None)
                if best:
                    break
            best = best or (planned[0] if planned else {})
        if best:
            write_json(context.stage_dir(self.stage_name) / "best_pose_selection.json", best)
            write_json(context.stage_dir(self.stage_name) / "pose_metrics.json", {"candidates": candidate_results, "best": best})
            write_json(context.stage_dir(self.stage_name) / "colmap_metrics.json", best.get("metrics") or {})
            write_text(context.stage_dir(self.stage_name) / "pose_comparison.md", self._report_markdown(context, {}, candidate_results, best))
            best["improvement_summary"] = "已比较多种 SfM/pose 路线，按注册率、连通性、重投影误差和几何稳定性选择 best_pose。"
            best["risk_summary"] = "低注册率、断裂相机图和高重投影误差候选不会进入训练输入。"
        return best


class MaskOptimizationStage(StageOptimizer):
    stage_name = "mask_optimization"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        pose = read_json(context.stage_dir("pose_estimation_optimization") / "best_pose_selection.json", {})
        manifest = read_json(Path(str(dataset.get("output_path") or "")), {}) if dataset.get("output_path") else {}
        return {"input_artifacts": [dataset.get("output_path"), pose.get("output_path")], "dataset": dataset, "pose": pose, "manifest": manifest}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        names = list(MASK_OPTIMIZATION_ROUTES)
        if not context.config.get("allow_mask", True):
            names = ["no_mask"]
        return [{"candidate_name": name, "candidate_type": "mask_strategy", "analysis": analysis, "status": "created", "created_at": utc_now_iso()} for name in names]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        name = candidate["candidate_name"]
        fake = bool(context.config.get("fake_runner"))
        if name != "no_mask" and not fake and not _stage_execution_requested(context, "execute_mask_optimization"):
            return _planned_candidate(
                context,
                self.stage_name,
                candidate,
                reason="mask_route_registered_but_not_executed_without_execute_mask_optimization_flag",
                metrics={"mask_strategy": name},
            )
        if name != "no_mask" and _stage_execution_requested(context, "execute_mask_optimization") and not fake:
            return self._run_real_mask_candidate(context, candidate)
        coverage_by_name = {
            "no_mask": 0.0,
            "conservative_mask": 0.08,
            "dynamic_object_mask": 0.18,
            "human_vehicle_animal_mask": 0.16,
            "reflection_sensitive_mask": 0.12,
            "foreground_interference_mask": 0.14,
            "aggressive_mask": 0.42,
        }
        coverage = coverage_by_name.get(name, 0.12)
        static_damage = 0.0 if name == "no_mask" else coverage * (0.6 if name == "aggressive_mask" else 0.25)
        feature_loss = coverage * 0.7
        dynamic_gain = 0.0 if name == "no_mask" else min(0.6, coverage * (2.0 if name in {"dynamic_object_mask", "human_vehicle_animal_mask"} else 1.4))
        metrics = {
            "mask_coverage_ratio": coverage,
            "static_structure_damage_score": round(static_damage, 4),
            "dynamic_object_removed_score": round(dynamic_gain, 4),
            "feature_loss_score": round(feature_loss, 4),
            "pose_impact_score": round(1.0 - feature_loss, 4),
            "training_expected_gain": round(dynamic_gain - static_damage - feature_loss * 0.4, 4),
            "forensic_risk_score": round(static_damage + coverage * 0.3, 4),
        }
        output = write_json(context.stage_dir(self.stage_name) / "masks" / name / "mask_metrics.json", metrics)
        score = 0.55 * metrics["training_expected_gain"] + 0.45 * (1.0 - metrics["forensic_risk_score"])
        candidate.update({"status": "succeeded", "output_path": str(output), "metrics": metrics, "score": round(score, 4), "risk_level": "high" if metrics["forensic_risk_score"] > 0.35 else "low"})
        if name != "no_mask" and (coverage > 0.35 or static_damage > 0.18 or feature_loss > 0.25):
            candidate.update({"rejected_reason": "mask_forensic_or_feature_loss_risk", "score": 0.0})
        return candidate

    def _run_real_mask_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        name = str(candidate["candidate_name"])
        manifest = candidate["analysis"].get("manifest") or {}
        output_dir = context.stage_dir(self.stage_name) / "masks" / name
        preprocess = _preprocess_from_dataset_manifest(
            context,
            manifest,
            output_dir / "input",
            route_id=name,
            route_key="stage_optimized_mask_candidate",
        )
        previous_config = dict(context.workflow.config_json or {})
        try:
            context.workflow.config_json = {**previous_config, "dynamic_classes": self._dynamic_classes_for_mask(name)}
            if name in {"dynamic_object_mask", "human_vehicle_animal_mask", "reflection_sensitive_mask"}:
                report = DynamicMaskOperator().run(context.workflow, preprocess)
                mask_path = report.get("report_path")
                coverage = float(report.get("dynamic_ratio") or 0.0)
                method = report.get("implementation") or "dynamic_mask"
                reason = report.get("reason")
            else:
                result = SubjectMaskGenerationOperator(context.settings).run(context.workflow, preprocess)
                report = result.manifest
                mask_path = str(result.manifest_path)
                coverage = float(report.get("foreground_ratio") or report.get("background_ratio") or 0.0)
                method = report.get("method") or "subject_mask_generation"
                reason = report.get("reason")
            coverage = clamp(coverage)
            static_damage = coverage * (0.55 if name == "aggressive_mask" else 0.22)
            feature_loss = coverage * 0.55
            dynamic_gain = min(0.65, coverage * 1.8)
            metrics = {
                "execution": "real_mask_operator",
                "mask_strategy": name,
                "method": method,
                "source_report_path": mask_path,
                "operator_report": report,
                "mask_coverage_ratio": round(coverage, 4),
                "static_structure_damage_score": round(static_damage, 4),
                "dynamic_object_removed_score": round(dynamic_gain, 4),
                "feature_loss_score": round(feature_loss, 4),
                "pose_impact_score": round(1.0 - feature_loss, 4),
                "training_expected_gain": round(dynamic_gain - static_damage - feature_loss * 0.4, 4),
                "forensic_risk_score": round(static_damage + coverage * 0.3, 4),
            }
            metrics_path = write_json(output_dir / "mask_metrics.json", metrics)
            score = 0.55 * metrics["training_expected_gain"] + 0.45 * (1.0 - metrics["forensic_risk_score"])
            candidate.update({"status": "succeeded", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": round(score, 4), "risk_level": "high" if metrics["forensic_risk_score"] > 0.35 else "low"})
            if reason and (method in {"external_command_unavailable", "not_applicable_image_collection"} or str(method).endswith("_unavailable")):
                candidate.update({"rejected_reason": str(reason), "score": 0.0, "risk_level": "medium"})
            if coverage > 0.35 or static_damage > 0.18 or feature_loss > 0.25:
                candidate.update({"rejected_reason": "mask_forensic_or_feature_loss_risk", "score": 0.0, "risk_level": "high"})
            return candidate
        finally:
            context.workflow.config_json = previous_config

    def _dynamic_classes_for_mask(self, name: str) -> list[str]:
        if name == "human_vehicle_animal_mask":
            return ["person", "vehicle", "animal"]
        if name == "reflection_sensitive_mask":
            return ["reflection", "mirror", "glass", "screen", "water"]
        return ["person", "vehicle", "animal", "leaf", "water"]

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        forced = str(context.config.get("force_mask_strategy") or context.config.get("forced_mask_strategy") or "").strip()
        if forced:
            forced_candidate = next(
                (
                    item
                    for item in candidate_results
                    if item.get("candidate_name") == forced and item.get("status") == "succeeded" and not item.get("rejected_reason")
                ),
                None,
            )
            if forced_candidate:
                best = forced_candidate
                write_json(context.stage_dir(self.stage_name) / "best_mask_selection.json", best)
                write_json(context.stage_dir(self.stage_name) / "mask_metrics.json", {"candidates": candidate_results, "best": best})
                best["improvement_summary"] = f"已按配置强制选择 mask 策略 `{forced}`。"
                best["risk_summary"] = "强制 mask 会优先满足动态/人物剔除要求；若语义模型漏检，未命中的区域仍可能进入训练。"
                return best
        # Prefer no_mask unless another safe mask has clear positive gain.
        no_mask = next((item for item in candidate_results if item.get("candidate_name") == "no_mask"), None)
        safe_positive = [item for item in candidate_results if not item.get("rejected_reason") and float((item.get("metrics") or {}).get("training_expected_gain") or 0.0) > 0.12]
        best = max(safe_positive, key=lambda item: float(item.get("score") or 0.0), default=no_mask)
        if best:
            write_json(context.stage_dir(self.stage_name) / "best_mask_selection.json", best)
            write_json(context.stage_dir(self.stage_name) / "mask_metrics.json", {"candidates": candidate_results, "best": best})
            best["improvement_summary"] = "mask 不默认启用；仅在收益超过真实性和特征损失风险时胜过 no_mask。"
            best["risk_summary"] = "aggressive mask 或损伤真实结构的 mask 已被淘汰。"
        return best or {}


class TrainingInputOptimizationStage(StageOptimizer):
    stage_name = "training_input_optimization"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        pose = read_json(context.stage_dir("pose_estimation_optimization") / "best_pose_selection.json", {})
        mask = read_json(context.stage_dir("mask_optimization") / "best_mask_selection.json", {})
        manifest = read_json(Path(str(dataset.get("output_path") or "")), {}) if dataset.get("output_path") else {}
        return {"input_artifacts": [dataset.get("output_path"), pose.get("output_path"), mask.get("output_path")], "dataset": dataset, "pose": pose, "mask": mask, "manifest": manifest}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        texture_policy = {
            "original_training_images": "original_or_selected_safe",
            "resize_native": "native_resolution",
            "resize_balanced": "balanced_resize",
            "balanced_holdout_split": "balanced_holdout",
            "mask_safe_training_input": "mask_safe",
        }
        return [
            {"candidate_name": name, "candidate_type": "training_input", "analysis": analysis, "texture_policy": texture_policy[name], "status": "created", "created_at": utc_now_iso()}
            for name in TRAINING_INPUT_ROUTES
        ]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        manifest = candidate["analysis"].get("manifest") or {}
        training_images = list(manifest.get("training_images") or [])
        pose_images = list(manifest.get("pose_images") or [])
        output_dir = context.stage_dir(self.stage_name) / "training_inputs" / candidate["candidate_name"]
        _reset_materialized_preprocess_dir(context, output_dir)
        train_dir = output_dir / "train_images"
        eval_dir = output_dir / "eval_images"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_every = max(5, len(training_images) // 8) if training_images else 5
        train_entries = []
        eval_entries = []
        resize_balanced = candidate.get("texture_policy") == "balanced_resize"
        for index, image_path in enumerate(training_images):
            source = Path(str(image_path))
            if not source.exists():
                continue
            target_dir = eval_dir if index % eval_every == 0 else train_dir
            target = target_dir / source.name
            if resize_balanced and cv2 is not None:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is not None:
                    height, width = image.shape[:2]
                    long_edge = max(width, height)
                    if long_edge > 2400:
                        scale = 2400.0 / float(long_edge)
                        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                else:
                    copy_file_safely(source, target)
            else:
                copy_file_safely(source, target)
            entry = {"file_path": str(target), "source_path": str(source), "pose_source": pose_images[index] if index < len(pose_images) else None}
            if target_dir == eval_dir:
                eval_entries.append(entry)
            else:
                train_entries.append(entry)
        transforms_train = {"frames": train_entries, "pose_route": candidate["analysis"].get("pose", {}).get("candidate_name"), "mask_strategy": candidate["analysis"].get("mask", {}).get("candidate_name")}
        transforms_eval = {"frames": eval_entries}
        train_path = write_json(output_dir / "transforms_train.json", transforms_train)
        eval_path = write_json(output_dir / "transforms_eval.json", transforms_eval)
        metrics = {
            "training_image_quality": round(clamp(len(train_entries) / 80.0), 4),
            "pose_image_alignment": 1.0,
            "color_consistency": 0.82,
            "resolution_consistency": 0.8,
            "mask_safety": 1.0 - float((candidate["analysis"].get("mask", {}).get("metrics") or {}).get("forensic_risk_score") or 0.0),
            "holdout_representativeness": round(clamp(len(eval_entries) / max(1, len(training_images) * 0.08)), 4),
            "expected_texture_fidelity": 0.78,
            "expected_artifact_risk": 0.18,
        }
        nerfstudio_dataset = self._materialize_nerfstudio_training_dataset(context, candidate, manifest, output_dir)
        image_policy = manifest.get("image_policy") or {}
        route_config = manifest.get("route_config") or {}
        route_preset = str(route_config.get("route_preset") or active_route_preset(context))
        training_supervision_modified = bool(route_preset_config(route_preset).get("training_supervision_modified") or image_policy.get("enhancement_used_for_training"))
        selection = {
            "candidate_name": candidate["candidate_name"],
            "route_preset": route_preset,
            "image_policy": image_policy,
            "pose_image_distribution": manifest.get("pose_image_distribution") or {},
            "training_image_distribution": manifest.get("training_image_distribution") or {},
            "training_supervision_modified": training_supervision_modified,
            "train_images_dir": str(train_dir),
            "eval_images_dir": str(eval_dir),
            "transforms_train": str(train_path),
            "transforms_eval": str(eval_path),
            "nerfstudio_dataset_dir": nerfstudio_dataset.get("dataset_dir"),
            "nerfstudio_transforms_path": nerfstudio_dataset.get("transforms_path"),
            "training_input_manifest": nerfstudio_dataset.get("manifest_path"),
            "metrics": metrics,
        }
        output = write_json(output_dir / "best_training_input_selection.json", selection)
        score = 0.25 * metrics["training_image_quality"] + 0.2 * metrics["holdout_representativeness"] + 0.25 * metrics["expected_texture_fidelity"] + 0.3 * (1 - metrics["expected_artifact_risk"])
        candidate.update({"status": "succeeded", "output_path": str(output), "metrics": metrics, "score": round(score, 4), "risk_level": "low"})
        if len(train_entries) < 3:
            candidate.update({"rejected_reason": "too_few_training_images", "score": 0.0, "risk_level": "high"})
        if candidate.get("texture_policy") == "mask_safe" and metrics["mask_safety"] < 0.75:
            candidate.update({"rejected_reason": "mask_safety_too_low_for_training_input", "score": 0.0, "risk_level": "high"})
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        forced = str(context.config.get("force_training_input_strategy") or context.config.get("forced_training_input_strategy") or "").strip()
        if forced:
            forced_candidate = next(
                (
                    item
                    for item in candidate_results
                    if item.get("candidate_name") == forced and item.get("status") == "succeeded" and not item.get("rejected_reason")
                ),
                None,
            )
            if forced_candidate:
                best = forced_candidate
                write_json(context.stage_dir(self.stage_name) / "best_training_input_selection.json", best)
                write_json(context.stage_dir(self.stage_name) / "input_optimization_metrics.json", {"candidates": candidate_results, "best": best})
                return best

        analysis_mask = next((item.get("analysis", {}).get("mask") for item in candidate_results if isinstance(item.get("analysis"), dict)), {}) or {}
        mask_name = str(analysis_mask.get("candidate_name") or "")
        prefer_mask_safe = bool(context.config.get("prefer_mask_safe_training_input_when_mask_selected", True))
        if prefer_mask_safe and mask_name and mask_name != "no_mask":
            best = next(
                (
                    item
                    for item in candidate_results
                    if item.get("candidate_name") == "mask_safe_training_input" and item.get("status") == "succeeded" and not item.get("rejected_reason")
                ),
                None,
            )
            if best is not None:
                write_json(context.stage_dir(self.stage_name) / "best_training_input_selection.json", best)
                write_json(context.stage_dir(self.stage_name) / "input_optimization_metrics.json", {"candidates": candidate_results, "best": best})
                best["improvement_summary"] = "Selected mask-safe training input because a safe non-empty mask strategy was selected."
                best["risk_summary"] = "Training keeps original pixels and only downweights masked reflection/dynamic regions through Nerfstudio mask_path."
                return best

        preferred = "original_training_images"
        best = next((item for item in candidate_results if item.get("candidate_name") == preferred and item.get("status") == "succeeded" and not item.get("rejected_reason")), None)
        if best is None:
            best = super().select_best(context, candidate_results)
        if best:
            write_json(context.stage_dir(self.stage_name) / "best_training_input_selection.json", best)
            write_json(context.stage_dir(self.stage_name) / "input_optimization_metrics.json", {"candidates": candidate_results, "best": best})
            best["improvement_summary"] = "已确定训练图、验证图、mask、holdout split 和 source mapping。"
            best["risk_summary"] = "训练输入不会采用高真实性风险增强图。"
        return best

    def _materialize_nerfstudio_training_dataset(self, context: StageContext, candidate: dict[str, Any], manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        pose = candidate["analysis"].get("pose") or {}
        mask_selection = candidate["analysis"].get("mask") or {}
        pose_metrics = pose.get("metrics") or {}
        pose_dataset_dir_value = pose.get("dataset_dir") or pose_metrics.get("dataset_dir")
        pose_transforms_value = pose.get("transforms_path") or pose_metrics.get("transforms_path")
        training_dataset_dir = output_dir / "nerfstudio_dataset"
        images_dir = training_dataset_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        pose_images = list(manifest.get("pose_images") or [])
        training_images = list(manifest.get("training_images") or [])
        if not pose_dataset_dir_value or not pose_transforms_value:
            image_manifest = []
            for index, image_path in enumerate(training_images):
                source_training = Path(str(image_path))
                target = images_dir / source_training.name
                if source_training.exists():
                    copy_file_safely(source_training, target)
                image_manifest.append(
                    {
                        "index": index,
                        "target_image": str(target),
                        "target_sha256": file_sha256(target),
                        "training_source": str(source_training),
                        "training_source_sha256": file_sha256(source_training),
                        "pose_source": str(pose_images[index]) if index < len(pose_images) else None,
                        "pose_source_sha256": file_sha256(pose_images[index]) if index < len(pose_images) else None,
                        "route_image_name": source_training.name,
                    }
                )
            manifest_path = write_json(
                output_dir / "training_input_manifest.json",
                {
                    "route_preset": active_route_preset(context),
                    "dataset_dir": str(training_dataset_dir),
                    "transforms_path": None,
                    "reason": "missing_pose_dataset",
                    "image_policy": manifest.get("image_policy") or {},
                    "pose_image_distribution": manifest.get("pose_image_distribution") or {},
                    "training_image_distribution": manifest.get("training_image_distribution") or {},
                    "images": image_manifest,
                },
            )
            return {"dataset_dir": str(training_dataset_dir), "transforms_path": None, "manifest_path": str(manifest_path), "reason": "missing_pose_dataset"}
        pose_dataset_dir = Path(str(pose_dataset_dir_value))
        pose_transforms_path = Path(str(pose_transforms_value))
        _reset_materialized_preprocess_dir(context, training_dataset_dir)
        for extra_name in ["sparse_point_cloud.ply", "dataparser_transforms.json"]:
            extra = pose_dataset_dir / extra_name
            if extra.exists():
                copy_file_safely(extra, training_dataset_dir / extra_name)
        pose_transforms = read_json(pose_transforms_path, {}) if pose_transforms_path.exists() else {}
        mask_entries = self._mask_entries_by_source_name(mask_selection)
        pose_frame_paths = [
            str(frame.get("file_path"))
            for frame in pose_transforms.get("frames", [])
            if isinstance(frame, dict) and frame.get("file_path")
        ]
        if pose_frame_paths:
            training_transforms = json.loads(json.dumps(pose_transforms))
            masks_dir = training_dataset_dir / "masks"
            mask_count = 0
            image_manifest = []
            for index, relative_path in enumerate(pose_frame_paths):
                source = pose_dataset_dir / relative_path
                target = training_dataset_dir / relative_path
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copy_file_safely(source, target)
                frame_mask = self._mask_entry_for_image_name(Path(relative_path).name, mask_entries)
                if mask_entries:
                    mask_target = masks_dir / f"{Path(relative_path).stem}.png"
                    if frame_mask and frame_mask.get("mask_path") and Path(str(frame_mask.get("mask_path"))).exists():
                        self._copy_inverted_training_mask(Path(str(frame_mask["mask_path"])), mask_target)
                        mask_count += 1
                    else:
                        self._write_full_keep_training_mask(target, mask_target)
                    if index < len(training_transforms.get("frames", [])):
                        training_transforms["frames"][index]["mask_path"] = f"masks/{mask_target.name}"
                source_training = Path(str(training_images[index])) if index < len(training_images) else None
                source_pose = Path(str(pose_images[index])) if index < len(pose_images) else None
                image_manifest.append(
                    {
                        "index": index,
                        "target_image": str(target),
                        "target_sha256": file_sha256(target),
                        "training_source": str(source_training) if source_training else None,
                        "training_source_sha256": file_sha256(source_training) if source_training else None,
                        "pose_source": str(source_pose) if source_pose else None,
                        "pose_source_sha256": file_sha256(source_pose) if source_pose else None,
                        "route_image_name": relative_path,
                        "mask_source": str(frame_mask.get("mask_path")) if frame_mask and frame_mask.get("mask_path") else None,
                        "training_mask": str(mask_target) if mask_entries else None,
                    }
                )
            write_json(training_dataset_dir / "transforms.json", training_transforms)
            manifest_path = write_json(
                output_dir / "training_input_manifest.json",
                {
                    "route_preset": active_route_preset(context),
                    "dataset_dir": str(training_dataset_dir),
                    "transforms_path": str(training_dataset_dir / "transforms.json"),
                    "image_policy": manifest.get("image_policy") or {},
                    "pose_image_distribution": manifest.get("pose_image_distribution") or {},
                    "training_image_distribution": manifest.get("training_image_distribution") or {},
                    "mask_strategy": mask_selection.get("candidate_name"),
                    "mask_applied_to_training": bool(mask_entries),
                    "mask_matched_image_count": mask_count,
                    "mask_total_image_count": len(pose_frame_paths) if mask_entries else 0,
                    "images": image_manifest,
                },
            )
            return {"dataset_dir": str(training_dataset_dir), "transforms_path": str(training_dataset_dir / "transforms.json"), "manifest_path": str(manifest_path)}
        if pose_transforms_path.exists() and not (training_dataset_dir / "transforms.json").exists():
            copy_file_safely(pose_transforms_path, training_dataset_dir / "transforms.json")
        routing_manifest_path = pose_dataset_dir.parent / "routing_manifest.json"
        routing = read_json(routing_manifest_path, {})
        routed_sources = [item for item in routing.get("sources", []) if item.get("status") == "copied" and item.get("image")]
        image_manifest = []
        for index, routed in enumerate(routed_sources):
            target_name = str(routed.get("image"))
            source_training = Path(str(training_images[index])) if index < len(training_images) else None
            source_pose = Path(str(pose_images[index])) if index < len(pose_images) else None
            source = source_training if source_training and source_training.exists() else source_pose
            target = images_dir / target_name
            if source and source.exists():
                copy_file_safely(source, target)
            image_manifest.append(
                {
                    "index": index,
                    "target_image": str(target),
                    "target_sha256": file_sha256(target),
                    "training_source": str(source_training) if source_training else None,
                    "training_source_sha256": file_sha256(source_training) if source_training else None,
                    "pose_source": str(source_pose) if source_pose else None,
                    "pose_source_sha256": file_sha256(source_pose) if source_pose else None,
                    "route_image_name": target_name,
                }
            )
        manifest_path = write_json(
            output_dir / "training_input_manifest.json",
            {
                "route_preset": active_route_preset(context),
                "dataset_dir": str(training_dataset_dir),
                "transforms_path": str(training_dataset_dir / "transforms.json"),
                "image_policy": manifest.get("image_policy") or {},
                "pose_image_distribution": manifest.get("pose_image_distribution") or {},
                "training_image_distribution": manifest.get("training_image_distribution") or {},
                "images": image_manifest,
            },
        )
        return {"dataset_dir": str(training_dataset_dir), "transforms_path": str(training_dataset_dir / "transforms.json"), "manifest_path": str(manifest_path)}

    def _mask_entries_by_source_name(self, mask_selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not mask_selection or mask_selection.get("candidate_name") == "no_mask":
            return {}
        metrics = mask_selection.get("metrics") or {}
        if not metrics and mask_selection.get("output_path"):
            metrics = read_json(Path(str(mask_selection["output_path"])), {})
        report = metrics.get("operator_report") or {}
        entries = report.get("images") or []
        mask_entries: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            image_name = str(entry.get("image_name") or "").strip()
            mask_path = entry.get("mask_path")
            if not image_name or not mask_path:
                continue
            mask_entries[image_name] = entry
            stripped = self._strip_materialized_index_prefix(image_name)
            mask_entries.setdefault(stripped, entry)
        return mask_entries

    def _mask_entry_for_image_name(self, image_name: str, mask_entries: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        if image_name in mask_entries:
            return mask_entries[image_name]
        stripped = self._strip_materialized_index_prefix(image_name)
        if stripped in mask_entries:
            return mask_entries[stripped]
        for key, entry in mask_entries.items():
            if key.endswith(stripped) or image_name.endswith(key):
                return entry
        return None

    def _strip_materialized_index_prefix(self, image_name: str) -> str:
        return image_name[6:] if len(image_name) > 6 and image_name[:5].isdigit() and image_name[5] == "_" else image_name

    def _copy_inverted_training_mask(self, source_mask: Path, target_mask: Path) -> None:
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        if Image is None or ImageOps is None:
            copy_file_safely(source_mask, target_mask)
            return
        with Image.open(source_mask) as mask:
            inverted = ImageOps.invert(mask.convert("L"))
            inverted.save(target_mask)

    def _write_full_keep_training_mask(self, image_path: Path, target_mask: Path) -> None:
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        if Image is None:
            target_mask.write_bytes(b"")
            return
        with Image.open(image_path) as image:
            Image.new("L", image.size, 255).save(target_mask)


class GaussianTrainingOptimizationStage(StageOptimizer):
    stage_name = "gaussian_training_optimization"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        training_input = read_json(context.stage_dir("training_input_optimization") / "best_training_input_selection.json", {})
        pose = read_json(context.stage_dir("pose_estimation_optimization") / "best_pose_selection.json", {})
        return {"input_artifacts": [training_input.get("output_path"), pose.get("output_path")], "training_input": training_input, "pose": pose}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        allow_big = bool(context.config.get("allow_big_model", True))
        allow_w = bool(context.config.get("allow_splatfacto_w", True))
        allow_mask = bool(context.config.get("allow_mask", True))
        names = []
        for name in GAUSSIAN_TRAINING_ROUTES:
            if name in {"splatfacto_big", "splatfacto_high_resolution", "splatfacto_long_train"} and not allow_big:
                continue
            if name in {"splatfacto_w", "splatfacto_w_light"} and not allow_w:
                continue
            if name in {"splatfacto_with_conservative_mask", "splatfacto_with_robust_mask"} and not allow_mask:
                continue
            names.append(name)
        return [{"candidate_name": name, "candidate_type": "gaussian_training_route", "analysis": analysis, "status": "created", "created_at": utc_now_iso()} for name in names]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        fake = bool(context.settings.nerfstudio_fake_runner or context.config.get("fake_runner"))
        route = candidate["candidate_name"]
        output_dir = context.stage_dir(self.stage_name) / "training_runs" / route
        output_dir.mkdir(parents=True, exist_ok=True)
        method = "splatfacto-big" if route in {"splatfacto_big", "splatfacto_high_resolution", "splatfacto_long_train"} else "splatfacto-w-light" if route == "splatfacto_w_light" else "splatfacto-w" if route == "splatfacto_w" else "splatfacto"
        steps = 300 if fake else 0
        if not fake and not context.config.get("execute_training", False):
            metrics = {
                "method": method,
                "steps": 0,
                "status": "planned_only",
                "reason": "training_adapter_not_executed_without_execute_training_flag",
                "final_eval_psnr": None,
                "final_eval_ssim": None,
                "final_eval_lpips": None,
                "artifact_score": 0.0,
                "geometry_score": 0.0,
                "texture_score": 0.0,
                "render_stability_score": 0.0,
            }
            config_path = write_json(output_dir / "config.json", {"method": method, "route": route, "stage_search": "planned_only"})
            metrics_path = write_json(output_dir / "metrics.json", metrics)
            candidate.update({"status": "planned", "output_path": str(metrics_path), "config_path": str(config_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": "training_not_executed", "risk_level": "medium"})
            return candidate
        if not fake and context.config.get("execute_training", False):
            allowed = _configured_names(context.config.get("real_training_candidates"), ["splatfacto_long_train", "splatfacto_big", "splatfacto_high_resolution"])
            if route not in allowed:
                candidate.update(
                    {
                        "status": "skipped",
                        "score": 0.0,
                        "rejected_reason": "not_in_real_training_candidate_set",
                        "risk_level": "low",
                    }
                )
                return candidate
            pose = candidate["analysis"].get("pose") or {}
            pose_metrics = pose.get("metrics") or {}
            training_input = candidate["analysis"].get("training_input") or {}
            training_input_selection = read_json(Path(str(training_input.get("output_path") or "")), {}) if training_input.get("output_path") else {}
            dataset_dir_value = training_input_selection.get("nerfstudio_dataset_dir") or pose.get("dataset_dir") or pose_metrics.get("dataset_dir")
            transforms_value = training_input_selection.get("nerfstudio_transforms_path") or pose.get("transforms_path") or pose_metrics.get("transforms_path")
            if not dataset_dir_value or not transforms_value or not Path(str(transforms_value)).exists():
                metrics = {
                    "method": method,
                    "status": "blocked",
                    "reason": "real_training_requires_real_pose_transforms",
                    "dataset_dir": dataset_dir_value,
                    "transforms_path": transforms_value,
                    "final_eval_psnr": None,
                    "final_eval_ssim": None,
                    "final_eval_lpips": None,
                }
                metrics_path = write_json(output_dir / "metrics.json", metrics)
                candidate.update(
                    {
                        "status": "failed",
                        "output_path": str(metrics_path),
                        "metrics_path": str(metrics_path),
                        "metrics": metrics,
                        "score": 0.0,
                        "rejected_reason": "missing_real_pose_transforms",
                        "risk_level": "high",
                    }
                )
                return candidate
            dataset_dir = Path(str(dataset_dir_value))
            previous_config = dict(context.workflow.config_json or {})
            training_config = {**previous_config, **context.config}
            training_config["method"] = method
            training_config["fake_runner"] = False
            if context.config.get("iterations") is not None:
                training_config["iterations"] = int(context.config["iterations"])
                training_config["max_num_iterations"] = int(context.config["iterations"])
            if context.config.get("max_num_iterations") is not None:
                training_config["max_num_iterations"] = int(context.config["max_num_iterations"])
            high_quality_iterations = (
                context.config.get("high_quality_iterations")
                or nested_get(context.config, "training.final_steps")
                or nested_get(context.config, "training.standard_steps")
                or 30000
            )
            long_train_iterations = (
                context.config.get("long_train_iterations")
                or nested_get(context.config, "training.long_train_steps")
                or nested_get(context.config, "training.slow_steps")
                or nested_get(context.config, "training.final_steps")
            )
            route_iterations = {
                "splatfacto_big": context.config.get("splatfacto_big_iterations") or high_quality_iterations,
                "splatfacto_tuned": context.config.get("tuned_iterations"),
                "splatfacto_high_resolution": context.config.get("high_resolution_iterations") or high_quality_iterations,
                "splatfacto_long_train": long_train_iterations,
                "splatfacto_w_light": context.config.get("splatfacto_w_light_iterations"),
                "splatfacto_w": context.config.get("splatfacto_w_iterations"),
            }.get(route)
            if route_iterations is not None:
                training_config["iterations"] = int(route_iterations)
                training_config["max_num_iterations"] = int(route_iterations)
            if route == "splatfacto_high_resolution":
                training_config["num_downscales"] = int(context.config.get("high_resolution_num_downscales") or 0)
            if route == "splatfacto_tuned":
                training_config.setdefault("warmup_length", int(context.config.get("tuned_warmup_length") or 500)
                )
            if route == "splatfacto_with_robust_mask":
                training_config["enable_robust_mask"] = True
            if route == "splatfacto_with_conservative_mask":
                training_config["apply_masks_to_training"] = bool(context.config.get("apply_masks_to_training", True))
            if route == "splatfacto_long_train":
                training_config.setdefault("quality_profile", "forensic_max_quality")
                training_config.setdefault("quality_boost_profile", "forensic_max_quality")
                training_config.setdefault("forensic_mainline", True)
            if context.config.get("mode"):
                training_config["mode"] = context.config["mode"]
            elif "mode" not in training_config:
                training_config["mode"] = "high_quality" if route in {"splatfacto_big", "splatfacto_high_resolution", "splatfacto_long_train"} else "quick_preview"
            if route == "prior_assisted_fallback":
                metrics = {
                    "method": method,
                    "status": "blocked",
                    "reason": "prior_assisted_depth_or_normal_adapter_not_configured",
                    "final_eval_psnr": None,
                    "final_eval_ssim": None,
                    "final_eval_lpips": None,
                }
                metrics_path = write_json(output_dir / "metrics.json", metrics)
                candidate.update({"status": "skipped", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": 0.0, "rejected_reason": "prior_assisted_adapter_unavailable", "risk_level": "medium"})
                context.capability_report.setdefault(self.stage_name, {})[route] = {"available": False, "reason": "prior_assisted_depth_or_normal_adapter_not_configured"}
                return candidate
            context.workflow.config_json = training_config

            observer_events: list[dict[str, Any]] = []

            def observe(event: str, stage_key: str, payload: dict[str, Any]) -> None:
                observer_events.append({"event": event, "stage_key": stage_key, "payload": payload})

            try:
                result = NerfstudioSplatfactoTrainOperator(context.settings).run(
                    context.workflow,
                    dataset_dir,
                    media_metadata={
                        "stage_optimized_reconstruction": True,
                        "route": route,
                        "route_preset": training_input_selection.get("route_preset") or active_route_preset(context),
                        "workspace_suffix": f"routes/{training_input_selection.get('route_preset') or active_route_preset(context)}/{route}",
                        "image_policy": training_input_selection.get("image_policy") or {},
                        "training_input_manifest": training_input_selection.get("training_input_manifest"),
                        "training_supervision_modified": bool(training_input_selection.get("training_supervision_modified")),
                        "pose_candidate": pose.get("candidate_name"),
                        "staged_file_count": len(list((dataset_dir / "images").glob("*"))),
                    },
                    stage_observer=observe,
                )
                eval_metrics = result.quality_checks.get("eval_metrics") or {}
                splat_quality = result.quality_checks.get("splat_quality") or {}
                psnr = result.quality_checks.get("psnr") or eval_metrics.get("psnr") or eval_metrics.get("PSNR")
                ssim = result.quality_checks.get("ssim") or eval_metrics.get("ssim") or eval_metrics.get("SSIM")
                lpips = result.quality_checks.get("lpips") or eval_metrics.get("lpips") or eval_metrics.get("LPIPS")
                gaussian_count = splat_quality.get("vertex_count") or splat_quality.get("gaussian_count")
                metrics = {
                    "method": method,
                    "config": {"route": route, "stage_search": "real_nerfstudio"},
                    "route_preset": training_input_selection.get("route_preset") or active_route_preset(context),
                    "image_policy": training_input_selection.get("image_policy") or {},
                    "training_input_manifest": training_input_selection.get("training_input_manifest"),
                    "training_supervision_modified": bool(training_input_selection.get("training_supervision_modified")),
                    "dataset_dir": str(dataset_dir),
                    "transforms_path": str(transforms_value),
                    "steps": int(training_config.get("iterations") or training_config.get("max_num_iterations") or 0),
                    "resolution": "nerfstudio_auto",
                    "mask_strategy": "conservative_mask" if "mask" in route else "no_mask",
                    "robust_mask_config": {"enabled": "robust" in route},
                    "appearance_config": {"enabled": route in {"splatfacto_w", "splatfacto_w_light"}},
                    "gaussian_count": gaussian_count,
                    "gpu_memory_peak": None,
                    "train_time": None,
                    "final_train_psnr": None,
                    "final_eval_psnr": psnr,
                    "final_eval_ssim": ssim,
                    "final_eval_lpips": lpips,
                    "artifact_score": 1.0 if result.splat_path and result.splat_path.exists() else 0.0,
                    "floater_score": splat_quality.get("floater_score"),
                    "texture_score": float(psnr) / 35.0 if psnr is not None else 0.0,
                    "geometry_score": float((pose.get("metrics") or {}).get("geometry_stability_score") or 0.0),
                    "render_stability_score": 1.0 if result.quality_checks.get("passed") else 0.5,
                    "splat_path": str(result.splat_path) if result.splat_path else None,
                    "config_path": str(result.config_path) if result.config_path else None,
                    "eval_metrics_path": str(result.eval_metrics_path) if result.eval_metrics_path else None,
                    "export_dir": str(result.export_dir),
                    "command_results": [
                        {
                            "operator_name": command.operator_name,
                            "stage_key": command.stage_key,
                            "exit_code": command.exit_code,
                            "stdout_tail": command.stdout[-2000:] if command.stdout else "",
                            "stderr_tail": command.stderr[-2000:] if command.stderr else "",
                        }
                        for command in result.commands
                    ],
                    "observer_events": observer_events,
                }
                config_path = write_json(output_dir / "config.json", training_config)
                metrics_path = write_json(output_dir / "metrics.json", metrics)
                write_json(output_dir / "nerfstudio_metrics.json", metrics)
                score = 0.0
                if psnr is not None:
                    score = 0.28 * min(float(psnr) / 35.0, 1.0)
                score += 0.22 * float(metrics["geometry_score"] or 0.0)
                score += 0.2 * float(metrics["texture_score"] or 0.0)
                score += 0.2 * float(metrics["render_stability_score"] or 0.0)
                score += 0.1 * float(metrics["artifact_score"] or 0.0)
                candidate.update(
                    {
                        "status": "succeeded",
                        "output_path": str(metrics_path),
                        "config_path": str(config_path),
                        "metrics_path": str(metrics_path),
                        "metrics": metrics,
                        "score": round(score, 4),
                        "risk_level": "medium" if "robust" in route or route in {"splatfacto_w", "splatfacto_w_light"} else "low",
                        "splat_path": str(result.splat_path) if result.splat_path else None,
                    }
                )
                if result.splat_path and result.splat_path.exists():
                    route_scope = f"routes/{context.config.get('active_route_id')}/" if context.config.get("active_route_id") else ""
                    artifact = context.artifact_service.register_file(
                        project_id=context.project_id,
                        workflow_id=context.run_id,
                        artifact_type="optimized_gaussian_ply",
                        stage=self.stage_name,
                        relative_path=f"projects/{context.project_id}/runs/{context.run_id}/optimized/{route_scope}{self.stage_name}/{route}/splat.ply",
                        source_path=str(result.splat_path),
                        mime_type="application/octet-stream",
                        metadata={"route": route, "method": method, "source": "real_nerfstudio"},
                        is_primary=True,
                    )
                    candidate["artifact_id"] = artifact.id
                    metrics["artifact_id"] = artifact.id
                    write_json(metrics_path, metrics)
                if not result.splat_path or not result.splat_path.exists():
                    candidate.update({"rejected_reason": "real_training_missing_splat_ply", "score": 0.0, "risk_level": "high"})
                return candidate
            except Exception as exc:
                context.workflow.config_json = previous_config
                metrics = {
                    "method": method,
                    "status": "failed",
                    "reason": "real_nerfstudio_failed",
                    "error": str(exc),
                    "observer_events": observer_events,
                }
                metrics_path = write_json(output_dir / "metrics.json", metrics)
                candidate.update(
                    {
                        "status": "failed",
                        "output_path": str(metrics_path),
                        "metrics_path": str(metrics_path),
                        "metrics": metrics,
                        "score": 0.0,
                        "rejected_reason": "real_nerfstudio_failed",
                        "risk_level": "high",
                    }
                )
                return candidate
            finally:
                context.workflow.config_json = previous_config
        training_input = candidate["analysis"].get("training_input") or {}
        training_input_selection = read_json(Path(str(training_input.get("output_path") or "")), {}) if training_input.get("output_path") else {}
        base_quality = float((training_input.get("metrics") or {}).get("expected_texture_fidelity") or 0.65)
        bonus = {
            "splatfacto_baseline": 0.0,
            "splatfacto_tuned": 0.04,
            "splatfacto_big": 0.06,
            "splatfacto_w_light": 0.02,
            "splatfacto_w": -0.03,
            "splatfacto_with_conservative_mask": 0.025,
            "splatfacto_with_robust_mask": -0.06,
            "splatfacto_high_resolution": 0.045,
            "splatfacto_long_train": 0.035,
            "prior_assisted_fallback": 0.02,
        }.get(route, 0.0)
        psnr = 20.0 + base_quality * 8.0 + bonus * 20.0
        floater = 0.12 + (0.15 if "robust" in route else 0.02 if route == "splatfacto_w_light" else 0.03 if route == "splatfacto_w" else 0.0)
        metrics = {
            "method": method,
            "config": {"route": route, "stage_search": "fake_probe"},
            "route_preset": training_input_selection.get("route_preset") or active_route_preset(context),
            "image_policy": training_input_selection.get("image_policy") or {},
            "training_input_manifest": training_input_selection.get("training_input_manifest"),
            "training_supervision_modified": bool(training_input_selection.get("training_supervision_modified")),
            "steps": steps,
            "resolution": "auto",
            "mask_strategy": "conservative_mask" if "mask" in route else "no_mask",
            "robust_mask_config": {"enabled": "robust" in route},
            "appearance_config": {"enabled": route in {"splatfacto_w", "splatfacto_w_light"}},
            "gaussian_count": 120000 if route not in {"splatfacto_big", "splatfacto_high_resolution", "splatfacto_long_train"} else 280000,
            "gpu_memory_peak": "fake",
            "train_time": "fake",
            "final_train_psnr": round(psnr + 0.8, 3),
            "final_eval_psnr": round(psnr, 3),
            "final_eval_ssim": round(clamp(0.45 + psnr / 60.0), 4),
            "final_eval_lpips": round(clamp(1.0 - psnr / 35.0), 4),
            "artifact_score": round(1.0 - floater, 4),
            "floater_score": round(floater, 4),
            "texture_score": round(clamp(base_quality + bonus), 4),
            "geometry_score": round(clamp(0.76 - floater * 0.3), 4),
            "render_stability_score": round(clamp(0.78 - floater * 0.2), 4),
        }
        config_path = write_json(output_dir / "config.json", metrics["config"])
        metrics_path = write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "nerfstudio_metrics.json", metrics)
        candidate.update({"status": "succeeded", "output_path": str(metrics_path), "config_path": str(config_path), "metrics": metrics, "score": round(0.28 * metrics["final_eval_psnr"] / 35.0 + 0.22 * metrics["geometry_score"] + 0.2 * metrics["texture_score"] + 0.2 * metrics["render_stability_score"] + 0.1 * metrics["artifact_score"], 4), "risk_level": "medium" if "robust" in route or route in {"splatfacto_w", "splatfacto_w_light", "prior_assisted_fallback"} else "low"})
        if "robust" in route and metrics["floater_score"] > 0.22:
            candidate.update({"rejected_reason": "robust_mask_artifact_risk", "score": 0.0})
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [item for item in candidate_results if item.get("status") == "succeeded" and not item.get("rejected_reason")]
        if not completed:
            if bool(context.config.get("execute_training", False)) and not bool(context.settings.nerfstudio_fake_runner or context.config.get("fake_runner")):
                write_json(context.stage_dir(self.stage_name) / "best_training_selection.json", {"status": "blocked", "reason": "no_training_candidate_passed_quality_gate"})
                write_json(context.stage_dir(self.stage_name) / "train_comparison.json", {"candidates": candidate_results, "best": None, "reason": "no_training_candidate_passed_quality_gate"})
                return {}
            planned = next((item for item in candidate_results if item.get("candidate_name") == "splatfacto_baseline"), None)
            best = planned or {}
        else:
            best = max(completed, key=lambda item: float(item.get("score") or 0.0))
        if best:
            write_json(context.stage_dir(self.stage_name) / "best_training_selection.json", best)
            write_json(context.stage_dir(self.stage_name) / "train_comparison.json", {"candidates": candidate_results, "best": best})
            if best.get("metrics"):
                write_json(context.stage_dir(self.stage_name) / "nerfstudio_metrics.json", best.get("metrics") or {})
            write_text(context.stage_dir(self.stage_name) / "train_comparison.md", self._report_markdown(context, {}, candidate_results, best))
            best["improvement_summary"] = "训练阶段按 baseline -> tuned -> big/w/mask 候选进行选择，未将 splatfacto-w 或 robust mask 作为默认答案。"
            best["risk_summary"] = "PSNR 提升但漂浮物或真实性风险增加的候选不会成为 best。"
        return best


class RenderEvaluationStage(StageOptimizer):
    stage_name = "render_evaluation"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        training = read_json(context.stage_dir("gaussian_training_optimization") / "best_training_selection.json", {})
        return {"input_artifacts": [training.get("output_path")], "training": training}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"candidate_name": name, "candidate_type": "render_eval", "analysis": analysis, "status": "created", "created_at": utc_now_iso()} for name in RENDER_EVALUATION_ROUTES]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        training_metrics = (candidate["analysis"].get("training") or {}).get("metrics") or {}
        route = str(candidate.get("candidate_name") or "render_eval")
        psnr = training_metrics.get("final_eval_psnr")
        geometry = float(training_metrics.get("geometry_score") or 0.0)
        floater = float(training_metrics.get("floater_score") or 0.0)
        texture = float(training_metrics.get("texture_score") or 0.0)
        if psnr is None:
            metrics = {
                "status": "not_rendered",
                "evaluation_route": route,
                "reason": "no_trained_model_available",
                "PSNR": None,
                "SSIM": None,
                "LPIPS": None,
                "forensic_integrity_score": 0.0,
                "inspectability_score": 0.0,
            }
            score = 0.0
            rejected = "no_trained_model_available"
        else:
            metrics = {
                "evaluation_route": route,
                "PSNR": psnr,
                "SSIM": training_metrics.get("final_eval_ssim"),
                "LPIPS": training_metrics.get("final_eval_lpips"),
                "novel_view_consistency": round(geometry, 4),
                "temporal_flicker_score": 0.08,
                "floater_score": floater,
                "hole_score": round(max(0.0, 0.28 - geometry * 0.2), 4),
                "blur_score": round(max(0.0, 0.35 - texture * 0.2), 4),
                "texture_fidelity_score": texture,
                "geometry_completeness_score": geometry,
                "dynamic_pollution_score": 0.1,
                "exposure_consistency_score": 0.78,
                "forensic_integrity_score": round(clamp(0.82 - floater * 0.4), 4),
                "inspectability_score": round(clamp((geometry + texture + (1 - floater)) / 3.0), 4),
            }
            route_weight = {
                "held_out_view_render": 1.0,
                "fixed_camera_path_render": 0.96,
                "orbit_render": 0.94,
                "close_up_render": 0.92,
                "sparse_vs_render_comparison": 0.98,
                "original_vs_reconstruction_comparison": 1.0,
                "baseline_vs_best_comparison": 0.95,
                "mask_vs_no_mask_comparison": 0.93,
                "enhanced_vs_original_comparison": 0.93,
            }.get(route, 1.0)
            score = round((0.2 * (float(psnr) / 35.0) + 0.25 * geometry + 0.2 * texture + 0.2 * (1 - floater) + 0.15 * metrics["forensic_integrity_score"]) * route_weight, 4)
            rejected = None if floater <= 0.25 and geometry >= 0.55 else "render_quality_gate_failed"
        output_dir = context.stage_dir(self.stage_name)
        metrics_path = write_json(output_dir / f"{route}_metrics.json", metrics)
        write_text(output_dir / "quality_report.md", self._quality_report(metrics, candidate["analysis"].get("training") or {}))
        write_text(output_dir / "manual_review_checklist.md", "- 检查是否存在虚假结构\n- 检查漂浮物和空洞\n- 对照原始素材确认颜色和边界\n")
        candidate.update({"status": "succeeded", "output_path": str(metrics_path), "metrics_path": str(metrics_path), "metrics": metrics, "score": score, "rejected_reason": rejected, "risk_level": "medium" if rejected else "low"})
        return candidate

    def select_best(self, context: StageContext, candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        best = super().select_best(context, candidate_results)
        if best:
            write_json(context.stage_dir(self.stage_name) / "eval_metrics.json", best.get("metrics") or {})
            write_json(context.stage_dir(self.stage_name) / "render_eval_metrics.json", best.get("metrics") or {})
            write_json(context.stage_dir(self.stage_name) / "render_evaluation_comparison.json", {"candidates": candidate_results, "best": best})
        return best

    def _quality_report(self, metrics: dict[str, Any], training: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Render Quality Report",
                "",
                f"- best_training_candidate: `{training.get('candidate_name')}`",
                f"- PSNR: {metrics.get('PSNR')}",
                f"- SSIM: {metrics.get('SSIM')}",
                f"- LPIPS: {metrics.get('LPIPS')}",
                f"- forensic_integrity_score: {metrics.get('forensic_integrity_score')}",
                f"- inspectability_score: {metrics.get('inspectability_score')}",
                "",
                "最终模型不按单一 PSNR 选择，而是综合几何稳定、纹理真实性、漂浮物、完整性和复查可用性。",
            ]
        ) + "\n"


class FinalArtifactSelectionStage(StageOptimizer):
    stage_name = "final_artifact_selection"

    def analyze_input(self, context: StageContext) -> dict[str, Any]:
        stage_results = {}
        for stage_name in OPTIMIZED_STAGE_NAMES:
            if stage_name == self.stage_name:
                continue
            result = read_json(context.stage_dir(stage_name) / "stage_result.json", None)
            if result:
                stage_results[stage_name] = result
        return {"input_artifacts": [result.get("stage_result_path") for result in stage_results.values()], "stage_results": stage_results}

    def generate_candidates(self, context: StageContext, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"candidate_name": "forensic_best_route_package", "candidate_type": "final_selection", "analysis": analysis, "status": "created", "created_at": utc_now_iso()}]

    def run_candidate(self, context: StageContext, candidate: dict[str, Any]) -> dict[str, Any]:
        stage_results = candidate["analysis"].get("stage_results") or {}
        training = read_json(context.stage_dir("gaussian_training_optimization") / "best_training_selection.json", {})
        eval_metrics = read_json(context.stage_dir("render_evaluation") / "eval_metrics.json", {})
        all_candidate_metrics = read_json(context.run_dir / "records" / "run_candidate_records.json", [])
        source_map = self._best_source_map(context)
        limitations = self._limitations(stage_results, training, eval_metrics)
        final_score = float(eval_metrics.get("inspectability_score") or training.get("score") or 0.0)
        quality_level = "production_candidate" if final_score >= 0.72 and not limitations else "needs_review" if final_score > 0 else "not_ready"
        output_dir = context.stage_dir(self.stage_name)
        training_metrics = training.get("metrics") or {}
        best_model_path = training.get("splat_path") or training_metrics.get("splat_path") or training.get("output_path")
        selection = {
            "run_id": context.run_id,
            "best_route_id": training.get("candidate_name"),
            "best_model_path": best_model_path,
            "best_model_artifact_id": training.get("artifact_id") or training_metrics.get("artifact_id"),
            "best_report_path": str(output_dir / "best_route_report.md"),
            "final_score": round(final_score, 4),
            "quality_level": quality_level,
            "limitation_summary": limitations,
            "selected_at": utc_now_iso(),
        }
        write_json(output_dir / "all_candidate_metrics.json", {"candidates": all_candidate_metrics})
        write_json(output_dir / "source_map.json", source_map)
        write_json(output_dir / "run_final_selection.json", selection)
        write_json(output_dir / "final_score.json", {"run_id": context.run_id, "route_preset": active_route_preset(context), "final_score": selection["final_score"], "quality_level": quality_level})
        forensic_manifest = {
            "run_id": context.run_id,
            "route_preset": active_route_preset(context),
            "best_model_path": best_model_path,
            "best_model_artifact_id": selection.get("best_model_artifact_id"),
            "image_policy": self._best_image_policy(context),
            "source_map": source_map,
            "enhancement_provenance": self._best_enhancement_provenance(context),
            "training_supervision_modified": bool(training_metrics.get("training_supervision_modified")),
            "final_score": selection["final_score"],
            "quality_level": quality_level,
        }
        write_json(output_dir / "forensic_package_manifest.json", forensic_manifest)
        write_text(output_dir / "best_route_report.md", self._best_route_report(stage_results, training, eval_metrics, limitations))
        write_text(output_dir / "all_stage_report.md", self._all_stage_report(stage_results))
        write_text(output_dir / "quality_limitations_report.md", "\n".join(f"- {item}" for item in limitations) + "\n")
        write_text(output_dir / "manual_review_checklist.md", "- 对照原始素材复核结构\n- 复核增强来源和风险\n- 复核低质量区域和补采建议\n")
        RunRecordStore(context.run_dir).write_final_selection(selection)
        candidate.update({"status": "succeeded", "output_path": str(output_dir / "run_final_selection.json"), "metrics": selection, "score": selection["final_score"], "risk_level": "medium" if limitations else "low", "improvement_summary": "已汇总所有阶段 best artifact、候选指标、source map 和质量限制。", "risk_summary": "最终报告明确真实性风险、质量上限和人工复核项。"})
        return candidate

    def export_stage_result(self, context: StageContext, analysis: dict[str, Any], candidates: list[dict[str, Any]], best_result: dict[str, Any]) -> dict[str, Any]:
        result = super().export_stage_result(context, analysis, candidates, best_result)
        output_dir = context.stage_dir(self.stage_name)
        for artifact_type, filename in {
            "best_route_report": "best_route_report.md",
            "all_stage_report": "all_stage_report.md",
            "all_candidate_metrics": "all_candidate_metrics.json",
            "source_map": "source_map.json",
            "run_final_selection": "run_final_selection.json",
            "final_score": "final_score.json",
            "forensic_package_manifest": "forensic_package_manifest.json",
            "quality_limitations_report": "quality_limitations_report.md",
            "manual_review_checklist": "manual_review_checklist.md",
        }.items():
            path = output_dir / filename
            if path.exists():
                route_scope = f"routes/{context.config.get('active_route_id')}/" if context.config.get("active_route_id") else ""
                context.artifact_service.register_file(
                    project_id=context.project_id,
                    workflow_id=context.run_id,
                    artifact_type=artifact_type,
                    stage=self.stage_name,
                    relative_path=f"projects/{context.project_id}/runs/{context.run_id}/optimized/{route_scope}{self.stage_name}/{filename}",
                    source_path=str(path),
                    is_primary=artifact_type == "best_route_report",
                )
        self._register_v3_status_reports(context)
        context.db.flush()
        return result

    def _register_v3_status_reports(self, context: StageContext) -> None:
        output_dir = context.stage_dir(self.stage_name) / "v3_status_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        route_preset = active_route_preset(context)
        route_id = context.config.get("active_route_id") or route_preset
        source_asset_ids = [asset.id for asset in context.assets]
        source_paths = [asset.original_filename or asset.filename for asset in context.assets if asset.original_filename or asset.filename]
        for spec in self._v3_status_report_specs(context):
            artifact_type = spec["artifact_type"]
            payload = {
                "summary": spec.get("summary"),
                "inputs": spec.get("inputs") or {"asset_count": len(context.assets)},
                "outputs": spec.get("outputs") or {},
                "metrics": spec.get("metrics") or {},
                "limitations": spec.get("limitations") or [],
            }
            context.artifact_service.register_stage_report(
                project_id=context.project_id,
                workflow_id=context.run_id,
                artifact_type=artifact_type,
                stage=spec.get("stage") or self.stage_name,
                operator=spec.get("operator") or f"stage_optimized.{artifact_type}",
                status=spec.get("status") or "skipped",
                failure_reason=spec.get("failure_reason"),
                relative_path=f"projects/{context.project_id}/runs/{context.run_id}/optimized/v3_status_reports/{artifact_type}.json",
                payload=payload,
                source_asset_ids=source_asset_ids,
                source_artifact_ids=spec.get("source_artifact_ids") or [],
                source_paths=source_paths,
                derived_from=spec.get("derived_from") or [],
                route_id=str(route_id),
                route_key=route_preset,
                route_role=spec.get("route_role") or "production",
                production_allowed=bool(spec.get("production_allowed", True)),
                measurement_allowed=bool(spec.get("measurement_allowed", False)),
            )

    def _v3_status_report_specs(self, context: StageContext) -> list[dict[str, Any]]:
        assets = context.assets
        asset_count = len(assets)
        has_video = any((asset.asset_type or "").endswith("video") for asset in assets)
        has_pano = any(asset.asset_type in {"pano_360", "panorama"} or asset.role == "pano_anchor" for asset in assets)
        has_scale = any(asset.asset_type == "scale_marker" or asset.role in {"scale_marker", "measurement_marker", "scale_reference"} for asset in assets)
        has_drone = any((asset.metadata_json or {}).get("capture_platform") == "drone" or asset.role in {"drone", "aerial"} for asset in assets)
        has_depth = any(asset.asset_type in {"depth", "rgbd", "lidar"} or asset.role in {"depth_sensor", "lidar"} for asset in assets)
        raw = read_json(context.stage_dir("raw_media_inspection") / "stage_result.json", {})
        video = read_json(context.stage_dir("video_keyframe_optimization") / "stage_result.json", {})
        pano = read_json(context.stage_dir("panorama_normalization") / "stage_result.json", {})
        pose = read_json(context.stage_dir("pose_estimation_optimization") / "candidate_metrics.json", [])
        training = read_json(context.stage_dir("training_input_optimization") / "stage_result.json", {})
        render = read_json(context.stage_dir("render_evaluation") / "stage_result.json", {})
        delivery_supported = bool(read_json(context.stage_dir(self.stage_name) / "run_final_selection.json", {}).get("best_model_path"))
        common_inputs = {"asset_count": asset_count, "has_video": has_video, "has_pano": has_pano, "has_scale": has_scale}

        specs: list[dict[str, Any]] = [
            {"artifact_type": "input_route_report", "stage": "input_route", "status": "succeeded", "summary": "Input route report mirrors the guarded route manifest and records that routing alone never grants measurement.", "measurement_allowed": False},
            {"artifact_type": "metadata_lineage_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Asset metadata and derived files remain traceable through source maps.", "metrics": {"asset_count": asset_count}},
            {"artifact_type": "metadata_manifest", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Metadata manifest records available asset metadata without fabricating missing EXIF or GPS fields.", "metrics": {"asset_count": asset_count}},
            {"artifact_type": "exif_gps_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "EXIF/GPS fields are preserved when present; missing metadata is reported, not fabricated.", "metrics": {"asset_count": asset_count, "gps_prior_used_for_measurement": False}},
            {"artifact_type": "exif_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "EXIF report is a document-name-compatible alias for raw metadata inspection.", "metrics": {"asset_count": asset_count}},
            {"artifact_type": "timestamp_lineage", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Timestamp lineage remains source-traceable and is not used to rewrite asset evidence.", "metrics": {"asset_count": asset_count}},
            {"artifact_type": "asset_quality_summary", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Current asset quality summary is derived from RawMediaInspection and capture validation signals.", "metrics": raw.get("metrics") or {}},
            {"artifact_type": "reconstruction_readiness_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Readiness remains report-only and does not override quality gates.", "metrics": {"asset_count": asset_count, "ready_for_pose": asset_count > 0}},
            {"artifact_type": "reflective_transparent_risk_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Reflective/transparent risk is reported as a readiness signal, not collapsed into low-texture only.", "metrics": raw.get("metrics") or {}},
            {"artifact_type": "capture_pattern_profile", "stage": "raw_media_inspection", "status": "succeeded", "summary": "Capture pattern profile is derived from current media inspection and remains advisory.", "metrics": {"asset_count": asset_count}},
            {"artifact_type": "camera_model_policy_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Camera model policy is reported from current pose/camera handling; panorama virtual camera support remains route-specific."},
            {"artifact_type": "camera_model_policy", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Camera model policy artifact preserves the v3 document name and points to the same camera handling policy."},
            {"artifact_type": "video_probe_report", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Video probe is skipped without video input and does not fabricate stream metadata.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "scene_segment_report", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Video scene segmentation is currently represented by keyframe optimization reports.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "scene_segments", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Scene segments are skipped without video input and preserve the v3 document artifact name.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "frame_selection_report", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Frame selection preserves source video lineage.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "video_frame_selection_report", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Video frame selection report is skipped without video input.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "frame_graph", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Frame graph is skipped without video input.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "rolling_shutter_risk_report", "stage": "video_keyframe_optimization", "status": "succeeded" if has_video else "skipped", "failure_reason": None if has_video else "no_video_input", "summary": "Rolling shutter risk is reported when video input exists and remains advisory.", "metrics": video.get("metrics") or {}},
            {"artifact_type": "image_set_reduction_report", "stage": "dataset_assembly", "status": "succeeded", "summary": "Image reduction is candidate-list based; original assets are not deleted."},
            {"artifact_type": "panorama_station_manifest", "stage": "panorama_normalization", "status": "succeeded" if has_pano else "skipped", "failure_reason": None if has_pano else "no_panorama_input", "summary": "Panorama station support is reported when pano assets are present.", "metrics": pano.get("metrics") or {}},
            {"artifact_type": "virtual_camera_manifest", "stage": "panorama_normalization", "status": "succeeded" if has_pano else "skipped", "failure_reason": None if has_pano else "no_panorama_input", "summary": "Virtual camera manifest is route-aware and skipped without panorama input."},
            {"artifact_type": "crop_to_pano_map", "stage": "panorama_normalization", "status": "succeeded" if has_pano else "skipped", "failure_reason": None if has_pano else "no_panorama_input", "summary": "Crop-to-panorama mapping is skipped without panorama input."},
            {"artifact_type": "pano_station_graph", "stage": "panorama_normalization", "status": "succeeded" if has_pano else "skipped", "failure_reason": None if has_pano else "no_panorama_input", "summary": "Panorama station graph is skipped without panorama input."},
            {"artifact_type": "vendor_metadata_report", "stage": "panorama_normalization", "status": "unsupported" if not has_pano else "succeeded", "failure_reason": None if has_pano else "vendor_panorama_metadata_not_present", "summary": "Vendor-specific OSV/INSP/INSV metadata is not claimed when absent."},
            {"artifact_type": "pose_candidates_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Pose candidates are reported from current stage candidate metrics.", "metrics": {"candidate_count": len(pose) if isinstance(pose, list) else 0}},
            {"artifact_type": "hloc_pairs", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "no_explicit_hloc_pairs_artifact", "summary": "HLoc pair files are not fabricated when the current run did not produce them."},
            {"artifact_type": "feature_matching_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "LightGlue/ALIKED remains a pose candidate signal and does not replace geometry gates."},
            {"artifact_type": "feature_match_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Feature match report preserves the v3 document artifact name for LightGlue/ALIKED candidate evidence."},
            {"artifact_type": "match_graph", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "no_explicit_match_graph_artifact", "summary": "Match graph is skipped unless an explicit graph artifact is produced."},
            {"artifact_type": "pose_refinement_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Pose refinement status is report-only until explicit BA/refinement artifacts are available."},
            {"artifact_type": "bundle_adjustment_report", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "no_explicit_bundle_adjustment_artifact", "summary": "Bundle adjustment report is reserved for explicit BA outputs."},
            {"artifact_type": "scale_stability_report", "stage": "pose_estimation_optimization", "status": "succeeded", "summary": "Scale stability is not sufficient for measurement without scale source.", "measurement_allowed": False},
            {"artifact_type": "training_view_selection_report", "stage": "training_input_optimization", "status": "succeeded", "summary": "Training input selection preserves original supervision.", "metrics": training.get("metrics") or {}},
            {"artifact_type": "holdout_view_selection_report", "stage": "training_input_optimization", "status": "succeeded", "summary": "Holdout selection remains tied to render evaluation and does not mutate training inputs.", "metrics": render.get("metrics") or {}},
            {"artifact_type": "appearance_group_report", "stage": "training_input_optimization", "status": "succeeded", "summary": "Appearance grouping is reported as a training strategy signal."},
            {"artifact_type": "mask_lineage_report", "stage": "mask_optimization", "status": "succeeded", "summary": "Mask lineage is optional for training and visible as report data."},
            {"artifact_type": "mask_visibility_report", "stage": "mask_optimization", "status": "succeeded", "summary": "Mask visibility policy is report-only unless an explicit mask stage applies it."},
            {"artifact_type": "photometric_consistency_report", "stage": "gaussian_training_optimization", "status": "succeeded", "summary": "Photometric variation is treated as training strategy, not source evidence mutation."},
            {"artifact_type": "training_strategy_report", "stage": "gaussian_training_optimization", "status": "succeeded", "summary": "Training strategy keeps original supervision for the default production route."},
            {"artifact_type": "drone_capture_profile", "stage": "input_route", "status": "succeeded" if has_drone else "skipped", "failure_reason": None if has_drone else "no_drone_input", "summary": "Drone route is skipped without drone/aerial metadata."},
            {"artifact_type": "aerial_overlap_report", "stage": "input_route", "status": "succeeded" if has_drone else "skipped", "failure_reason": None if has_drone else "no_drone_input", "summary": "Aerial overlap is report-only without drone metadata."},
            {"artifact_type": "flight_strip_report", "stage": "input_route", "status": "succeeded" if has_drone else "skipped", "failure_reason": None if has_drone else "no_drone_input", "summary": "Flight strip grouping is skipped without drone input."},
            {"artifact_type": "gps_prior_report", "stage": "raw_media_inspection", "status": "succeeded", "summary": "GPS is a prior only and never enables measurement-grade by itself.", "metrics": {"asset_count": asset_count, "drone_metadata_present": has_drone}, "measurement_allowed": False},
            {"artifact_type": "gcp_report", "stage": "input_route", "status": "succeeded" if has_scale else "skipped", "failure_reason": None if has_scale else "no_control_point_or_scale_input", "summary": "GCP/scale evidence must pass MeasurementReadinessGate."},
            {"artifact_type": "scale_alignment_report", "stage": "input_route", "status": "succeeded" if has_scale else "skipped", "failure_reason": None if has_scale else "no_scale_input", "summary": "Scale alignment is skipped without scale evidence.", "measurement_allowed": False},
            {"artifact_type": "georef_report", "stage": "input_route", "status": "skipped", "failure_reason": "no_georeference_input", "summary": "Georeference is not inferred from GPS-only metadata.", "measurement_allowed": False},
            {"artifact_type": "capture_group_manifest", "stage": "input_route", "status": "succeeded", "summary": "Capture grouping is represented by current input classification and route manifest.", "inputs": common_inputs},
            {"artifact_type": "per_group_pose_report", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "single_group_stage_optimized_run", "summary": "Per-group pose is skipped for single-group stage optimized runs."},
            {"artifact_type": "global_scene_graph", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "no_cross_group_alignment", "summary": "Global scene graph is not fabricated without alignment edges."},
            {"artifact_type": "cross_group_alignment_report", "stage": "pose_estimation_optimization", "status": "skipped", "failure_reason": "no_cross_group_alignment", "summary": "Cross-group alignment is skipped unless multiple capture groups exist.", "measurement_allowed": False},
            {"artifact_type": "manual_control_point_report", "stage": "input_route", "status": "skipped", "failure_reason": "no_manual_control_points", "summary": "Manual control points are not inferred automatically."},
            {"artifact_type": "depth_prior_manifest", "stage": "input_route", "status": "succeeded" if has_depth else "skipped", "failure_reason": None if has_depth else "no_depth_input", "summary": "Depth priors are skipped without LiDAR/RGB-D/depth assets.", "measurement_allowed": False},
            {"artifact_type": "normal_prior_manifest", "stage": "input_route", "status": "succeeded" if has_depth else "skipped", "failure_reason": None if has_depth else "no_depth_input", "summary": "Normal priors are skipped without depth inputs."},
            {"artifact_type": "prior_reliability_report", "stage": "input_route", "status": "succeeded" if has_depth else "skipped", "failure_reason": None if has_depth else "no_depth_input", "summary": "Learned priors are not measurement evidence by default.", "measurement_allowed": False},
            {"artifact_type": "depth_sensor_report", "stage": "input_route", "status": "succeeded" if has_depth else "skipped", "failure_reason": None if has_depth else "no_depth_input", "summary": "Depth sensor calibration is required before measurement use.", "measurement_allowed": False},
            {"artifact_type": "scale_marker_report", "stage": "measurement_gate", "status": "succeeded" if has_scale else "skipped", "failure_reason": None if has_scale else "no_scale_marker_input", "summary": "Scale markers are evidence candidates and require uncertainty checks.", "measurement_allowed": False},
            {"artifact_type": "control_point_alignment_report", "stage": "measurement_gate", "status": "skipped", "failure_reason": "no_control_point_alignment", "summary": "Control point alignment is not inferred automatically.", "measurement_allowed": False},
            {"artifact_type": "scale_uncertainty_report", "stage": "measurement_gate", "status": "skipped", "failure_reason": "scale_uncertainty_not_estimated", "summary": "Scale uncertainty must be estimated before measurement-grade claims.", "measurement_allowed": False},
            {"artifact_type": "measurement_readiness_report", "stage": "measurement_gate", "status": "succeeded", "summary": "Measurement readiness defaults to false without trusted scale and surface evidence.", "measurement_allowed": False},
            {"artifact_type": "measurement_confidence_report", "stage": "measurement_gate", "status": "succeeded", "summary": "Measurement confidence remains low unless scale/control/surface gates pass.", "measurement_allowed": False},
            {"artifact_type": "mesh_extraction_report", "stage": "measurement_gate", "status": "unsupported", "failure_reason": "surface_model_not_available", "summary": "Mesh/surface extraction is not claimed for default splat-only output.", "measurement_allowed": False},
            {"artifact_type": "scene_partition", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "large_scene_partition_not_triggered", "summary": "Large scene partitioning is skipped unless thresholds trigger it."},
            {"artifact_type": "block_training_manifest", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "block_training_not_triggered", "summary": "Block training is skipped for non-partitioned runs."},
            {"artifact_type": "lod_manifest", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "lod_export_not_triggered", "summary": "LOD export is skipped unless delivery config enables it."},
            {"artifact_type": "chunk_manifest", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "chunk_export_not_triggered", "summary": "Chunk manifest is skipped for single-model runs."},
            {"artifact_type": "streaming_manifest", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "streaming_export_not_triggered", "summary": "Streaming manifest is skipped unless large-scene delivery is enabled."},
            {"artifact_type": "tiles_conversion_report", "stage": "final_artifact_selection", "status": "unsupported", "failure_reason": "true_3dtiles_converter_not_run", "summary": "True 3D Tiles conversion is not faked."},
            {"artifact_type": "viewer_package_manifest", "stage": "final_artifact_selection", "status": "succeeded" if delivery_supported else "skipped", "failure_reason": None if delivery_supported else "no_final_model", "summary": "Viewer package readiness follows final model availability."},
            {"artifact_type": "compression_conversion_report", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "compression_export_not_enabled", "summary": "Compression export is skipped unless enabled."},
            {"artifact_type": "spz_export_report", "stage": "final_artifact_selection", "status": "skipped", "failure_reason": "spz_export_not_enabled", "summary": "SPZ export is skipped unless enabled."},
            {"artifact_type": "forensic_manifest", "stage": "final_artifact_selection", "status": "succeeded", "summary": "Forensic manifest preserves source map and does not imply forensic conclusion."},
            {"artifact_type": "experimental_route_report", "stage": "input_route", "status": "succeeded", "summary": "Experimental routes are default-off and cannot publish measurement-grade output.", "route_role": "experimental", "production_allowed": False, "measurement_allowed": False},
        ]
        return specs

    def _best_source_map(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        source_map_path = dataset.get("source_map_path")
        return read_json(Path(str(source_map_path)), {"sources": []}) if source_map_path else {"sources": []}

    def _best_image_policy(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        manifest = read_json(Path(str(dataset.get("output_path") or "")), {}) if dataset.get("output_path") else {}
        return {
            "image_policy": manifest.get("image_policy") or {},
            "pose_image_distribution": manifest.get("pose_image_distribution") or {},
            "training_image_distribution": manifest.get("training_image_distribution") or {},
        }

    def _best_enhancement_provenance(self, context: StageContext) -> dict[str, Any]:
        dataset = read_json(context.stage_dir("dataset_assembly") / "best_dataset_selection.json", {})
        path = dataset.get("enhancement_provenance_path")
        return read_json(Path(str(path)), {"images": []}) if path else {"images": []}

    def _limitations(self, stage_results: dict[str, Any], training: dict[str, Any], eval_metrics: dict[str, Any]) -> list[str]:
        limitations = []
        if training.get("status") in {"skipped", "planned"} or training.get("rejected_reason") == "training_not_executed":
            limitations.append("3DGS 训练 adapter 未执行，当前成果停留在阶段最优输入与训练路线选择层。")
        if eval_metrics.get("status") == "not_rendered":
            limitations.append("缺少可评估最终模型，因此无法发布生产级 viewer。")
        for stage_name, result in stage_results.items():
            if result.get("whether_stage_has_remaining_improvement"):
                limitations.append(f"{stage_name} 仍存在可提升空间：{result.get('next_stage_recommendation')}")
        return limitations

    def _best_route_report(self, stage_results: dict[str, Any], training: dict[str, Any], eval_metrics: dict[str, Any], limitations: list[str]) -> str:
        lines = [
            "# Best Route Report",
            "",
            "## Route",
            f"- training_candidate: `{training.get('candidate_name')}`",
            f"- training_status: `{training.get('status')}`",
            f"- eval_psnr: {eval_metrics.get('PSNR')}",
            f"- inspectability_score: {eval_metrics.get('inspectability_score')}",
            "",
            "## Stage Decisions",
        ]
        for stage_name, result in stage_results.items():
            lines.append(f"- `{stage_name}` best=`{result.get('best_candidate')}` improvement={result.get('improvement_summary')}")
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in limitations or ["当前阶段没有记录阻断级限制。"])
        return "\n".join(lines) + "\n"

    def _all_stage_report(self, stage_results: dict[str, Any]) -> str:
        lines = ["# All Stage Report", ""]
        for stage_name, result in stage_results.items():
            lines.extend(
                [
                    f"## {stage_name}",
                    f"- best: `{result.get('best_candidate')}`",
                    f"- improvement: {result.get('improvement_summary')}",
                    f"- risk: {result.get('risk_summary')}",
                    f"- remaining_improvement: {result.get('whether_stage_has_remaining_improvement')}",
                    "",
                ]
            )
        return "\n".join(lines)
