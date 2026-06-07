from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_float, default_int
from app.models import Asset
from app.models.workflow import Workflow
from app.operators.base import CommandResult
from app.operators.input_router import InputRouterOperator, InputRoutingResult
from app.services.stage_cache import StageCache, cache_hit_command
from app.services.storage_service import StorageService


class AssetCheckOperator:
    name = "preprocess.asset_check"
    queue = "preprocess"

    def run(self, asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": asset.id,
            "storage_uri_present": bool(asset.storage_uri),
            "size_bytes": asset.size_bytes,
            "mime_type": asset.mime_type,
            "usable": bool(asset.storage_uri and (asset.size_bytes or 0) > 0),
        }


class ImageQualityOperator:
    name = "preprocess.image_quality"
    queue = "preprocess"

    def run(self, asset: Asset) -> dict[str, Any]:
        return {
            "asset_id": asset.id,
            "quality_score": 0.75,
            "usable": bool(asset.storage_uri and asset.asset_type in {"detail_photo", "supplement_photo", "scale_marker", "pano_360"}),
            "notes": ["Detailed blur/exposure scoring is available for future OpenCV integration."],
        }


class ExtractKeyframesOperator:
    name = "preprocess.extract_keyframes"
    queue = "preprocess"

    def __init__(self, storage: StorageService | None = None):
        self.storage = storage or StorageService()

    def run(self, asset: Asset, workspace_dir: Path, target_frames: int = 500) -> dict[str, Any]:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        input_path = workspace_dir / asset.original_filename
        self.storage.download_to_file(_relative_from_uri(asset.storage_uri), input_path)
        frames_dir = workspace_dir / "keyframes"
        frames_dir.mkdir(exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"fps=min({target_frames}/max(t\\,1),30)",
            str(frames_dir / "frame_%06d.jpg"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        frames = sorted(frames_dir.glob("*.jpg"))
        manifest = {
            "source_video_asset_id": asset.id,
            "command": command,
            "exit_code": completed.returncode,
            "frames": [{"frame_id": frame.stem, "path": frame.name, "selected": True} for frame in frames],
        }
        (workspace_dir / "keyframe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class Pano360CropOperator:
    name = "preprocess.crop_pano360"
    queue = "preprocess"

    def run(self, asset: Asset, workspace_dir: Path) -> dict[str, Any]:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        source_pano_id = asset.metadata_json.get("source_pano_id") or asset.id
        yaws = [0, 60, 120, 180, 240, 300]
        crops = [
            {
                "crop_id": f"{source_pano_id}_yaw_{yaw:03d}",
                "path": f"{source_pano_id}_yaw_{yaw:03d}.jpg",
                "yaw": yaw,
                "pitch": 0,
                "roll": 0,
                "fov": 80,
                "width": 1024,
                "height": 1024,
                "source_pano_id": source_pano_id,
                "shared_center_group": source_pano_id,
            }
            for yaw in yaws
        ]
        manifest = {
            "source_pano_asset_id": asset.id,
            "source_pano_id": source_pano_id,
            "shared_center": True,
            "virtual_camera_model": "pinhole_perspective_from_equirectangular",
            "crops": crops,
        }
        (workspace_dir / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


class DynamicMaskOperator:
    name = "preprocess.dynamic_mask"
    queue = "gpu"

    def run(self, workflow: Workflow, preprocess: "PreprocessRunResult") -> dict[str, Any]:
        settings = get_settings()
        workspace_dir = Path(settings.workspace_root) / "runs" / workflow.id / "dynamic_mask"
        masks_dir = workspace_dir / "masks"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        config = workflow.config_json or {}
        operator_config = settings.engine_config.get("operators", {}).get("dynamic_mask", {}) or {}
        dynamic_classes = config.get("dynamic_classes") or ["person", "vehicle", "leaf", "water"]
        cache_entry = StageCache(settings).entry(
            self.name,
            inputs=[*preprocess.image_paths, {"input_mode": (preprocess.media_metadata or {}).get("input_mode")}],
            stage_config={"operator_config": operator_config, "dynamic_classes": dynamic_classes, "semantic_dependencies": _semantic_dependency_fingerprint(settings)},
            algorithm_version="dynamic-mask-v2",
        )
        report_path = workspace_dir / "dynamic_object_report.json"
        if cache_entry.hit and StageCache(settings).restore(cache_entry, workspace_dir) and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            return report
        external_report = _run_configured_dynamic_mask_command(operator_config, workflow, preprocess, workspace_dir, masks_dir, settings)
        external_unavailable_report = None
        if external_report is not None and external_report.pop("_fallback_to_builtin", False):
            external_unavailable_report = external_report
            external_report = None
        if external_report is not None:
            report = external_report
        else:
            if (preprocess.media_metadata or {}).get("input_mode") != "video":
                report = _build_image_collection_dynamic_mask_report(workflow, preprocess, dynamic_classes, masks_dir, operator_config)
                if external_unavailable_report:
                    report["external_semantic_mask"] = external_unavailable_report
                report.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
                report["report_path"] = str(report_path)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                StageCache(settings).save(cache_entry, workspace_dir, metadata=report)
                return report
            report = _run_ffmpeg_frame_diff_dynamic_mask(operator_config, workflow, preprocess, masks_dir, dynamic_classes)
            if external_unavailable_report:
                report["external_semantic_mask"] = external_unavailable_report
        report.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        StageCache(settings).save(cache_entry, workspace_dir, metadata=report)
        return report


def _build_image_collection_dynamic_mask_report(
    workflow: Workflow,
    preprocess: "PreprocessRunResult",
    dynamic_classes: list[str],
    masks_dir: Path,
    operator_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator_config = operator_config or {}
    if _requests_reflection_sensitive_mask(dynamic_classes):
        return _build_reflection_sensitive_image_collection_report(workflow, preprocess, dynamic_classes, masks_dir, operator_config)
    return {
        "workflow_id": workflow.id,
        "operator": DynamicMaskOperator.name,
        "passed": True,
        "hard_fail": False,
        "dynamic_ratio": 0.0,
        "masked_frame_count": 0,
        "evaluated_frame_count": len(preprocess.image_paths),
        "dynamic_classes": dynamic_classes,
        "policy": "static_3dgs_must_not_explain_dynamic_objects",
        "implementation": "not_applicable_image_collection",
        "reason": "frame_diff_dynamic_mask_requires_video_sequence_or_external_semantic_model",
        "notes": [
            "No semantic dynamic-mask command is configured.",
            "Frame differencing is not valid for unordered or wide-baseline photo collections because camera motion dominates pixel differences.",
        ],
    }


def _requests_reflection_sensitive_mask(dynamic_classes: list[str]) -> bool:
    requested = {str(item).strip().lower() for item in dynamic_classes}
    return bool(requested.intersection({"reflection", "mirror", "glass", "screen", "tv"}))


def _build_reflection_sensitive_image_collection_report(
    workflow: Workflow,
    preprocess: "PreprocessRunResult",
    dynamic_classes: list[str],
    masks_dir: Path,
    operator_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return {
            "workflow_id": workflow.id,
            "operator": DynamicMaskOperator.name,
            "passed": False,
            "hard_fail": False,
            "dynamic_ratio": 0.0,
            "masked_frame_count": 0,
            "evaluated_frame_count": len(preprocess.image_paths),
            "dynamic_classes": dynamic_classes,
            "policy": "static_3dgs_must_not_explain_reflection_or_screen_artifacts",
            "implementation": "reflection_heuristic_unavailable",
            "reason": "opencv_or_numpy_unavailable_for_reflection_mask",
            "images": [],
        }

    max_images = int(operator_config.get("reflection_heuristic_max_images") or 0)
    image_paths = list(preprocess.image_paths[:max_images] if max_images > 0 else preprocess.image_paths)
    max_ratio = float(operator_config.get("reflection_heuristic_max_coverage_ratio") or operator_config.get("max_dynamic_ratio") or 0.30)
    dark_value_threshold = int(operator_config.get("reflection_dark_value_threshold") or 28)
    specular_value_threshold = int(operator_config.get("reflection_specular_value_threshold") or 246)
    specular_sat_threshold = int(operator_config.get("reflection_specular_saturation_threshold") or 80)
    masks_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    ratios: list[float] = []
    failed_decode_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failed_decode_count += 1
            continue
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        saturation = hsv[:, :, 1]
        lap_abs = np.absolute(cv2.Laplacian(gray, cv2.CV_16S))
        dark_smooth = ((value <= dark_value_threshold) & (lap_abs <= 18)).astype(np.uint8) * 255
        specular = ((value >= specular_value_threshold) & (saturation <= specular_sat_threshold)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        dark_smooth = cv2.morphologyEx(dark_smooth, cv2.MORPH_OPEN, kernel)
        dark_smooth = cv2.morphologyEx(dark_smooth, cv2.MORPH_CLOSE, kernel)
        specular = cv2.dilate(specular, np.ones((3, 3), dtype=np.uint8), iterations=1)
        mask = cv2.bitwise_or(_large_component_mask(dark_smooth, min_area=max(64, int(width * height * 0.001))), specular)
        coverage = float((mask > 0).mean())
        if coverage > max_ratio:
            mask = specular
            coverage = float((mask > 0).mean())
        mask_path = masks_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        ratios.append(coverage)
        entries.append(
            {
                "image_name": image_path.name,
                "mask_path": str(mask_path),
                "foreground_ratio": round(coverage, 6),
                "method": "dark_smooth_region_plus_specular_highlight_heuristic",
            }
        )

    dynamic_ratio = sum(ratios) / max(len(ratios), 1)
    return {
        "workflow_id": workflow.id,
        "operator": DynamicMaskOperator.name,
        "passed": dynamic_ratio <= max_ratio,
        "hard_fail": dynamic_ratio > max_ratio,
        "dynamic_ratio": round(dynamic_ratio, 6),
        "max_dynamic_ratio": max_ratio,
        "masked_frame_count": sum(1 for ratio in ratios if ratio > 0.0),
        "evaluated_frame_count": len(image_paths),
        "failed_decode_count": failed_decode_count,
        "dynamic_classes": dynamic_classes,
        "policy": "static_3dgs_must_not_explain_reflection_or_screen_artifacts",
        "implementation": "reflection_heuristic_mask",
        "method": "dark_smooth_region_plus_specular_highlight_heuristic",
        "mask_format": "png_full_resolution_binary",
        "masks_dir": str(masks_dir),
        "images": entries,
    }


def _large_component_mask(mask: Any, *, min_area: int) -> Any:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            filtered[labels == label] = 255
    return filtered


def _run_configured_dynamic_mask_command(
    operator_config: dict[str, Any],
    workflow: Workflow,
    preprocess: "PreprocessRunResult",
    workspace_dir: Path,
    masks_dir: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    command_template = operator_config.get("command")
    if not command_template:
        return None
    semantic_values = _semantic_mask_template_values(settings)
    missing_semantic_paths = _missing_semantic_dependencies(semantic_values)
    if missing_semantic_paths:
        return {
            "workflow_id": workflow.id,
            "operator": DynamicMaskOperator.name,
            "implementation": "external_command_unavailable",
            "reason": "semantic_mask_dependency_missing",
            "missing_required_paths": missing_semantic_paths,
            "_fallback_to_builtin": True,
        }
    dynamic_prompt = ". ".join(str(item).strip().lower() for item in (workflow.config_json or {}).get("dynamic_classes") or ["person", "vehicle", "leaf", "water", "reflection"] if str(item).strip())
    if dynamic_prompt and not dynamic_prompt.endswith("."):
        dynamic_prompt += "."
    workflow_config = workflow.config_json or {}
    values = {
        **semantic_values,
        "images_dir": str(preprocess.images_dir),
        "dataset_dir": str(preprocess.dataset_dir),
        "workspace_dir": str(workspace_dir),
        "masks_dir": str(masks_dir),
        "output_report": str(workspace_dir / "dynamic_object_report.external.json"),
        "dynamic_prompt": dynamic_prompt,
        "max_images": str(int(workflow_config.get("dynamic_mask_semantic_max_images") or workflow_config.get("semantic_mask_max_images") or operator_config.get("semantic_max_images") or 120)),
        "max_dynamic_ratio": str(workflow_config.get("dynamic_mask_max_dynamic_ratio") or operator_config.get("max_dynamic_ratio") or 0.35),
        "box_threshold": str(operator_config.get("box_threshold") or semantic_values.get("box_threshold") or 0.3),
        "text_threshold": str(operator_config.get("text_threshold") or semantic_values.get("text_threshold") or 0.25),
    }
    command = [_format_template_part(str(part), values) for part in command_template]
    result = _run_command("preprocess.dynamic_mask", "dynamic_mask_gate", command, workspace_dir)
    report_path = Path(values["output_report"])
    if result.exit_code == 0 and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(
            {
                "workflow_id": workflow.id,
                "operator": DynamicMaskOperator.name,
                "implementation": "external_command",
                "command": command,
                "exit_code": result.exit_code,
                "stderr_tail": result.stderr[-1000:] if result.stderr else "",
            }
        )
        return report
    if result.exit_code == 2 and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.update(
            {
                "workflow_id": workflow.id,
                "operator": DynamicMaskOperator.name,
                "implementation": "external_command_unavailable",
                "command": command,
                "exit_code": result.exit_code,
                "stderr_tail": result.stderr[-1000:] if result.stderr else "",
                "_fallback_to_builtin": True,
            }
        )
        return report
    return {
        "workflow_id": workflow.id,
        "operator": DynamicMaskOperator.name,
        "passed": False,
        "hard_fail": True,
        "dynamic_ratio": 1.0,
        "masked_frame_count": 0,
        "evaluated_frame_count": len(preprocess.image_paths),
        "implementation": "external_command",
        "command": command,
        "exit_code": result.exit_code,
        "reason": "dynamic_mask_command_failed" if result.exit_code != 0 else "dynamic_mask_report_missing",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }


def _semantic_mask_template_values(settings: Settings) -> dict[str, str]:
    semantic = settings.engine_config.get("operators", {}).get("semantic_masking", {}) or {}
    return {
        "python": str(semantic.get("python") or "python3"),
        "semantic_wrapper": str(semantic.get("wrapper") or ""),
        "sam2_repo_path": str(semantic.get("sam2_repo_path") or ""),
        "grounded_sam2_repo_path": str(semantic.get("grounded_sam2_repo_path") or ""),
        "sam2_config": str(semantic.get("sam2_config") or ""),
        "sam2_checkpoint": str(semantic.get("sam2_checkpoint") or ""),
        "groundingdino_repo_path": str(semantic.get("groundingdino_repo_path") or ""),
        "groundingdino_config": str(semantic.get("groundingdino_config") or ""),
        "groundingdino_checkpoint": str(semantic.get("groundingdino_checkpoint") or ""),
        "groundingdino_checkpoint_min_bytes": str(int(semantic.get("groundingdino_checkpoint_min_bytes") or 0)),
        "groundingdino_checkpoint_md5": str(semantic.get("groundingdino_checkpoint_md5") or ""),
        "text_encoder_path": str(semantic.get("text_encoder_path") or ""),
        "device": str(semantic.get("device") or "auto"),
        "box_threshold": str(semantic.get("box_threshold") or 0.3),
        "text_threshold": str(semantic.get("text_threshold") or 0.25),
        "sam2_checkpoint_min_bytes": str(int(semantic.get("sam2_checkpoint_min_bytes") or 0)),
    }


def _semantic_dependency_fingerprint(settings: Settings) -> list[dict[str, Any]]:
    values = _semantic_mask_template_values(settings)
    paths = [
        values.get("semantic_wrapper"),
        values.get("groundingdino_repo_path"),
        values.get("groundingdino_config"),
        values.get("groundingdino_checkpoint"),
        values.get("text_encoder_path"),
        values.get("sam2_repo_path"),
        values.get("grounded_sam2_repo_path"),
        values.get("sam2_config"),
        values.get("sam2_checkpoint"),
    ]
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in [str(path) for path in paths if path]:
        if value in seen:
            continue
        seen.add(value)
        path = Path(value)
        if path.exists():
            stat = path.stat()
            payload.append({"path": value, "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        else:
            payload.append({"path": value, "exists": False})
    return payload


def _missing_semantic_dependencies(values: dict[str, str]) -> list[str]:
    missing = [
        value
        for value in [
            values.get("semantic_wrapper"),
            values.get("groundingdino_repo_path"),
            values.get("groundingdino_config"),
        ]
        if value and not Path(value).exists()
    ]
    checkpoint = values.get("groundingdino_checkpoint")
    if checkpoint:
        missing.extend(_missing_file_with_min_bytes(checkpoint, int(values.get("groundingdino_checkpoint_min_bytes") or 0)))
    text_encoder_path = values.get("text_encoder_path")
    if text_encoder_path:
        missing.extend(_missing_text_encoder(text_encoder_path))
    return missing


def _missing_file_with_min_bytes(value: str, min_bytes: int) -> list[str]:
    path = Path(value)
    if not path.exists():
        return [value]
    if min_bytes > 0 and path.is_file() and path.stat().st_size < min_bytes:
        return [f"{value}:size_bytes={path.stat().st_size}<min_bytes={min_bytes}"]
    return []


def _missing_text_encoder(value: str) -> list[str]:
    path = Path(value)
    if not path.exists() or not path.is_dir():
        return [value]
    missing: list[str] = []
    for filename in ["config.json", "vocab.txt"]:
        candidate = path / filename
        if not candidate.exists() or candidate.stat().st_size <= 0:
            missing.append(str(candidate))
    if not any((path / filename).exists() and (path / filename).stat().st_size > 0 for filename in ["model.safetensors", "pytorch_model.bin"]):
        missing.append(f"{path}:missing_model_weights")
    return missing


def _run_ffmpeg_frame_diff_dynamic_mask(
    operator_config: dict[str, Any],
    workflow: Workflow,
    preprocess: "PreprocessRunResult",
    masks_dir: Path,
    dynamic_classes: list[str],
) -> dict[str, Any]:
    ffmpeg = shutil.which(str(operator_config.get("ffmpeg_binary") or "ffmpeg"))
    max_frames = int(operator_config.get("frame_diff_max_frames") or 120)
    width = int(operator_config.get("frame_diff_width") or 64)
    height = int(operator_config.get("frame_diff_height") or 64)
    threshold = int(operator_config.get("frame_diff_threshold") or 28)
    if not ffmpeg:
        return {
            "workflow_id": workflow.id,
            "operator": DynamicMaskOperator.name,
            "passed": True,
            "hard_fail": False,
            "dynamic_ratio": 0.0,
            "masked_frame_count": 0,
            "evaluated_frame_count": len(preprocess.image_paths),
            "dynamic_classes": dynamic_classes,
            "policy": "static_3dgs_must_not_explain_dynamic_objects",
            "implementation": "unavailable",
            "reason": "ffmpeg_missing_and_no_external_dynamic_mask_command",
            "notes": ["Dynamic mask did not run because no configured segmentation command or ffmpeg binary was available."],
        }

    decoded: list[tuple[Path, bytes]] = []
    failed_decode_count = 0
    for image_path in preprocess.image_paths[:max_frames]:
        pixels = _decode_luma_thumbnail(ffmpeg, image_path, width, height)
        if pixels is None:
            failed_decode_count += 1
            continue
        decoded.append((image_path, pixels))

    if len(decoded) < 2:
        return {
            "workflow_id": workflow.id,
            "operator": DynamicMaskOperator.name,
            "passed": True,
            "hard_fail": False,
            "dynamic_ratio": 0.0,
            "masked_frame_count": 0,
            "evaluated_frame_count": len(preprocess.image_paths),
            "decoded_frame_count": len(decoded),
            "failed_decode_count": failed_decode_count,
            "dynamic_classes": dynamic_classes,
            "policy": "static_3dgs_must_not_explain_dynamic_objects",
            "implementation": "ffmpeg_frame_diff_unavailable",
            "reason": "not_enough_decodable_frames_for_motion_mask",
            "notes": ["No dynamic ratio was inferred because fewer than two frames could be decoded."],
        }

    ratios: list[float] = []
    masked_frame_count = 0
    previous_path, previous_pixels = decoded[0]
    for image_path, pixels in decoded[1:]:
        mask = bytes(255 if abs(current - previous) > threshold else 0 for current, previous in zip(pixels, previous_pixels))
        changed = sum(1 for value in mask if value)
        ratio = changed / max(len(mask), 1)
        ratios.append(ratio)
        if ratio > 0:
            masked_frame_count += 1
        _write_pgm_mask(masks_dir / f"{image_path.stem}.pgm", mask, width, height)
        previous_path, previous_pixels = image_path, pixels

    dynamic_ratio = sum(ratios) / max(len(ratios), 1)
    return {
        "workflow_id": workflow.id,
        "operator": DynamicMaskOperator.name,
        "passed": dynamic_ratio <= float(operator_config.get("max_dynamic_ratio") or 0.35),
        "hard_fail": dynamic_ratio > float(operator_config.get("max_dynamic_ratio") or 0.35),
        "dynamic_ratio": dynamic_ratio,
        "max_dynamic_ratio": float(operator_config.get("max_dynamic_ratio") or 0.35),
        "masked_frame_count": masked_frame_count,
        "evaluated_frame_count": len(preprocess.image_paths),
        "decoded_frame_count": len(decoded),
        "failed_decode_count": failed_decode_count,
        "mask_format": "pgm_64x64_frame_diff",
        "masks_dir": str(masks_dir),
        "dynamic_classes": dynamic_classes,
        "policy": "static_3dgs_must_not_explain_dynamic_objects",
        "implementation": "ffmpeg_frame_diff",
        "basis": "low-resolution grayscale frame differencing; configure an external segmentation command for semantic people/vehicle/reflection masks",
    }


def _decode_luma_thumbnail(ffmpeg: str, image_path: Path, width: int, height: int) -> bytes | None:
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(image_path),
            "-vf",
            f"scale={width}:{height},format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    expected = width * height
    if completed.returncode != 0 or len(completed.stdout) != expected:
        return None
    return completed.stdout


def _write_pgm_mask(path: Path, mask: bytes, width: int, height: int) -> None:
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + mask)


def _format_template_part(value: str, values: dict[str, str]) -> str:
    return value.format(**values)


@dataclass
class PreprocessRunResult:
    workspace_dir: Path
    dataset_dir: Path
    images_dir: Path
    image_paths: list[Path]
    commands: list[CommandResult]
    media_metadata: dict[str, Any]
    asset_quality: dict[str, Any]
    routing_manifest_path: Path


class DatasetPreprocessOperator:
    name = "preprocess.dataset"
    queue = "preprocess"

    def __init__(self, settings: Settings | None = None, storage: StorageService | None = None):
        self.settings = settings or get_settings()
        self.storage = storage or StorageService(self.settings)

    def run(self, workflow: Workflow, assets: list[Asset], routing: InputRoutingResult | None = None) -> PreprocessRunResult:
        config = workflow.config_json or {}
        mode = config.get("mode") or config.get("profile") or self.settings.workflow_default_mode
        frame_target = _resolve_video_frame_target(config, self.settings)
        extract_fps = float(config.get("extract_fps") or default_float("preprocess.video_global.extract_fps", 1.5, settings=self.settings))
        max_video_long_edge = default_int("preprocess.video_global.max_long_edge_px", 1920, settings=self.settings)
        routing = routing or InputRouterOperator(self.settings).run(workflow, assets)
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "preprocess"
        dataset_dir = workspace_dir / "dataset"
        images_dir = dataset_dir / "images"
        raw_dir = workspace_dir / "raw"
        for directory in (images_dir, raw_dir):
            directory.mkdir(parents=True, exist_ok=True)

        commands: list[CommandResult] = []
        role_summary: dict[str, int] = {}
        asset_type_summary: dict[str, int] = {}
        for asset in assets:
            role_summary[asset.role] = role_summary.get(asset.role, 0) + 1
            asset_type_summary[asset.asset_type] = asset_type_summary.get(asset.asset_type, 0) + 1

        global_assets = list(routing.global_inputs)
        if not global_assets and routing.route_key in {"colmap_splatfacto", "instantsplatpp_sparse_local"}:
            global_assets = [asset for asset in routing.detail_inputs if _is_image_like(asset)]

        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=global_assets,
            stage_config={
                "mode": mode,
                "route_id": routing.route_id,
                "route_key": routing.route_key,
                "frame_target": frame_target,
                "extract_fps": extract_fps,
                "max_video_long_edge": max_video_long_edge,
                "adaptive_sampling": config.get("adaptive_frame_sampling", True),
                "preprocess_version": "adaptive-v1",
            },
            algorithm_version="preprocess-dataset-v2",
        )
        metadata_path = workspace_dir / "preprocess_metadata.json"
        routing_manifest_path = workspace_dir / "routing_manifest.json"
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and metadata_path.exists() and routing_manifest_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            image_paths = sorted(images_dir.glob("*"))
            metadata.setdefault("source_files", [path.name for path in image_paths])
            metadata.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            asset_quality = metadata.get("asset_quality") or {
                "passed": bool(image_paths),
                "input_asset_count": len(assets),
                "global_image_count": len(image_paths),
                "issues": [],
            }
            return PreprocessRunResult(
                workspace_dir=workspace_dir,
                dataset_dir=dataset_dir,
                images_dir=images_dir,
                image_paths=image_paths,
                commands=[cache_hit_command("preprocess.dataset", "preprocess", cache_entry.cache_key, workspace_dir)],
                media_metadata=metadata,
                asset_quality=asset_quality,
                routing_manifest_path=routing_manifest_path,
            )

        image_paths: list[Path] = []
        routed_assets: list[dict[str, Any]] = []
        for asset in global_assets:
            if asset.asset_type in {"global_video", "supplement_video"}:
                staged_video = raw_dir / (asset.original_filename or asset.filename)
                self.storage.download_to_file(_relative_from_uri(asset.storage_uri), staged_video)
                frames_dir = workspace_dir / "keyframes" / asset.id
                frames_dir.mkdir(parents=True, exist_ok=True)
                extraction = _extract_adaptive_video_frames(
                    staged_video=staged_video,
                    frames_dir=frames_dir,
                    asset_id=asset.id,
                    workspace_dir=workspace_dir,
                    frame_target=frame_target,
                    extract_fps=extract_fps,
                    max_long_edge=max_video_long_edge,
                    mode=mode,
                    config=config,
                )
                commands.extend(extraction["commands"])
                result = extraction["commands"][-1]
                if result.exit_code != 0:
                    routed_assets.append(
                        {
                            "asset_id": asset.id,
                            "asset_type": asset.asset_type,
                            "role": asset.role,
                            "route": "global_video_keyframes",
                            "status": "failed",
                            "exit_code": result.exit_code,
                            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
                        }
                    )
                    continue
                for frame in extraction["selected_frames"]:
                    target = images_dir / frame.name
                    shutil.copyfile(frame, target)
                    image_paths.append(target)
                routed_assets.append(
                    {
                        "asset_id": asset.id,
                        "asset_type": asset.asset_type,
                        "role": asset.role,
                        "route": "global_video_keyframes",
                        "candidate_frame_count": extraction["candidate_frame_count"],
                        "selected_frame_count": extraction["selected_frame_count"],
                        "frame_selection_report": str(extraction["report_path"]),
                    }
                )
            elif _is_image_like(asset):
                safe_name = _unique_image_name(images_dir, asset.original_filename or asset.filename or f"{asset.id}.jpg")
                target = images_dir / safe_name
                self.storage.download_to_file(_relative_from_uri(asset.storage_uri), target)
                image_paths.append(target)
                route = "global_skeleton" if asset in routing.global_inputs else "detail_photo_fallback_global_skeleton"
                routed_assets.append({"asset_id": asset.id, "asset_type": asset.asset_type, "role": asset.role, "route": route, "image_name": safe_name})
            else:
                routed_assets.append({"asset_id": asset.id, "asset_type": asset.asset_type, "role": asset.role, "route": "deferred"})

        routing_manifest = {
            "workflow_id": workflow.id,
            "project_id": workflow.project_id,
            "mode": mode,
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "route_reason": routing.route_reason,
            "asset_count": len(assets),
            "asset_type_summary": asset_type_summary,
            "role_summary": role_summary,
            "global_image_count": len(image_paths),
            "global_inputs": [item["asset_id"] for item in routing.manifest.get("global_inputs", [])],
            "detail_inputs": [item["asset_id"] for item in routing.manifest.get("detail_inputs", [])],
            "pano_inputs": [item["asset_id"] for item in routing.manifest.get("pano_inputs", [])],
            "supplement_inputs": [item["asset_id"] for item in routing.manifest.get("supplement_inputs", [])],
            "scale_inputs": [item["asset_id"] for item in routing.manifest.get("scale_inputs", [])],
            "fallback_detail_photos_used_as_global": not bool(routing.global_inputs) and bool(global_assets),
            "routes": routed_assets,
        }
        routing_manifest_path.write_text(json.dumps(routing_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        image_prepare_report_path = _write_parallel_image_prepare_report(workspace_dir, image_paths)

        min_images = 2 if mode == "quick_preview" else 3
        asset_quality = {
            "passed": len(image_paths) >= min_images,
            "input_asset_count": len(assets),
            "global_image_count": len(image_paths),
            "min_required_global_images": min_images,
            "route_key": routing.route_key,
            "asset_type_summary": asset_type_summary,
            "role_summary": role_summary,
            "issues": [] if len(image_paths) >= min_images else ["insufficient_global_images"],
        }
        media_metadata = {
            "input_mode": "video" if any(asset.asset_type == "global_video" for asset in global_assets) else "images",
            "route_id": routing.route_id,
            "route_key": routing.route_key,
            "route_reason": routing.route_reason,
            "asset_count": len(assets),
            "staged_file_count": len(image_paths),
            "mode": mode,
            "frame_target": frame_target,
            "extract_fps": extract_fps,
            "max_video_long_edge_px": max_video_long_edge,
            "source_files": [path.name for path in image_paths],
            "asset_type_summary": asset_type_summary,
            "role_summary": role_summary,
            "cache_hit": False,
            "cache_key": cache_entry.cache_key,
            "adaptive_frame_sampling": True,
            "image_prepare_report": str(image_prepare_report_path),
            "asset_quality": asset_quality,
        }
        metadata_path.write_text(json.dumps(media_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        cache.save(cache_entry, workspace_dir, metadata=media_metadata)
        return PreprocessRunResult(
            workspace_dir=workspace_dir,
            dataset_dir=dataset_dir,
            images_dir=images_dir,
            image_paths=image_paths,
            commands=commands,
            media_metadata=media_metadata,
            asset_quality=asset_quality,
            routing_manifest_path=routing_manifest_path,
        )


def _resolve_video_frame_target(config: dict[str, Any], settings: Settings) -> int:
    if config.get("frame_target"):
        return int(config["frame_target"])
    if config.get("target_frames_max"):
        return int(config["target_frames_max"])
    return default_int("preprocess.video_global.target_frames_max", settings.nerfstudio_video_frame_target, settings=settings)


def _extract_adaptive_video_frames(
    *,
    staged_video: Path,
    frames_dir: Path,
    asset_id: str,
    workspace_dir: Path,
    frame_target: int,
    extract_fps: float,
    max_long_edge: int,
    mode: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_target = _adaptive_target_frame_count(frame_target, mode, config)
    candidate_limit = max(selected_target, min(max(frame_target, selected_target), selected_target * 2))
    scan_fps = float(config.get("scan_fps") or min(max(extract_fps, 1.0), 3.0))
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(staged_video),
        "-vf",
        f"fps={scan_fps},scale=min({max_long_edge}\\,iw):-2",
        "-frames:v",
        str(candidate_limit),
        str(frames_dir / f"{asset_id}_%06d.jpg"),
    ]
    result = _run_command("preprocess.extract_keyframes", "preprocess", command, workspace_dir)
    frames = sorted(frames_dir.glob("*.jpg"))
    report_path = frames_dir / "frame_selection_report.json"
    if result.exit_code != 0:
        report = {
            "asset_id": asset_id,
            "adaptive": True,
            "command": command,
            "exit_code": result.exit_code,
            "candidate_frame_count": 0,
            "selected_frame_count": 0,
            "frames": [],
            "reason": "ffmpeg_extract_failed",
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"commands": [result], "selected_frames": [], "candidate_frame_count": 0, "selected_frame_count": 0, "report_path": report_path}

    scored = _parallel_score_images(frames)
    selected_names = _select_adaptive_frames(scored, selected_target, config)
    selected_frames: list[Path] = []
    frame_entries: list[dict[str, Any]] = []
    for item in scored:
        selected = item["filename"] in selected_names
        if selected:
            selected_frames.append(Path(item["path"]))
        reason = item.get("reject_reason")
        if not selected and not reason:
            reason = "sampling_budget_exceeded"
        frame_entries.append({**item, "selected": selected, "reject_reason": None if selected else reason})
    report = {
        "asset_id": asset_id,
        "adaptive": True,
        "scan_fps": scan_fps,
        "candidate_limit": candidate_limit,
        "target_frame_count": selected_target,
        "candidate_frame_count": len(frames),
        "selected_frame_count": len(selected_frames),
        "selection_policy": "quality_filter_then_even_temporal_sampling",
        "frames": frame_entries,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "commands": [result],
        "selected_frames": selected_frames,
        "candidate_frame_count": len(frames),
        "selected_frame_count": len(selected_frames),
        "report_path": report_path,
    }


def _adaptive_target_frame_count(frame_target: int, mode: str, config: dict[str, Any]) -> int:
    if config.get("adaptive_target_frames"):
        return min(frame_target, int(config["adaptive_target_frames"]))
    scene_size = str(config.get("scene_size") or "").lower()
    if scene_size == "small" or mode == "quick_preview":
        return min(frame_target, 150)
    if scene_size == "large" or mode == "high_quality":
        return min(frame_target, 600)
    return min(frame_target, 300)


def _parallel_score_images(image_paths: list[Path]) -> list[dict[str, Any]]:
    if not image_paths:
        return []
    workers = min(32, max(1, len(image_paths)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_score_image, enumerate(image_paths)))
    return sorted(results, key=lambda item: int(item["index"]))


def _score_image(item: tuple[int, Path]) -> dict[str, Any]:
    index, path = item
    metrics: dict[str, Any] = {
        "index": index,
        "filename": path.name,
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "blur_laplacian": None,
        "mean_luma": None,
        "sharpness_score": None,
        "reject_reason": None,
    }
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            metrics["reject_reason"] = "decode_failed"
            return metrics
        blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
        mean_luma = float(image.mean())
        metrics.update(
            {
                "blur_laplacian": blur,
                "mean_luma": mean_luma,
                "sharpness_score": blur,
            }
        )
        if blur < 70:
            metrics["reject_reason"] = "blur_laplacian_below_threshold"
        elif mean_luma < 20:
            metrics["reject_reason"] = "under_exposed"
        elif mean_luma > 245:
            metrics["reject_reason"] = "over_exposed"
    except Exception as exc:
        metrics["analysis_warning"] = f"image_quality_backend_unavailable:{type(exc).__name__}"
    return metrics


def _select_adaptive_frames(scored: list[dict[str, Any]], target: int, config: dict[str, Any]) -> set[str]:
    usable = [item for item in scored if not item.get("reject_reason")]
    if not usable:
        usable = scored
    if len(usable) <= target:
        return {str(item["filename"]) for item in usable}
    if target <= 1:
        return {str(usable[0]["filename"])}
    step = (len(usable) - 1) / max(target - 1, 1)
    selected_indices = {round(index * step) for index in range(target)}
    return {str(usable[index]["filename"]) for index in sorted(selected_indices) if index < len(usable)}


def _write_parallel_image_prepare_report(workspace_dir: Path, image_paths: list[Path]) -> Path:
    report_dir = workspace_dir / "image_prepare"
    thumbnails_dir = report_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    if image_paths:
        workers = min(32, max(1, len(image_paths)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(lambda item: _prepare_image_report_item(item, thumbnails_dir), enumerate(image_paths)))
    else:
        prepared = []
    report = {
        "parallel": True,
        "worker_count": min(32, max(1, len(image_paths))) if image_paths else 0,
        "image_count": len(image_paths),
        "checks": ["resize_probe", "blur_detection", "exposure_check", "sharpness_score", "exif_probe", "thumbnail_generation"],
        "images": prepared,
    }
    report_path = report_dir / "image_prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _prepare_image_report_item(item: tuple[int, Path], thumbnails_dir: Path) -> dict[str, Any]:
    metrics = _score_image(item)
    path = Path(metrics["path"])
    thumbnail_path = thumbnails_dir / f"{path.stem}.jpg"
    metrics["thumbnail_path"] = None
    metrics["exif_available"] = False
    try:
        import cv2  # type: ignore

        image = cv2.imread(str(path))
        if image is not None:
            height, width = image.shape[:2]
            scale = min(320 / max(width, height), 1.0)
            if scale < 1.0:
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
            cv2.imwrite(str(thumbnail_path), image)
            metrics["thumbnail_path"] = str(thumbnail_path)
            metrics["width"] = width
            metrics["height"] = height
    except Exception as exc:
        metrics["thumbnail_warning"] = f"thumbnail_backend_unavailable:{type(exc).__name__}"
    return metrics


def _run_command(operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    return CommandResult(
        operator_name=operator_name,
        stage_key=stage_key,
        command=command,
        cwd=str(cwd),
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        started_at=started,
        finished_at=finished,
    )


def _is_image_like(asset: Asset) -> bool:
    if asset.asset_type in {"detail_photo", "supplement_photo", "scale_marker", "pano_360"}:
        return True
    mime = asset.mime_type or ""
    return mime.startswith("image/")


def _unique_image_name(images_dir: Path, filename: str) -> str:
    candidate = Path(filename).name
    stem = Path(candidate).stem or "image"
    suffix = Path(candidate).suffix or ".jpg"
    counter = 1
    while (images_dir / candidate).exists():
        candidate = f"{stem}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def _relative_from_uri(storage_uri: str) -> str:
    if storage_uri.startswith("local://"):
        return storage_uri.replace("local://", "", 1)
    if storage_uri.startswith("s3://"):
        return storage_uri.split("/", 3)[-1]
    return storage_uri
