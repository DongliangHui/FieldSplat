from __future__ import annotations

import binascii
import json
import math
import os
import shutil
import subprocess
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.models import Workflow
from app.operators.preprocess import PreprocessRunResult
from app.services.stage_cache import StageCache


DEFAULT_SCOPE_CONFIG: dict[str, Any] = {
    "reconstruction_scope": "roi_first",
    "reconstruction_roi": "auto",
    "preserve_context": True,
    "context_quality": "low",
    "foreground_loss_weight": 1.0,
    "background_loss_weight": 0.15,
    "prune_background_gaussians": True,
    "export_full_debug_model": True,
    "publish_default": "raw_model",
    "foreground_ratio": 0.68,
    "apply_masks_to_colmap": False,
    "apply_masks_to_training": False,
    "max_generated_masks": 1200,
    "viewer_max_gaussians": 350_000,
    "viewer_max_size_mb": 160,
    "layered_loading": {
        "enabled": True,
        "preserve_raw_model": True,
        "viewer_is_preview_proxy": True,
    },
}


@dataclass(frozen=True)
class SubjectMaskResult:
    workspace_dir: Path
    masks_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    cache_hit: bool
    cache_key: str


@dataclass(frozen=True)
class SpatialCropResult:
    workspace_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    cache_hit: bool
    cache_key: str


@dataclass(frozen=True)
class GaussianPruningResult:
    workspace_dir: Path
    report_path: Path
    outputs: dict[str, Path]
    report: dict[str, Any]
    cache_hit: bool
    cache_key: str


class SubjectMaskGenerationOperator:
    name = "scope.subject_mask_generation"
    queue = "preprocess"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self, workflow: Workflow, preprocess: PreprocessRunResult) -> SubjectMaskResult:
        scope_config = scope_config_for(workflow, self.settings)
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "subject_mask_generation"
        masks_dir = workspace_dir / "masks"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[*preprocess.image_paths],
            stage_config={
                "scope_config": scope_config,
                "operator_config": self.settings.engine_config.get("operators", {}).get("subject_mask_generation", {}) or {},
                "semantic_dependencies": _semantic_dependency_fingerprint(self.settings),
            },
            algorithm_version="subject-mask-contract-v2",
        )
        manifest_path = workspace_dir / "mask_manifest.json"
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return SubjectMaskResult(workspace_dir, masks_dir, manifest_path, manifest, True, cache_entry.cache_key)

        operator_config = self.settings.engine_config.get("operators", {}).get("subject_mask_generation", {}) or {}
        external_manifest = _configured_mask_manifest(scope_config)
        if external_manifest and Path(external_manifest).exists():
            manifest = _load_external_mask_manifest(workflow, preprocess, scope_config, Path(external_manifest), masks_dir)
        else:
            external_command_manifest = _run_configured_subject_mask_command(operator_config, workflow, preprocess, scope_config, workspace_dir, masks_dir, self.settings)
            external_unavailable_manifest = None
            if external_command_manifest is not None and external_command_manifest.pop("_fallback_to_heuristic", False):
                external_unavailable_manifest = external_command_manifest
                external_command_manifest = None
            if external_command_manifest is not None:
                manifest = external_command_manifest
            else:
                manifest = _generate_heuristic_mask_manifest(workflow, preprocess, scope_config, masks_dir)
                if external_unavailable_manifest:
                    manifest["external_semantic_mask"] = external_unavailable_manifest

        manifest.update({"cache_hit": False, "cache_key": cache_entry.cache_key})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cache.save(cache_entry, workspace_dir, metadata=manifest)
        return SubjectMaskResult(workspace_dir, masks_dir, manifest_path, manifest, False, cache_entry.cache_key)


class SpatialCropOperator:
    name = "scope.spatial_crop"
    queue = "cpu"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self, workflow: Workflow, pose_quality: dict[str, Any], subject_mask: SubjectMaskResult) -> SpatialCropResult:
        scope_config = scope_config_for(workflow, self.settings)
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "spatial_crop"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[subject_mask.manifest_path, pose_quality],
            stage_config=scope_config,
            algorithm_version="spatial-crop-contract-v1",
        )
        manifest_path = workspace_dir / "spatial_crop_manifest.json"
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return SpatialCropResult(workspace_dir, manifest_path, manifest, True, cache_entry.cache_key)

        foreground_ratio = float(subject_mask.manifest.get("foreground_ratio") or scope_config["foreground_ratio"])
        registered = int(pose_quality.get("registered_camera_count") or 0)
        sparse_points = int(pose_quality.get("sparse_point_count") or 0)
        bbox_scale = max(1.0, math.log10(max(sparse_points, 10)))
        manifest = {
            "workflow_id": workflow.id,
            "operator": self.name,
            "schema": "fieldsplat.spatial_crop.v1",
            "enabled": True,
            "crop_policy": "subject_first_context_preserved",
            "crop_type": "largest_foreground_cluster_bbox",
            "source": "pose_quality_and_subject_mask",
            "foreground_ratio": round(foreground_ratio, 4),
            "background_ratio": round(1.0 - foreground_ratio, 4),
            "registered_camera_count": registered,
            "sparse_point_count": sparse_points,
            "estimated_subject_bbox": {
                "type": "axis_aligned_bbox",
                "center": [0.0, 0.0, 0.0],
                "extent": [round(bbox_scale, 3), round(bbox_scale, 3), round(max(0.5, bbox_scale * 0.5), 3)],
                "basis": "heuristic until sparse foreground labels are available",
            },
            "actions": [
                "prefer foreground sparse cluster",
                "keep low-resolution context layer",
                "mark far background for stronger gaussian pruning",
            ],
            "applied_to_dataset": False,
            "reason": "spatial crop manifest is available; destructive sparse-model crop requires foreground-labeled points",
            "cache_hit": False,
            "cache_key": cache_entry.cache_key,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cache.save(cache_entry, workspace_dir, metadata=manifest)
        return SpatialCropResult(workspace_dir, manifest_path, manifest, False, cache_entry.cache_key)


class GaussianPruningOperator:
    name = "scope.gaussian_pruning"
    queue = "export"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(
        self,
        workflow: Workflow,
        *,
        splat_path: Path | None,
        subject_mask: SubjectMaskResult,
        spatial_crop: SpatialCropResult,
        gaussian_quality: dict[str, Any],
    ) -> GaussianPruningResult:
        scope_config = scope_config_for(workflow, self.settings)
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "gaussian_pruning"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[splat_path or "missing_splat", subject_mask.manifest_path, spatial_crop.manifest_path],
            stage_config={**scope_config, "gaussian_quality": gaussian_quality},
            algorithm_version="gaussian-pruning-layered-export-v1",
        )
        report_path = workspace_dir / "gaussian_pruning_report.json"
        output_names = {
            "raw_model": "raw_model.ply",
            "model_full": "model_full.ply",
            "model_roi": "model_roi.ply",
            "subject_model": "subject_model.ply",
            "viewer_model": "viewer_model.ply",
            "context_model_lowres": "context_model_lowres.ply",
            "full_model_debug": "full_model_debug.ply",
            "scale_outlier_cleaned": "scale_outlier_cleaned.ply",
        }
        outputs = {key: workspace_dir / filename for key, filename in output_names.items()}
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return GaussianPruningResult(workspace_dir, report_path, outputs, report, True, cache_entry.cache_key)

        if not splat_path or not splat_path.exists() or splat_path.stat().st_size <= 0:
            report = {
                "workflow_id": workflow.id,
                "operator": self.name,
                "passed": False,
                "reason": "splat_missing_or_empty",
                "cache_hit": False,
                "cache_key": cache_entry.cache_key,
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return GaussianPruningResult(workspace_dir, report_path, outputs, report, False, cache_entry.cache_key)

        foreground_ratio = float(subject_mask.manifest.get("foreground_ratio") or scope_config["foreground_ratio"])
        foreground_ratio = min(0.98, max(0.05, foreground_ratio))
        viewer_max_gaussians = int(scope_config.get("viewer_max_gaussians") or DEFAULT_SCOPE_CONFIG["viewer_max_gaussians"])
        viewer_max_size_mb = int(scope_config.get("viewer_max_size_mb") or DEFAULT_SCOPE_CONFIG["viewer_max_size_mb"])
        cleanup_config = _scale_outlier_cleanup_config(self.settings)
        pruning_summary = _write_layered_ply_outputs(
            splat_path,
            outputs,
            foreground_ratio,
            viewer_max_gaussians,
            viewer_max_size_mb,
            gaussian_quality=gaussian_quality,
            cleanup_config=cleanup_config,
        )
        sizes = {key: _size_payload(path) for key, path in outputs.items() if path.exists()}
        cleanup_summary = pruning_summary.get("scale_outlier_cleanup") or {}
        raw_gaussian_count = pruning_summary.get("raw_gaussian_count") or pruning_summary.get("source_gaussian_count")
        layered_loading = _layered_loading_payload(scope_config, outputs, sizes, pruning_summary)
        report = {
            "workflow_id": workflow.id,
            "operator": self.name,
            "schema": "fieldsplat.gaussian_pruning.v1",
            "passed": bool(outputs["subject_model"].exists() and outputs["subject_model"].stat().st_size > 0),
            "reconstruction_scope": scope_config["reconstruction_scope"],
            "publish_default": scope_config["publish_default"],
            "viewer_default": "viewer_model",
            "viewer_model_role": "preview_proxy",
            "quality_model_not_capped_for_viewer": True,
            "viewer_max_gaussians": viewer_max_gaussians,
            "viewer_max_size_mb": viewer_max_size_mb,
            "layered_loading": layered_loading,
            "preserve_context": bool(scope_config["preserve_context"]),
            "context_quality": scope_config["context_quality"],
            "foreground_ratio": round(foreground_ratio, 4),
            "background_ratio": round(1.0 - foreground_ratio, 4),
            "roi_coverage_score": int(round(foreground_ratio * 100)),
            "foreground_loss_weight": float(scope_config["foreground_loss_weight"]),
            "background_loss_weight": float(scope_config["background_loss_weight"]),
            "prune_background_gaussians": bool(scope_config["prune_background_gaussians"]),
            "pruning_mode": pruning_summary["mode"],
            "scale_outlier_cleanup_triggered": bool(cleanup_summary.get("triggered")),
            "scale_outlier_cleanup_applied": bool(cleanup_summary.get("applied")),
            "scale_outlier_cleanup": cleanup_summary,
            "pruned_gaussian_count": pruning_summary.get("pruned_gaussian_count", 0),
            "source_gaussian_count": pruning_summary.get("source_gaussian_count"),
            "raw_gaussian_count": raw_gaussian_count,
            "subject_gaussian_count": pruning_summary.get("subject_gaussian_count"),
            "viewer_gaussian_count": pruning_summary.get("viewer_gaussian_count"),
            "context_gaussian_count": pruning_summary.get("context_gaussian_count"),
            "raw_model_size": sizes.get("raw_model", {}),
            "final_subject_model_size": sizes.get("subject_model", {}),
            "viewer_model_size": sizes.get("viewer_model", {}),
            "context_model_size": sizes.get("context_model_lowres", {}),
            "full_debug_model_size": sizes.get("full_model_debug", {}),
            "outputs": {key: str(path) for key, path in outputs.items()},
            "subject_mask": {
                "manifest_path": str(subject_mask.manifest_path),
                "method": subject_mask.manifest.get("method"),
                "semantic_model_used": subject_mask.manifest.get("semantic_model_used"),
            },
            "spatial_crop": spatial_crop.manifest,
            "gaussian_quality": gaussian_quality,
            "notes": pruning_summary.get("notes", []),
            "cache_hit": False,
            "cache_key": cache_entry.cache_key,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        cache.save(cache_entry, workspace_dir, metadata=report)
        return GaussianPruningResult(workspace_dir, report_path, outputs, report, False, cache_entry.cache_key)


def scope_config_for(workflow: Workflow, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    engine_scope = settings.engine_config.get("reconstruction_scope", {}) or {}
    operator_scope = (settings.engine_config.get("operators", {}).get("subject_mask_generation", {}) or {}).get("scope_defaults", {}) or {}
    pruning_scope = settings.engine_config.get("operators", {}).get("gaussian_pruning", {}) or {}
    workflow_config = workflow.config_json or {}
    workflow_scope = workflow_config.get("reconstruction_scope") or {}
    if isinstance(workflow_scope, str):
        workflow_scope = {"reconstruction_scope": workflow_scope}
    workflow_direct = {
        key: workflow_config[key]
        for key in DEFAULT_SCOPE_CONFIG
        if key in workflow_config
    }
    roi = workflow_config.get("reconstruction_roi")
    roi_config = {"reconstruction_roi": roi} if roi else {}
    merged = {**DEFAULT_SCOPE_CONFIG, **engine_scope, **operator_scope, **pruning_scope, **workflow_scope, **workflow_direct, **roi_config}
    return merged


def _configured_mask_manifest(scope_config: dict[str, Any]) -> str | None:
    roi = scope_config.get("reconstruction_roi")
    if isinstance(roi, dict):
        manifest = roi.get("mask_manifest_path") or roi.get("mask_manifest")
        if manifest:
            return str(manifest)
    manifest = scope_config.get("mask_manifest_path")
    return str(manifest) if manifest else None


def _run_configured_subject_mask_command(
    operator_config: dict[str, Any],
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    scope_config: dict[str, Any],
    workspace_dir: Path,
    masks_dir: Path,
    settings: Settings,
) -> dict[str, Any] | None:
    command_template = operator_config.get("command")
    if not command_template:
        return None
    output_manifest = workspace_dir / "mask_manifest.external.json"
    semantic_values = _semantic_mask_template_values(settings)
    prompt = str(operator_config.get("prompt") or scope_config.get("subject_prompt") or "building. structure. object. foreground.")
    if prompt and not prompt.endswith("."):
        prompt += "."
    values = {
        **semantic_values,
        "images_dir": str(preprocess.images_dir),
        "dataset_dir": str(preprocess.dataset_dir),
        "workspace_dir": str(workspace_dir),
        "masks_dir": str(masks_dir),
        "output_manifest": str(output_manifest),
        "output_report": str(workspace_dir / "subject_mask_report.external.json"),
        "prompt": prompt,
        "max_images": str(int(operator_config.get("semantic_max_images") or 1200)),
        "box_threshold": str(operator_config.get("box_threshold") or semantic_values.get("box_threshold") or 0.3),
        "text_threshold": str(operator_config.get("text_threshold") or semantic_values.get("text_threshold") or 0.25),
    }
    command = [_format_template_part(str(part), values) for part in command_template]
    completed = subprocess.run(command, cwd=workspace_dir, capture_output=True, text=True, check=False)
    if completed.returncode == 0 and output_manifest.exists():
        manifest = _load_external_mask_manifest(workflow, preprocess, scope_config, output_manifest, masks_dir)
        manifest.update(
            {
                "method": manifest.get("method") or "external_semantic_mask_command",
                "source": "configured_external_command",
                "command": command,
                "exit_code": completed.returncode,
                "stderr_tail": completed.stderr[-1000:] if completed.stderr else "",
            }
        )
        return manifest
    if completed.returncode == 2 and output_manifest.exists():
        manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
        manifest.update(
            {
                "workflow_id": workflow.id,
                "operator": SubjectMaskGenerationOperator.name,
                "method": "external_semantic_mask_unavailable",
                "source": "configured_external_command",
                "semantic_model_used": False,
                "command": command,
                "exit_code": completed.returncode,
                "stderr_tail": completed.stderr[-1000:] if completed.stderr else "",
                "_fallback_to_heuristic": True,
            }
        )
        return manifest
    return {
        "workflow_id": workflow.id,
        "operator": SubjectMaskGenerationOperator.name,
        "method": "external_semantic_mask_failed",
        "source": "configured_external_command",
        "semantic_model_used": False,
        "command": command,
        "exit_code": completed.returncode,
        "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        "_fallback_to_heuristic": True,
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


def _load_external_mask_manifest(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    scope_config: dict[str, Any],
    external_path: Path,
    masks_dir: Path,
) -> dict[str, Any]:
    loaded = json.loads(external_path.read_text(encoding="utf-8"))
    foreground_ratio = _safe_ratio(loaded.get("foreground_ratio"), scope_config["foreground_ratio"])
    return {
        "workflow_id": workflow.id,
        "operator": SubjectMaskGenerationOperator.name,
        "schema": "fieldsplat.mask_manifest.v1",
        "method": loaded.get("method") or "external_mask_manifest",
        "semantic_model_used": bool(loaded.get("semantic_model_used")),
        "mask_format": loaded.get("mask_format") or "external",
        "input_image_count": len(preprocess.image_paths),
        "mask_count": len(loaded.get("images") or []),
        "foreground_ratio": foreground_ratio,
        "background_ratio": round(1.0 - foreground_ratio, 4),
        "masks_dir": str(masks_dir),
        "external_manifest_path": str(external_path),
        "manual_annotations": loaded.get("manual_annotations") or [],
        "colmap_masking": {
            "supported": True,
            "apply_to_colmap": bool(scope_config.get("apply_masks_to_colmap")),
            "mask_path": str(masks_dir) if scope_config.get("apply_masks_to_colmap") else None,
        },
        "training_masking": {
            "supported": True,
            "apply_to_training": bool(scope_config.get("apply_masks_to_training")),
            "foreground_loss_weight": scope_config["foreground_loss_weight"],
            "background_loss_weight": scope_config["background_loss_weight"],
        },
        "images": loaded.get("images") or [],
        "source": "configured_external_manifest",
    }


def _generate_heuristic_mask_manifest(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    scope_config: dict[str, Any],
    masks_dir: Path,
) -> dict[str, Any]:
    image_paths = list(preprocess.image_paths)
    max_masks = int(scope_config.get("max_generated_masks") or len(image_paths))
    foreground_ratio = _safe_ratio(scope_config.get("foreground_ratio"), DEFAULT_SCOPE_CONFIG["foreground_ratio"])
    width = 64
    height = 64
    rect_ratio = math.sqrt(foreground_ratio)
    rect_w = max(4, min(width, int(width * rect_ratio)))
    rect_h = max(4, min(height, int(height * rect_ratio)))
    x0 = (width - rect_w) // 2
    y0 = (height - rect_h) // 2
    image_entries: list[dict[str, Any]] = []
    for image_path in image_paths[:max_masks]:
        mask_path = masks_dir / f"{Path(image_path).stem}.png"
        _write_rect_mask_png(mask_path, width, height, x0, y0, rect_w, rect_h)
        image_entries.append(
            {
                "image_name": Path(image_path).name,
                "mask_path": str(mask_path),
                "foreground_ratio": round(foreground_ratio, 4),
                "background_ratio": round(1.0 - foreground_ratio, 4),
                "method": "heuristic_center_roi",
            }
        )
    return {
        "workflow_id": workflow.id,
        "operator": SubjectMaskGenerationOperator.name,
        "schema": "fieldsplat.mask_manifest.v1",
        "method": "heuristic_center_roi_contract",
        "semantic_model_used": False,
        "semantic_model_configured": False,
        "mask_format": "png_64x64_binary",
        "input_image_count": len(image_paths),
        "mask_count": len(image_entries),
        "foreground_ratio": round(foreground_ratio, 4),
        "background_ratio": round(1.0 - foreground_ratio, 4),
        "irrelevant_environment_ratio": round(1.0 - foreground_ratio, 4),
        "masks_dir": str(masks_dir),
        "manual_annotations": [],
        "colmap_masking": {
            "supported": True,
            "apply_to_colmap": False,
            "reason": "heuristic masks are low-resolution advisory masks; configure full-resolution SAM/SAM2/GroundingDINO masks to enable COLMAP masked feature extraction",
        },
        "training_masking": {
            "supported": True,
            "apply_to_training": False,
            "foreground_loss_weight": scope_config["foreground_loss_weight"],
            "background_loss_weight": scope_config["background_loss_weight"],
            "reason": "heuristic masks are recorded for scope and export policy; full-resolution masks are required for loss masking",
        },
        "images": image_entries,
        "source": "auto_without_external_segmenter",
        "notes": [
            "This is a deterministic scope contract and preview mask set, not a semantic SAM result.",
            "Configure external full-resolution masks to make COLMAP/3DGS consume masks directly.",
        ],
    }


def _scale_outlier_cleanup_config(settings: Settings) -> dict[str, Any]:
    gate_config = settings.engine_config.get("gaussian_quality_gate", {}) or {}
    cleanup = gate_config.get("scale_outlier_cleanup", {}) or {}
    if not cleanup:
        hard_fail = gate_config.get("hard_fail", {}) or {}
        cleanup = {
            "enabled": True,
            "scale_p99_over_p50_gt": hard_fail.get("scale_p99_over_p50_gt", 80),
            "scale_max_over_p50_gt": hard_fail.get("scale_max_over_p50_gt", 300),
            "scale_outlier_ratio_gt": hard_fail.get("scale_outlier_ratio_gt", 0.03),
        }
    return cleanup


def _write_layered_ply_outputs(
    source: Path,
    outputs: dict[str, Path],
    foreground_ratio: float,
    viewer_max_gaussians: int,
    viewer_max_size_mb: int,
    *,
    gaussian_quality: dict[str, Any] | None = None,
    cleanup_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _link_or_copy(source, outputs["raw_model"])
    _link_or_copy(source, outputs["full_model_debug"])
    working_source = source
    cleanup_summary = {"triggered": False, "applied": False, "reason": "not_required"}
    if _scale_outlier_cleanup_required(gaussian_quality or {}, cleanup_config or {}):
        cleanup_target = outputs.get("scale_outlier_cleaned") or outputs["subject_model"].with_name("scale_outlier_cleaned.ply")
        cleanup_summary = _write_scale_outlier_cleanup_ply(source, cleanup_target, gaussian_quality or {}, cleanup_config or {})
        if cleanup_summary.get("applied") and cleanup_target.exists() and cleanup_target.stat().st_size > 0:
            working_source = cleanup_target
    _link_or_copy(working_source, outputs["model_full"])
    parsed = _try_write_pruned_ply(working_source, outputs["subject_model"], outputs["context_model_lowres"], foreground_ratio)
    if parsed["mode"] == "unsupported_identity_fallback":
        _link_or_copy(working_source, outputs["subject_model"])
        _write_empty_ascii_ply(outputs["context_model_lowres"])
    _link_or_copy(outputs["subject_model"], outputs["model_roi"])
    viewer_summary = _try_write_sampled_ply(outputs["subject_model"], outputs["viewer_model"], max_vertices=viewer_max_gaussians, max_size_mb=viewer_max_size_mb)
    parsed.update(
        {
            "scale_outlier_cleanup": cleanup_summary,
            "raw_gaussian_count": parsed.get("source_gaussian_count"),
            "viewer_gaussian_count": viewer_summary.get("viewer_gaussian_count"),
            "viewer_sampling_mode": viewer_summary.get("mode"),
            "viewer_sampling_step": viewer_summary.get("sampling_step"),
        }
    )
    parsed.setdefault("notes", []).extend(viewer_summary.get("notes", []))
    return parsed


def _layered_loading_payload(
    scope_config: dict[str, Any],
    outputs: dict[str, Path],
    sizes: dict[str, dict[str, Any]],
    pruning_summary: dict[str, Any],
) -> dict[str, Any]:
    layered_config = scope_config.get("layered_loading") or {}
    enabled = layered_config.get("enabled", True) is not False
    layers = [
        {
            "id": "viewer_model",
            "role": "interactive_preview",
            "path": str(outputs["viewer_model"]),
            "gaussian_count": pruning_summary.get("viewer_gaussian_count"),
            "size": sizes.get("viewer_model", {}),
            "quality_role": "budgeted_preview_proxy",
            "load_priority": 0,
        },
        {
            "id": "raw_model",
            "role": "canonical_quality_model",
            "path": str(outputs["raw_model"]),
            "gaussian_count": pruning_summary.get("raw_gaussian_count") or pruning_summary.get("source_gaussian_count"),
            "size": sizes.get("raw_model", {}),
            "quality_role": "full_quality_not_viewer_capped",
            "load_priority": 1,
        },
        {
            "id": "subject_model",
            "role": "roi_quality_layer",
            "path": str(outputs["subject_model"]),
            "gaussian_count": pruning_summary.get("subject_gaussian_count"),
            "size": sizes.get("subject_model", {}),
            "quality_role": "foreground_roi_layer",
            "load_priority": 2,
        },
        {
            "id": "context_model_lowres",
            "role": "context_reference",
            "path": str(outputs["context_model_lowres"]),
            "gaussian_count": pruning_summary.get("context_gaussian_count"),
            "size": sizes.get("context_model_lowres", {}),
            "quality_role": "lowres_context_layer",
            "load_priority": 3,
        },
    ]
    return {
        "enabled": enabled,
        "strategy": "load_viewer_proxy_first_then_full_quality_layers",
        "preserve_raw_model": layered_config.get("preserve_raw_model", True) is not False,
        "viewer_is_preview_proxy": layered_config.get("viewer_is_preview_proxy", True) is not False,
        "quality_ceiling_layer": "raw_model",
        "interactive_default_layer": "viewer_model",
        "layers": layers,
    }


def _scale_outlier_cleanup_required(gaussian_quality: dict[str, Any], cleanup_config: dict[str, Any]) -> bool:
    if cleanup_config.get("enabled") is False:
        return False
    triggers = [str(value) for value in gaussian_quality.get("quality_triggers") or []]
    legacy_issues = [str(value) for value in gaussian_quality.get("issues") or []]
    if any(trigger.startswith("splat_scale") for trigger in [*triggers, *legacy_issues]):
        return True
    ratio = float(gaussian_quality.get("scale_outlier_ratio") or 0.0)
    max_ratio = float(cleanup_config.get("scale_outlier_ratio_gt") or 0.03)
    p99_over_p50 = float(gaussian_quality.get("scale_p99_over_p50") or 0.0)
    max_p99_over_p50 = float(cleanup_config.get("scale_p99_over_p50_gt") or 80.0)
    max_over_p50 = float(gaussian_quality.get("scale_max_over_p50") or 0.0)
    max_max_over_p50 = float(cleanup_config.get("scale_max_over_p50_gt") or 300.0)
    return ratio > max_ratio or p99_over_p50 > max_p99_over_p50 or max_over_p50 > max_max_over_p50


def _write_scale_outlier_cleanup_ply(source: Path, target: Path, gaussian_quality: dict[str, Any], cleanup_config: dict[str, Any]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as fh:
        prefix = fh.read(512 * 1024)
    header_end = prefix.find(b"end_header\n")
    newline_len = len(b"end_header\n")
    if header_end < 0:
        header_end = prefix.find(b"end_header\r\n")
        newline_len = len(b"end_header\r\n")
    if header_end < 0:
        return {"triggered": True, "applied": False, "reason": "ply_header_end_not_found", "gaussian_quality": gaussian_quality}
    header_bytes = prefix[: header_end + newline_len]
    try:
        header_text = header_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return {"triggered": True, "applied": False, "reason": "ply_header_not_ascii", "gaussian_quality": gaussian_quality}
    info = _parse_ply_header(header_text)
    vertex_count = int(info.get("vertex_count") or 0)
    row_size = int(info.get("vertex_row_size") or 0)
    if info.get("format") != "binary_little_endian" or vertex_count <= 0 or row_size <= 0:
        return {
            "triggered": True,
            "applied": False,
            "reason": "unsupported_cleanup_ply_format",
            "format": info.get("format"),
            "vertex_count": vertex_count,
            "gaussian_quality": gaussian_quality,
        }
    layout = _binary_property_layout(info.get("property_layout") or [])
    required = {"opacity", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - set(layout))
    if missing:
        return {"triggered": True, "applied": False, "reason": "cleanup_properties_missing", "missing_properties": missing, "gaussian_quality": gaussian_quality}
    if any(layout[name]["type"] not in {"float", "float32"} for name in required):
        return {"triggered": True, "applied": False, "reason": "cleanup_properties_not_float32", "gaussian_quality": gaussian_quality}

    scan = _scan_scale_cleanup_stats(source, len(header_bytes), row_size, vertex_count, layout)
    scale_p50 = _percentile(scan["scale_radii"], 50)
    scale_p95 = _percentile(scan["scale_radii"], 95)
    scale_p99 = _percentile(scan["scale_radii"], 99)
    opacity_threshold = _percentile(scan["opacities"], float(cleanup_config.get("low_opacity_percentile") or 25.0))
    threshold = _scale_cleanup_threshold(scale_p50, scale_p95, scale_p99, cleanup_config)
    if threshold is None or threshold <= 0 or not math.isfinite(threshold):
        return {"triggered": True, "applied": False, "reason": "cleanup_threshold_unavailable", "gaussian_quality": gaussian_quality}

    shrink_high_opacity = cleanup_config.get("shrink_high_opacity", True) is not False
    log_threshold = math.log(threshold)
    temp_body = target.with_suffix(target.suffix + ".body.tmp")
    kept_count = 0
    pruned_count = 0
    shrunk_count = 0
    high_opacity_outlier_count = 0
    low_opacity_outlier_count = 0
    max_after = 0.0
    with source.open("rb") as src, temp_body.open("wb") as dst:
        src.seek(len(header_bytes))
        for _ in range(vertex_count):
            row = src.read(row_size)
            if len(row) != row_size:
                break
            values = _row_scale_opacity(row, layout)
            if values is None:
                continue
            radius, opacity = values
            is_outlier = radius > threshold
            is_low_opacity = opacity_threshold is not None and opacity < opacity_threshold
            if is_outlier and is_low_opacity:
                low_opacity_outlier_count += 1
                pruned_count += 1
                continue
            mutable = bytearray(row)
            if is_outlier:
                high_opacity_outlier_count += 1
                if shrink_high_opacity:
                    for name in ("scale_0", "scale_1", "scale_2"):
                        offset = int(layout[name]["offset"])
                        value = struct.unpack_from("<f", mutable, offset)[0]
                        if math.isfinite(value) and value > log_threshold:
                            struct.pack_into("<f", mutable, offset, log_threshold)
                    shrunk_count += 1
                    radius = min(radius, threshold)
            if math.isfinite(radius):
                max_after = max(max_after, radius)
            dst.write(mutable)
            kept_count += 1

    target.write_bytes(_header_with_vertex_count(header_text, kept_count))
    with temp_body.open("rb") as body, target.open("ab") as final:
        shutil.copyfileobj(body, final, length=1024 * 1024)
    temp_body.unlink(missing_ok=True)
    applied = pruned_count > 0 or shrunk_count > 0
    if not applied:
        _link_or_copy(source, target)
    return {
        "triggered": True,
        "applied": applied,
        "reason": "scale_outlier_cleanup_applied" if applied else "no_scale_outliers_above_cleanup_threshold",
        "source_gaussian_count": vertex_count,
        "cleaned_gaussian_count": kept_count if applied else vertex_count,
        "pruned_low_opacity_large_scale_count": pruned_count,
        "shrunk_high_opacity_large_scale_count": shrunk_count,
        "low_opacity_large_scale_count": low_opacity_outlier_count,
        "suspected_geometry_patch_count": high_opacity_outlier_count,
        "depth_or_mask_review_required": high_opacity_outlier_count > 0,
        "scale_mean": scan["scale_mean"],
        "scale_p50": scale_p50,
        "scale_p95": scale_p95,
        "scale_p99": scale_p99,
        "scale_max_before": scan["scale_max"],
        "scale_max_after": max_after if applied else scan["scale_max"],
        "scale_cleanup_threshold": threshold,
        "opacity_low_threshold": opacity_threshold,
        "outlier_rule": "radius > max(P95, min(P99, P95 * p95_multiplier, P50 * median_multiplier)); low-opacity outliers are pruned, high-opacity outliers are shrunk and flagged.",
        "next_repair_actions_if_visual_quality_fails": [
            "recheck_camera_poses",
            "rerun_feature_matching_and_sfm",
            "increase_image_overlap",
            "train_chunked",
            "apply_dynamic_object_masks",
            "verify_camera_model_and_distortion",
        ],
        "output_path": str(target),
    }


def _try_write_pruned_ply(source: Path, subject_path: Path, context_path: Path, foreground_ratio: float) -> dict[str, Any]:
    with source.open("rb") as fh:
        prefix = fh.read(512 * 1024)
    header_end = prefix.find(b"end_header\n")
    newline_len = len(b"end_header\n")
    if header_end < 0:
        header_end = prefix.find(b"end_header\r\n")
        newline_len = len(b"end_header\r\n")
    if header_end < 0:
        return {"mode": "unsupported_identity_fallback", "notes": ["PLY header end not found."]}
    header_bytes = prefix[: header_end + newline_len]
    try:
        header_text = header_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return {"mode": "unsupported_identity_fallback", "notes": ["PLY header is not ASCII-decodable."]}
    info = _parse_ply_header(header_text)
    vertex_count = int(info.get("vertex_count") or 0)
    if vertex_count <= 0:
        _write_empty_ascii_ply(subject_path)
        _write_empty_ascii_ply(context_path)
        return {"mode": "empty_ply", "source_gaussian_count": 0, "subject_gaussian_count": 0, "context_gaussian_count": 0, "pruned_gaussian_count": 0}
    subject_count = max(1, min(vertex_count, int(vertex_count * foreground_ratio)))
    context_count = max(0, min(vertex_count - subject_count, int(vertex_count * max(0.02, (1.0 - foreground_ratio) * 0.15))))
    if info["format"] == "binary_little_endian":
        row_size = int(info.get("vertex_row_size") or 0)
        if row_size <= 0:
            return {"mode": "unsupported_identity_fallback", "source_gaussian_count": vertex_count, "notes": ["Unsupported PLY property layout."]}
        _write_binary_vertex_slice(source, subject_path, header_text, len(header_bytes), row_size, subject_count, subject_count)
        _write_binary_vertex_slice(source, context_path, header_text, len(header_bytes), row_size, context_count, context_count)
        return {
            "mode": "binary_ratio_prune_without_semantic_labels",
            "source_gaussian_count": vertex_count,
            "subject_gaussian_count": subject_count,
            "context_gaussian_count": context_count,
            "pruned_gaussian_count": max(0, vertex_count - subject_count - context_count),
            "notes": ["Pruning is ratio-based until semantic Gaussian labels are produced by a configured foreground segmenter."],
        }
    if info["format"] == "ascii":
        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
        end_index = next((idx for idx, line in enumerate(lines) if line.strip() == "end_header"), -1)
        vertices = lines[end_index + 1 : end_index + 1 + vertex_count] if end_index >= 0 else []
        _write_ascii_vertices(subject_path, header_text, vertices[:subject_count])
        context_stride = max(1, int(max(1, len(vertices) - subject_count) / max(1, context_count))) if context_count else 1
        _write_ascii_vertices(context_path, header_text, vertices[subject_count::context_stride][:context_count])
        return {
            "mode": "ascii_ratio_prune_without_semantic_labels",
            "source_gaussian_count": vertex_count,
            "subject_gaussian_count": min(subject_count, len(vertices)),
            "context_gaussian_count": min(context_count, max(0, len(vertices) - subject_count)),
            "pruned_gaussian_count": max(0, vertex_count - subject_count - context_count),
            "notes": ["Pruning is ratio-based until semantic Gaussian labels are produced by a configured foreground segmenter."],
        }
    return {"mode": "unsupported_identity_fallback", "source_gaussian_count": vertex_count, "notes": [f"Unsupported PLY format: {info['format']}"]}


def _try_write_sampled_ply(source: Path, target: Path, *, max_vertices: int, max_size_mb: int) -> dict[str, Any]:
    max_vertices = max(1, int(max_vertices))
    with source.open("rb") as fh:
        prefix = fh.read(512 * 1024)
    header_end = prefix.find(b"end_header\n")
    newline_len = len(b"end_header\n")
    if header_end < 0:
        header_end = prefix.find(b"end_header\r\n")
        newline_len = len(b"end_header\r\n")
    if header_end < 0:
        _write_empty_ascii_ply(target)
        return {"mode": "viewer_sampling_failed_empty_fallback", "viewer_gaussian_count": 0, "notes": ["viewer_model: PLY header end not found."]}
    header_bytes = prefix[: header_end + newline_len]
    try:
        header_text = header_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        _write_empty_ascii_ply(target)
        return {"mode": "viewer_sampling_failed_empty_fallback", "viewer_gaussian_count": 0, "notes": ["viewer_model: PLY header is not ASCII-decodable."]}

    info = _parse_ply_header(header_text)
    vertex_count = int(info.get("vertex_count") or 0)
    if vertex_count <= 0:
        _write_empty_ascii_ply(target)
        return {"mode": "viewer_empty_source", "viewer_gaussian_count": 0, "sampling_step": 1}
    max_size_bytes = max(1, int(max_size_mb)) * 1024 * 1024
    if info["format"] == "binary_little_endian" and int(info.get("vertex_row_size") or 0) > 0:
        row_size = int(info["vertex_row_size"])
        max_vertices = min(max_vertices, max(1, (max_size_bytes - len(header_bytes)) // row_size))
    elif info["format"] == "ascii" and source.stat().st_size > len(header_bytes):
        avg_row_size = max(1, (source.stat().st_size - len(header_bytes)) // max(1, vertex_count))
        max_vertices = min(max_vertices, max(1, (max_size_bytes - len(header_bytes)) // avg_row_size))
    if vertex_count <= max_vertices:
        _link_or_copy(source, target)
        return {"mode": "viewer_identity_within_budget", "viewer_gaussian_count": vertex_count, "sampling_step": 1}

    step = max(1, math.ceil(vertex_count / max_vertices))
    sampled_count = math.ceil(vertex_count / step)
    if info["format"] == "binary_little_endian":
        row_size = int(info.get("vertex_row_size") or 0)
        if row_size <= 0:
            _write_empty_ascii_ply(target)
            return {"mode": "viewer_sampling_failed_empty_fallback", "viewer_gaussian_count": 0, "sampling_step": step, "notes": ["viewer_model: unsupported binary PLY property layout."]}
        _write_binary_vertex_sample(source, target, header_text, len(header_bytes), row_size, vertex_count, step, sampled_count)
        return {"mode": "viewer_even_sample_binary", "viewer_gaussian_count": sampled_count, "sampling_step": step}
    if info["format"] == "ascii":
        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
        end_index = next((idx for idx, line in enumerate(lines) if line.strip() == "end_header"), -1)
        vertices = lines[end_index + 1 : end_index + 1 + vertex_count] if end_index >= 0 else []
        sampled = vertices[::step][:sampled_count]
        _write_ascii_vertices(target, header_text, sampled)
        return {"mode": "viewer_even_sample_ascii", "viewer_gaussian_count": len(sampled), "sampling_step": step}

    _write_empty_ascii_ply(target)
    return {"mode": "viewer_sampling_failed_empty_fallback", "viewer_gaussian_count": 0, "sampling_step": step, "notes": [f"viewer_model: unsupported PLY format {info['format']}."]}


def _parse_ply_header(header_text: str) -> dict[str, Any]:
    fmt = "unknown"
    vertex_count = 0
    in_vertex = False
    property_types: list[str] = []
    property_names: list[str] = []
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex "):
            vertex_count = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element ") and not line.startswith("element vertex "):
            in_vertex = False
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3 and parts[1] != "list":
                property_types.append(parts[1])
                property_names.append(parts[2])
    row_size = sum(_ply_type_size(prop_type) for prop_type in property_types)
    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_row_size": row_size,
        "property_types": property_types,
        "property_names": property_names,
        "property_layout": list(zip(property_types, property_names)),
    }


def _binary_property_layout(property_layout: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    offset = 0
    for prop_type, name in property_layout:
        size = _ply_type_size(prop_type)
        if size <= 0:
            return {}
        layout[name] = {"type": prop_type, "offset": offset, "size": size}
        offset += size
    return layout


def _scan_scale_cleanup_stats(source: Path, data_offset: int, row_size: int, vertex_count: int, layout: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scale_radii: list[float] = []
    opacities: list[float] = []
    scale_sum = 0.0
    scale_count = 0
    scale_max = 0.0
    with source.open("rb") as handle:
        handle.seek(data_offset)
        for _ in range(vertex_count):
            row = handle.read(row_size)
            if len(row) != row_size:
                break
            values = _row_scale_opacity(row, layout)
            if values is None:
                continue
            radius, opacity = values
            if math.isfinite(radius):
                scale_radii.append(radius)
                scale_sum += radius
                scale_count += 1
                scale_max = max(scale_max, radius)
            if math.isfinite(opacity):
                opacities.append(opacity)
    return {
        "scale_radii": scale_radii,
        "opacities": opacities,
        "scale_mean": scale_sum / scale_count if scale_count else None,
        "scale_max": scale_max,
    }


def _row_scale_opacity(row: bytes | bytearray, layout: dict[str, dict[str, Any]]) -> tuple[float, float] | None:
    try:
        scale_logs = [struct.unpack_from("<f", row, int(layout[f"scale_{index}"]["offset"]))[0] for index in range(3)]
        opacity = struct.unpack_from("<f", row, int(layout["opacity"]["offset"]))[0]
    except (KeyError, struct.error):
        return None
    if not all(math.isfinite(value) for value in scale_logs):
        return None
    max_log = max(scale_logs)
    radius = math.exp(max_log) if max_log < 80 else math.inf
    return radius, opacity


def _scale_cleanup_threshold(scale_p50: float | None, scale_p95: float | None, scale_p99: float | None, cleanup_config: dict[str, Any]) -> float | None:
    finite = [value for value in (scale_p50, scale_p95, scale_p99) if value is not None and math.isfinite(value) and value > 0]
    if not finite:
        return None
    p50 = scale_p50 if scale_p50 is not None and math.isfinite(scale_p50) and scale_p50 > 0 else min(finite)
    p95 = scale_p95 if scale_p95 is not None and math.isfinite(scale_p95) and scale_p95 > 0 else p50
    candidates = [
        scale_p99 if scale_p99 is not None and math.isfinite(scale_p99) and scale_p99 > 0 else None,
        p95 * float(cleanup_config.get("p95_multiplier") or 2.0),
        p50 * float(cleanup_config.get("median_multiplier") or 80.0),
    ]
    candidate_values = [value for value in candidates if value is not None and math.isfinite(value) and value > 0]
    if not candidate_values:
        return None
    return max(p50, min(candidate_values))


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


def _ply_type_size(prop_type: str) -> int:
    return {
        "char": 1,
        "uchar": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "ushort": 2,
        "int16": 2,
        "uint16": 2,
        "int": 4,
        "uint": 4,
        "int32": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }.get(prop_type, 0)


def _header_with_vertex_count(header_text: str, count: int) -> bytes:
    lines = []
    replaced = False
    for line in header_text.splitlines():
        if line.startswith("element vertex "):
            lines.append(f"element vertex {count}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(2, f"element vertex {count}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _write_binary_vertex_slice(source: Path, target: Path, header_text: str, data_offset: int, row_size: int, count: int, header_count: int) -> None:
    target.write_bytes(_header_with_vertex_count(header_text, header_count))
    if count <= 0:
        return
    bytes_to_copy = count * row_size
    with source.open("rb") as src, target.open("ab") as dst:
        src.seek(data_offset)
        remaining = bytes_to_copy
        while remaining > 0:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)


def _write_binary_vertex_sample(source: Path, target: Path, header_text: str, data_offset: int, row_size: int, vertex_count: int, step: int, sampled_count: int) -> None:
    target.write_bytes(_header_with_vertex_count(header_text, sampled_count))
    with source.open("rb") as src, target.open("ab") as dst:
        src.seek(data_offset)
        index = 0
        written = 0
        while index < vertex_count and written < sampled_count:
            row = src.read(row_size)
            if len(row) != row_size:
                break
            if index % step == 0:
                dst.write(row)
                written += 1
            index += 1
    if written != sampled_count:
        target_bytes = target.read_bytes()
        old_header = _header_with_vertex_count(header_text, sampled_count)
        new_header = _header_with_vertex_count(header_text, written)
        target.write_bytes(new_header + target_bytes[len(old_header) :])


def _write_ascii_vertices(target: Path, header_text: str, vertices: list[str]) -> None:
    target.write_text(_header_with_vertex_count(header_text, len(vertices)).decode("ascii") + "\n".join(vertices) + ("\n" if vertices else ""), encoding="utf-8")


def _write_empty_ascii_ply(path: Path) -> None:
    path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n", encoding="utf-8")


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _size_payload(path: Path) -> dict[str, Any]:
    size = path.stat().st_size if path.exists() else 0
    return {"path": str(path), "size_bytes": size, "size_mb": round(size / 1024 / 1024, 3)}


def _write_rect_mask_png(path: Path, width: int, height: int, x0: int, y0: int, rect_w: int, rect_h: int) -> None:
    rows = []
    for y in range(height):
        row = bytearray(width)
        for x in range(width):
            row[x] = 255 if x0 <= x < x0 + rect_w and y0 <= y < y0 + rect_h else 0
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)


def _safe_ratio(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    return round(min(0.98, max(0.02, parsed)), 4)


def _format_template_part(value: str, values: dict[str, str]) -> str:
    return value.format(**values)
