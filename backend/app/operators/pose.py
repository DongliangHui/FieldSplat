from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.fieldsplat_defaults import default_at
from app.models import Workflow
from app.operators.base import CommandResult
from app.operators.colmap import ColmapGlobalSkeletonOperator, ColmapRunResult, evaluate_colmap_quality
from app.operators.preprocess import PreprocessRunResult
from app.services.resource_locks import resource_lock
from app.services.stage_cache import StageCache, cache_hit_command


@dataclass
class PoseAttemptsRunResult:
    selected: ColmapRunResult | None
    attempts_report_path: Path
    attempts: list[dict[str, Any]]
    selected_attempt_key: str | None
    selected_route_key: str | None
    commands: list[CommandResult]
    passed: bool
    reason: str | None


@dataclass
class Mast3rSfmRunResult:
    workspace_dir: Path
    dataset_dir: Path
    final_export_dir: Path
    debug_artifacts_dir: Path
    cache_dir: Path
    final_export_archive_path: Path | None
    debug_archive_path: Path | None
    camera_trajectory_path: Path
    sparse_point_cloud_path: Path
    registration_report_path: Path
    transforms_path: Path
    metadata_path: Path
    commands: list[CommandResult]
    quality: dict[str, Any]
    report_path: Path
    passed: bool
    reason: str | None


class ColmapAttemptsOperator:
    name = "pose.colmap_attempts"
    queue = "colmap"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(
        self,
        workflow: Workflow,
        preprocess: PreprocessRunResult,
        subject_mask: dict[str, Any] | None = None,
        local_feature_matching: dict[str, Any] | None = None,
    ) -> PoseAttemptsRunResult:
        mode = _workflow_mode(workflow)
        workspace_root = Path(self.settings.workspace_root) / "runs" / workflow.id / "pose_colmap_attempts"
        workspace_suffix = str((preprocess.media_metadata or {}).get("workspace_suffix") or "").strip()
        workspace_dir = workspace_root / _safe_workspace_suffix(workspace_suffix) if workspace_suffix else workspace_root
        workspace_dir.mkdir(parents=True, exist_ok=True)
        workspace_name_prefix = "pose_colmap_attempts"
        if workspace_suffix:
            workspace_name_prefix = f"{workspace_name_prefix}/{_safe_workspace_suffix(workspace_suffix).as_posix()}"
        attempts: list[dict[str, Any]] = []
        commands: list[CommandResult] = []
        selected: ColmapRunResult | None = None
        selected_score = -1.0
        selected_attempt_key: str | None = None
        stop_policy = _attempt_stop_policy(workflow, self.settings)

        for index, spec in enumerate(_attempt_specs(workflow, preprocess, self.settings, local_feature_matching=local_feature_matching), start=1):
            attempt_name = str(spec.get("name") or f"{spec['matcher']}_{spec['camera_model'].lower()}")
            attempt_key = f"attempt_{index:03d}_{attempt_name}"
            if spec["matcher"] == "vocabtree" and not _vocabtree_available(workflow):
                attempts.append(
                    {
                        "attempt_key": attempt_key,
                        **spec,
                        "status": "skipped",
                        "passed": False,
                        "reason": "vocab_tree_path_not_configured",
                    }
                )
                continue
            try:
                result = ColmapGlobalSkeletonOperator(self.settings).run(
                    workflow,
                    preprocess,
                    attempt_key=attempt_key,
                    matcher=spec["matcher"],
                    camera_model=spec["camera_model"],
                    attempt_spec=spec,
                    workspace_name=f"{workspace_name_prefix}/{attempt_key}",
                    subject_mask=subject_mask,
                )
                commands.extend(result.commands)
                quality = evaluate_colmap_quality(result.quality, mode=mode)
                score = _score_attempt(quality)
                attempt_report = {
                    "attempt_key": attempt_key,
                    **spec,
                    "status": "succeeded" if quality.get("passed") else "blocked",
                    "passed": bool(quality.get("passed")),
                    "score": score,
                    "registration_report": result.quality,
                    "quality": quality,
                }
                attempts.append(attempt_report)
                if quality.get("passed") and score > selected_score:
                    selected = result
                    selected_score = score
                    selected_attempt_key = attempt_key
                if selected is result and _attempt_stop_policy_accepts(attempt_report, stop_policy):
                    attempt_report["early_stop_selected"] = True
                    attempt_report["early_stop_policy"] = stop_policy
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "attempt_key": attempt_key,
                        **spec,
                        "status": "failed",
                        "passed": False,
                        "reason": str(exc),
                    }
                )

        if selected is None:
            passed_attempts = [attempt for attempt in attempts if attempt.get("registration_report")]
            if passed_attempts:
                best_attempt = max(passed_attempts, key=lambda item: float(item.get("score") or 0.0))
                selected_attempt_key = str(best_attempt["attempt_key"])
            reason = "no_colmap_attempt_passed_pose_gate"
        else:
            reason = None

        report = {
            "workflow_id": workflow.id,
            "operator": self.name,
            "mode": mode,
            "input_image_count": len(preprocess.image_paths),
            "subject_mask": _pose_mask_summary(subject_mask),
            "attempt_stop_policy": stop_policy,
            "selected_attempt_key": selected_attempt_key,
            "passed": selected is not None,
            "reason": reason,
            "attempts": attempts,
        }
        attempts_report_path = workspace_dir / "pose_attempts_report.json"
        attempts_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return PoseAttemptsRunResult(
            selected=selected,
            attempts_report_path=attempts_report_path,
            attempts=attempts,
            selected_attempt_key=selected_attempt_key,
            selected_route_key="colmap_splatfacto" if selected else None,
            commands=commands,
            passed=selected is not None,
            reason=reason,
        )


def _safe_workspace_suffix(value: str) -> Path:
    parts: list[str] = []
    for raw_part in str(value).replace("\\", "/").split("/"):
        part = raw_part.strip()
        if not part or part in {".", ".."}:
            continue
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe:
            parts.append(safe[:120])
    return Path(*parts) if parts else Path()


class Mast3rSfmFallbackOperator:
    name = "pose.mast3r_sfm_fallback"
    queue = "gpu"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def run(self, workflow: Workflow, preprocess: PreprocessRunResult, reason: str) -> Mast3rSfmRunResult:
        workspace_dir = Path(self.settings.workspace_root) / "runs" / workflow.id / "mast3r_sfm_fallback"
        final_export_dir = workspace_dir / "01_final_export"
        debug_artifacts_dir = workspace_dir / "02_debug_artifacts"
        cache_dir = workspace_dir / "03_cache"
        raw_output_dir = cache_dir / "mast3r_raw_output"
        for path in (final_export_dir, debug_artifacts_dir / "logs", debug_artifacts_dir / "registration_report", raw_output_dir):
            path.mkdir(parents=True, exist_ok=True)
        commands: list[CommandResult] = []
        operator_config = self.settings.engine_config.get("operators", {}).get("mast3r_sfm", {}) or {}
        cache = StageCache(self.settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[*preprocess.image_paths],
            stage_config={"operator_config": operator_config, "trigger_reason": reason, "image_count": len(preprocess.image_paths)},
            algorithm_version="mast3r-sfm-fallback-v2",
        )
        available, unavailable_reason = _mast3r_sfm_available(operator_config)
        report_path = debug_artifacts_dir / "mast3r_sfm_fallback_report.json"
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir) and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            passed = bool(report.get("passed"))
            quality = report.get("quality") or {}
            return Mast3rSfmRunResult(
                workspace_dir=workspace_dir,
                dataset_dir=preprocess.dataset_dir,
                final_export_dir=final_export_dir,
                debug_artifacts_dir=debug_artifacts_dir,
                cache_dir=cache_dir,
                final_export_archive_path=workspace_dir / "mast3r_sfm_final_export.zip" if (workspace_dir / "mast3r_sfm_final_export.zip").exists() else None,
                debug_archive_path=workspace_dir / "mast3r_sfm_debug_artifacts.zip" if (workspace_dir / "mast3r_sfm_debug_artifacts.zip").exists() else None,
                camera_trajectory_path=final_export_dir / "cameras.json",
                sparse_point_cloud_path=final_export_dir / "sparse_point_cloud.ply",
                registration_report_path=debug_artifacts_dir / "registration_report" / "registration_report.json",
                transforms_path=final_export_dir / "transforms.json",
                metadata_path=final_export_dir / "metadata.json",
                commands=[cache_hit_command(self.name, "pose_mast3r_sfm_fallback", cache_entry.cache_key, workspace_dir)],
                quality=quality,
                report_path=report_path,
                passed=passed,
                reason=report.get("reason"),
            )
        if not available:
            report = {
                "workflow_id": workflow.id,
                "operator": self.name,
                "status": "unavailable",
                "trigger_reason": reason,
                "image_count": len(preprocess.image_paths),
                "passed": False,
                "reason": unavailable_reason,
                "expected_outputs": _mast3r_expected_outputs(operator_config, preprocess, workspace_dir, raw_output_dir),
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return _empty_mast3r_result(workflow, preprocess, workspace_dir, final_export_dir, debug_artifacts_dir, cache_dir, report_path, report, commands)

        command_template = operator_config.get("command")
        if not command_template:
            report = {
                "workflow_id": workflow.id,
                "operator": self.name,
                "status": "unavailable",
                "trigger_reason": reason,
                "image_count": len(preprocess.image_paths),
                "passed": False,
                "reason": "mast3r_sfm_command_not_configured",
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return _empty_mast3r_result(workflow, preprocess, workspace_dir, final_export_dir, debug_artifacts_dir, cache_dir, report_path, report, commands)

        values = _mast3r_template_values(operator_config, preprocess, workspace_dir, raw_output_dir)
        command = [_format_value(str(part), values) for part in command_template]
        with resource_lock("gpu-heavy", settings=self.settings):
            commands.append(_run_command(self.name, "pose_mast3r_sfm_fallback", command, workspace_dir, cwd=operator_config.get("repo_path")))

        expected = _mast3r_expected_outputs(operator_config, preprocess, workspace_dir, raw_output_dir)
        raw_transforms_path = Path(expected["transforms_path"])
        raw_sparse_point_cloud_path = Path(expected["sparse_point_cloud_path"])
        raw_registration_report_path = Path(expected["registration_report_path"])
        raw_camera_trajectory_path = Path(expected["camera_trajectory_path"])

        transforms_path = final_export_dir / "transforms.json"
        camera_trajectory_path = final_export_dir / "cameras.json"
        sparse_point_cloud_path = final_export_dir / "sparse_point_cloud.ply"
        registration_report_path = debug_artifacts_dir / "registration_report" / "registration_report.json"
        metadata_path = final_export_dir / "metadata.json"

        if raw_transforms_path.exists():
            preprocess.dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_sparse_point_cloud_path = preprocess.dataset_dir / sparse_point_cloud_path.name
            if raw_sparse_point_cloud_path.exists():
                shutil.copy2(raw_sparse_point_cloud_path, sparse_point_cloud_path)
                shutil.copy2(sparse_point_cloud_path, dataset_sparse_point_cloud_path)
            dataset_transforms_path = _copy_mast3r_transforms_to_dataset(raw_transforms_path, preprocess.dataset_dir, dataset_sparse_point_cloud_path)
            _write_final_export_transforms(dataset_transforms_path, transforms_path)
        if raw_camera_trajectory_path.exists():
            shutil.copy2(raw_camera_trajectory_path, camera_trajectory_path)
        elif transforms_path.exists():
            _write_camera_trajectory_from_transforms(workflow, transforms_path, camera_trajectory_path, source=self.name)
        if raw_registration_report_path.exists():
            shutil.copy2(raw_registration_report_path, registration_report_path)
        else:
            _write_mast3r_registration_report(workflow, preprocess, registration_report_path, transforms_path, sparse_point_cloud_path, commands)
        _copy_final_export_images(preprocess, transforms_path, final_export_dir / "images")
        registration_report = json.loads(registration_report_path.read_text(encoding="utf-8")) if registration_report_path.exists() else {}
        quality = evaluate_colmap_quality({**registration_report, "operator": self.name})
        passed = commands[-1].exit_code == 0 and transforms_path.exists() and bool(quality.get("passed"))
        failure_reason = None if passed else _mast3r_failure_reason(commands[-1], transforms_path, quality)
        _write_cache_summary(cache_dir, debug_artifacts_dir / "cache_summary.json")
        _write_mast3r_logs(commands[-1], debug_artifacts_dir / "logs")
        _write_final_export_metadata(metadata_path, workflow, preprocess, reason, quality, passed, failure_reason)
        final_export_archive_path = Path(shutil.make_archive(str(workspace_dir / "mast3r_sfm_final_export"), "zip", final_export_dir)) if passed else None
        debug_archive_path = None if passed else Path(shutil.make_archive(str(workspace_dir / "mast3r_sfm_debug_artifacts"), "zip", debug_artifacts_dir))
        registration_report = json.loads(registration_report_path.read_text(encoding="utf-8")) if registration_report_path.exists() else {}
        report = {
            "workflow_id": workflow.id,
            "operator": self.name,
            "status": "succeeded" if passed else "failed",
            "trigger_reason": reason,
            "image_count": len(preprocess.image_paths),
            "passed": passed,
            "reason": failure_reason,
            "quality": quality,
            "expected_outputs": expected,
            "final_export_dir": str(final_export_dir),
            "debug_artifacts_dir": str(debug_artifacts_dir),
            "cache_dir": str(cache_dir),
            "command": command,
            "exit_code": commands[-1].exit_code,
            "cache_hit": False,
            "cache_key": cache_entry.cache_key,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        cache.save(
            cache_entry,
            workspace_dir,
            metadata=report,
            exclude_names={"03_cache", "images", "mast3r_sfm_final_export.zip", "mast3r_sfm_debug_artifacts.zip"},
        )
        return Mast3rSfmRunResult(
            workspace_dir=workspace_dir,
            dataset_dir=preprocess.dataset_dir,
            final_export_dir=final_export_dir,
            debug_artifacts_dir=debug_artifacts_dir,
            cache_dir=cache_dir,
            final_export_archive_path=final_export_archive_path,
            debug_archive_path=debug_archive_path,
            camera_trajectory_path=camera_trajectory_path,
            sparse_point_cloud_path=sparse_point_cloud_path,
            registration_report_path=registration_report_path,
            transforms_path=transforms_path,
            metadata_path=metadata_path,
            commands=commands,
            quality=quality,
            report_path=report_path,
            passed=passed,
            reason=failure_reason,
        )


def _attempt_specs(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    settings: Settings | None = None,
    *,
    local_feature_matching: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = workflow.config_json or {}
    configured = config.get("colmap_attempts")
    if isinstance(configured, list) and configured:
        return _with_learned_feature_attempt(
            workflow,
            preprocess,
            [
                {
                    "name": str(item.get("name") or f"{item.get('matcher') or 'exhaustive'}_{item.get('camera_model') or 'SIMPLE_RADIAL'}").lower(),
                    "matcher": str(item.get("matcher") or "exhaustive"),
                    "camera_model": str(item.get("camera_model") or "SIMPLE_RADIAL"),
                    **{key: value for key, value in item.items() if key not in {"name", "matcher", "camera_model"}},
                }
                for item in configured
                if isinstance(item, dict)
            ],
            local_feature_matching,
            settings,
        )
    baseline_attempts = default_at("colmap_attempts.attempts", settings=settings)
    if isinstance(baseline_attempts, list) and baseline_attempts:
        input_mode = (preprocess.media_metadata or {}).get("input_mode")
        selected: list[dict[str, Any]] = []
        for item in baseline_attempts:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if input_mode == "video" and name.startswith("photo_"):
                continue
            if input_mode != "video" and name.startswith("video_"):
                continue
            selected.append(
                {
                    "name": name or f"{item.get('matcher') or 'exhaustive'}_{item.get('camera_model') or 'SIMPLE_RADIAL'}".lower(),
                    "matcher": str(item.get("matcher") or "exhaustive"),
                    "camera_model": str(item.get("camera_model") or "SIMPLE_RADIAL"),
                    **{key: value for key, value in item.items() if key not in {"name", "matcher", "camera_model"}},
                }
            )
        if selected:
            if input_mode != "video":
                return _with_learned_feature_attempt(workflow, preprocess, _photo_attempt_policy(workflow, preprocess, selected), local_feature_matching, settings)
            return _with_learned_feature_attempt(workflow, preprocess, selected, local_feature_matching, settings)
    input_mode = (preprocess.media_metadata or {}).get("input_mode")
    if input_mode == "video":
        return _with_learned_feature_attempt(
            workflow,
            preprocess,
            [
                {"name": "video_sequential_opencv", "matcher": "sequential", "camera_model": "OPENCV", "single_camera": True, "sequential_overlap": 20},
                {"name": "video_sequential_simple_radial", "matcher": "sequential", "camera_model": "SIMPLE_RADIAL", "single_camera": True, "sequential_overlap": 30},
                {"name": "mixed_vocabtree_opencv", "matcher": "vocabtree", "camera_model": "OPENCV", "single_camera": False},
            ],
            local_feature_matching,
            settings,
        )
    return _with_learned_feature_attempt(
        workflow,
        preprocess,
        _photo_attempt_policy(
            workflow,
            preprocess,
            [
                {"name": "photo_vocabtree_opencv", "matcher": "vocabtree", "camera_model": "OPENCV", "single_camera": False},
                {"name": "photo_exhaustive_opencv", "matcher": "exhaustive", "camera_model": "OPENCV", "single_camera": False},
            ],
        ),
        local_feature_matching,
        settings,
    )


def _with_learned_feature_attempt(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    attempts: list[dict[str, Any]],
    local_feature_matching: dict[str, Any] | None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    config = workflow.config_json or {}
    if config.get("learned_colmap_import") is False or any(str(item.get("matcher")) == "imported" for item in attempts):
        return attempts
    if not local_feature_matching or not local_feature_matching.get("passed"):
        return attempts
    colmap_import = local_feature_matching.get("colmap_import") or {}
    if not colmap_import.get("import_ready"):
        return attempts
    features_dir = colmap_import.get("features_dir")
    match_list_path = colmap_import.get("match_list_path")
    if not features_dir or not match_list_path:
        return attempts
    learned_attempt = {
        "name": "photo_lightglue_aliked_opencv",
        "matcher": "imported",
        "camera_model": "OPENCV",
        "single_camera": False,
        "feature_source": str(local_feature_matching.get("method") or "lightglue_aliked"),
        "colmap_features_dir": str(features_dir),
        "colmap_match_list_path": str(match_list_path),
        "match_type": str(colmap_import.get("match_type") or "raw"),
        "mapper_min_model_size": int(config.get("learned_colmap_min_model_size") or 3),
        "mapper_ba_refine_extra_params": True,
    }
    learned_attempts = [learned_attempt]
    operator_config = (((settings or get_settings()).engine_config.get("operators", {}) or {}).get("colmap", {}) or {}).get("local_feature_matching", {}) or {}
    strong_match_min = int(config.get("learned_colmap_strong_match_min") or operator_config.get("strong_match_min") or 0)
    base_min = int(colmap_import.get("min_matches_per_pair") or operator_config.get("min_matches") or 0)
    if strong_match_min > max(base_min, 0):
        filtered_path, filtered_count = _filtered_match_list(Path(str(match_list_path)), strong_match_min)
        if filtered_path and filtered_count > 0:
            learned_attempts.insert(
                0,
                {
                    **learned_attempt,
                    "name": f"photo_lightglue_aliked_min{strong_match_min}_opencv",
                    "colmap_match_list_path": str(filtered_path),
                    "learned_match_filter_min_matches": strong_match_min,
                    "learned_match_filter_pair_count": filtered_count,
                },
            )
    input_mode = (preprocess.media_metadata or {}).get("input_mode")
    if input_mode == "video" and attempts:
        return [attempts[0], *learned_attempts, *attempts[1:]]
    return [*learned_attempts, *attempts]


def _filtered_match_list(match_list_path: Path, min_matches: int) -> tuple[Path | None, int]:
    if not match_list_path.exists():
        return None, 0
    output_path = match_list_path.with_name(f"{match_list_path.stem}_min{int(min_matches)}{match_list_path.suffix}")
    lines = match_list_path.read_text(encoding="utf-8").splitlines()
    kept: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header:
            continue
        rows: list[str] = []
        while index < len(lines) and lines[index].strip():
            rows.append(lines[index])
            index += 1
        if len(rows) >= min_matches:
            kept.append((header, rows))
    with output_path.open("w", encoding="utf-8") as handle:
        for header, rows in kept:
            handle.write(f"{header}\n")
            handle.write("\n".join(rows))
            handle.write("\n\n")
    return output_path, len(kept)


def _photo_attempt_policy(workflow: Workflow, preprocess: PreprocessRunResult, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = workflow.config_json or {}
    image_count = len(getattr(preprocess, "image_paths", []) or [])
    max_exhaustive = int(config.get("colmap_exhaustive_max_images") or 150)
    has_vocab_tree = bool(config.get("vocab_tree_path"))
    has_gps_exif = bool((preprocess.media_metadata or {}).get("has_gps_exif"))
    ordered: list[dict[str, Any]] = []
    if has_gps_exif:
        ordered.append({"name": "photo_spatial_opencv", "matcher": "spatial", "camera_model": "OPENCV", "single_camera": False})
    if has_vocab_tree:
        ordered.extend([spec for spec in selected if spec.get("matcher") == "vocabtree"])
    if image_count <= max_exhaustive:
        ordered.extend([spec for spec in selected if spec.get("matcher") == "exhaustive"])
    ordered.extend([spec for spec in selected if spec.get("matcher") == "sequential"])
    if not ordered:
        ordered.append({"name": "photo_exhaustive_opencv", "matcher": "exhaustive", "camera_model": "OPENCV", "single_camera": False})
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for spec in ordered:
        key = f"{spec.get('matcher')}:{spec.get('camera_model')}:{spec.get('name')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _pose_mask_summary(subject_mask: dict[str, Any] | None) -> dict[str, Any]:
    if not subject_mask:
        return {"available": False}
    colmap_masking = subject_mask.get("colmap_masking") or {}
    return {
        "available": True,
        "method": subject_mask.get("method"),
        "semantic_model_used": subject_mask.get("semantic_model_used"),
        "foreground_ratio": subject_mask.get("foreground_ratio"),
        "apply_to_colmap": bool(colmap_masking.get("apply_to_colmap")),
        "mask_path": colmap_masking.get("mask_path"),
    }


def _legacy_photo_attempts() -> list[dict[str, Any]]:
    return [
        {"name": "photo_exhaustive_opencv", "matcher": "exhaustive", "camera_model": "OPENCV", "single_camera": False},
        {"name": "mixed_vocabtree_opencv", "matcher": "vocabtree", "camera_model": "OPENCV", "single_camera": False},
    ]


def _score_attempt(quality: dict[str, Any]) -> float:
    registration_rate = float(quality.get("registration_rate") or 0.0)
    sparse_points = min(float(quality.get("sparse_point_count") or 0.0) / 100000.0, 1.0)
    reprojection_error = quality.get("mean_reprojection_error")
    reprojection_score = 1.0 if reprojection_error is None else max(0.0, 1.0 - min(float(reprojection_error), 10.0) / 10.0)
    continuity = 1.0 if (quality.get("trajectory_continuity") or {}).get("passed", True) else 0.0
    return registration_rate * 0.55 + sparse_points * 0.15 + reprojection_score * 0.2 + continuity * 0.1


def _attempt_stop_policy(workflow: Workflow, settings: Settings) -> dict[str, Any]:
    config = workflow.config_json or {}
    stage_config = settings.engine_config.get("stage_optimized_reconstruction")
    if not isinstance(stage_config, dict):
        stage_config = {}
    policy_config = stage_config.get("pose_attempts")
    if not isinstance(policy_config, dict):
        policy_config = {}
    enabled_value = config.get("colmap_attempt_stop_after_quality_accept")
    enabled = bool(policy_config.get("stop_after_quality_accept", False)) if enabled_value is None else bool(enabled_value)
    if bool(config.get("benchmark_mode") or config.get("explore_all_pose_attempts")):
        enabled = False
    return {
        "enabled": enabled,
        "accept_registered_ratio": float(config.get("colmap_attempt_accept_registered_ratio") or policy_config.get("accept_registered_ratio") or 0.99),
        "accept_max_reprojection_error_px": float(config.get("colmap_attempt_accept_max_reprojection_error_px") or policy_config.get("accept_max_reprojection_error_px") or 2.0),
        "accept_min_sparse_points": int(config.get("colmap_attempt_accept_min_sparse_points") or policy_config.get("accept_min_sparse_points") or 10000),
    }


def _attempt_stop_policy_accepts(attempt_report: dict[str, Any], policy: dict[str, Any]) -> bool:
    if not bool(policy.get("enabled")) or not bool(attempt_report.get("passed")):
        return False
    report = attempt_report.get("registration_report") or {}
    registered = float(report.get("registered_camera_count") or 0.0)
    total = float(report.get("input_image_count") or 0.0)
    registration_rate = float(report.get("registration_rate") or (registered / total if total > 0 else 0.0))
    reprojection_error = report.get("mean_reprojection_error")
    sparse_points = int(report.get("sparse_point_count") or 0)
    if registration_rate < float(policy.get("accept_registered_ratio") or 0.99):
        return False
    if reprojection_error is not None and float(reprojection_error) > float(policy.get("accept_max_reprojection_error_px") or 2.0):
        return False
    return sparse_points >= int(policy.get("accept_min_sparse_points") or 10000)


def _vocabtree_available(workflow: Workflow) -> bool:
    config = workflow.config_json or {}
    return bool(config.get("vocab_tree_path"))


def _workflow_mode(workflow: Workflow) -> str:
    config = workflow.config_json or {}
    mode = config.get("mode") or config.get("profile") or get_settings().workflow_default_mode
    if mode == "smoke":
        return "quick_preview"
    return str(mode if mode in {"quick_preview", "standard", "high_quality"} else "standard")


def _mast3r_sfm_available(operator_config: dict[str, Any]) -> tuple[bool, str | None]:
    if not operator_config.get("enabled", False):
        return False, "mast3r_sfm_disabled"
    repo_path = Path(str(operator_config.get("repo_path") or ""))
    if not repo_path.exists():
        return False, "mast3r_sfm_repo_missing"
    code_root = operator_config.get("code_root")
    if code_root and not Path(str(code_root)).exists():
        return False, "mast3r_sfm_code_root_missing"
    python_path = str(operator_config.get("python") or "python3")
    if not _binary_available(python_path):
        return False, "mast3r_sfm_python_missing"
    checkpoint = operator_config.get("checkpoint")
    if checkpoint and not Path(str(checkpoint)).exists():
        return False, "mast3r_sfm_checkpoint_missing"
    wrapper = operator_config.get("wrapper")
    if wrapper and not Path(str(wrapper)).exists():
        return False, "mast3r_sfm_wrapper_missing"
    return True, None


def _mast3r_template_values(operator_config: dict[str, Any], preprocess: PreprocessRunResult, workspace_dir: Path, output_dir: Path) -> dict[str, str]:
    return {
        "python": str(operator_config.get("python") or "python3"),
        "repo_path": str(operator_config.get("repo_path") or ""),
        "code_root": str(operator_config.get("code_root") or operator_config.get("repo_path") or ""),
        "wrapper": str(operator_config.get("wrapper") or ""),
        "checkpoint": str(operator_config.get("checkpoint") or ""),
        "images_dir": str(preprocess.images_dir),
        "dataset_dir": str(preprocess.dataset_dir),
        "workspace_dir": str(workspace_dir),
        "output_dir": str(output_dir),
        "final_export_dir": str(workspace_dir / "01_final_export"),
        "debug_artifacts_dir": str(workspace_dir / "02_debug_artifacts"),
        "cache_dir": str(workspace_dir / "03_cache"),
        "transforms_path": str(output_dir / "transforms.json"),
        "camera_trajectory_path": str(output_dir / "camera_trajectory.json"),
        "sparse_point_cloud_path": str(output_dir / "sparse_point_cloud.ply"),
        "registration_report_path": str(output_dir / "registration_report.json"),
    }


def _mast3r_expected_outputs(operator_config: dict[str, Any], preprocess: PreprocessRunResult, workspace_dir: Path, output_dir: Path) -> dict[str, str]:
    values = _mast3r_template_values(operator_config, preprocess, workspace_dir, output_dir)
    return {
        "transforms_path": _format_value(str(operator_config.get("transforms_path") or values["transforms_path"]), values),
        "camera_trajectory_path": _format_value(str(operator_config.get("camera_trajectory_path") or values["camera_trajectory_path"]), values),
        "sparse_point_cloud_path": _format_value(str(operator_config.get("sparse_point_cloud_path") or values["sparse_point_cloud_path"]), values),
        "registration_report_path": _format_value(str(operator_config.get("registration_report_path") or values["registration_report_path"]), values),
    }


def _empty_mast3r_result(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    workspace_dir: Path,
    final_export_dir: Path,
    debug_artifacts_dir: Path,
    cache_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    commands: list[CommandResult],
) -> Mast3rSfmRunResult:
    transforms_path = final_export_dir / "transforms.json"
    camera_trajectory_path = final_export_dir / "cameras.json"
    sparse_point_cloud_path = final_export_dir / "sparse_point_cloud.ply"
    registration_report_path = debug_artifacts_dir / "registration_report" / "registration_report.json"
    metadata_path = final_export_dir / "metadata.json"
    if not registration_report_path.exists():
        registration_report_path.parent.mkdir(parents=True, exist_ok=True)
        registration_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_cache_summary(cache_dir, debug_artifacts_dir / "cache_summary.json")
    _write_final_export_metadata(metadata_path, workflow, preprocess, str(report.get("trigger_reason") or ""), {"passed": False, **report}, False, str(report.get("reason") or "mast3r_sfm_unavailable"))
    debug_archive_path = Path(shutil.make_archive(str(workspace_dir / "mast3r_sfm_debug_artifacts"), "zip", debug_artifacts_dir))
    quality = {"passed": False, "hard_fail": True, "issues": [str(report.get("reason") or "mast3r_sfm_unavailable")], **report}
    return Mast3rSfmRunResult(
        workspace_dir=workspace_dir,
        dataset_dir=preprocess.dataset_dir,
        final_export_dir=final_export_dir,
        debug_artifacts_dir=debug_artifacts_dir,
        cache_dir=cache_dir,
        final_export_archive_path=None,
        debug_archive_path=debug_archive_path,
        camera_trajectory_path=camera_trajectory_path,
        sparse_point_cloud_path=sparse_point_cloud_path,
        registration_report_path=registration_report_path,
        transforms_path=transforms_path,
        metadata_path=metadata_path,
        commands=commands,
        quality=quality,
        report_path=report_path,
        passed=False,
        reason=str(report.get("reason") or "mast3r_sfm_unavailable"),
    )


def _write_final_export_transforms(source_path: Path, target_path: Path) -> None:
    transforms = json.loads(source_path.read_text(encoding="utf-8"))
    for frame in transforms.get("frames") or []:
        file_path = str(frame.get("file_path") or "")
        if file_path:
            frame["file_path"] = f"images/{Path(file_path).name}"
    transforms["ply_file_path"] = "sparse_point_cloud.ply"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_final_export_images(preprocess: PreprocessRunResult, transforms_path: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    image_names: set[str] = set()
    if transforms_path.exists():
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
        image_names = {Path(str(frame.get("file_path") or "")).name for frame in transforms.get("frames") or []}
        image_names.discard("")
    for source_path in preprocess.image_paths:
        source = Path(source_path)
        if not source.exists():
            continue
        if image_names and source.name not in image_names:
            continue
        target = images_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)


def _write_mast3r_logs(command: CommandResult, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "command.json").write_text(
        json.dumps(
            {
                "operator_name": command.operator_name,
                "stage_key": command.stage_key,
                "command": command.command,
                "cwd": command.cwd,
                "exit_code": command.exit_code,
                "started_at": command.started_at.isoformat(),
                "finished_at": command.finished_at.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (logs_dir / "stdout.txt").write_text(command.stdout or "", encoding="utf-8")
    (logs_dir / "stderr.txt").write_text(command.stderr or "", encoding="utf-8")


def _write_cache_summary(cache_dir: Path, output_path: Path) -> None:
    total_bytes = 0
    file_count = 0
    top_level: dict[str, dict[str, int]] = {}
    if cache_dir.exists():
        for path in cache_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            total_bytes += size
            file_count += 1
            try:
                root_name = path.relative_to(cache_dir).parts[0]
            except (ValueError, IndexError):
                root_name = "."
            entry = top_level.setdefault(root_name, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += size
    payload = {
        "cache_dir": str(cache_dir),
        "policy": "local_only_not_uploaded",
        "file_count": file_count,
        "size_bytes": total_bytes,
        "size_mb": round(total_bytes / 1024 / 1024, 3),
        "top_level": top_level,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_final_export_metadata(
    metadata_path: Path,
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    trigger_reason: str,
    quality: dict[str, Any],
    passed: bool,
    failure_reason: str | None,
) -> None:
    payload = {
        "workflow_id": workflow.id,
        "operator": "pose.mast3r_sfm_fallback",
        "schema": "fieldsplat.mast3r_final_export.v1",
        "trigger_reason": trigger_reason,
        "passed": passed,
        "failure_reason": failure_reason,
        "image_count": len(preprocess.image_paths),
        "quality": quality,
        "files": {
            "transforms": "transforms.json",
            "cameras": "cameras.json",
            "sparse_point_cloud": "sparse_point_cloud.ply",
            "images": "images/",
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_mast3r_transforms_to_dataset(source_path: Path, dataset_dir: Path, sparse_point_cloud_path: Path) -> Path:
    transforms = json.loads(source_path.read_text(encoding="utf-8"))
    if sparse_point_cloud_path.exists():
        transforms["ply_file_path"] = sparse_point_cloud_path.name
    target_path = dataset_dir / "transforms.json"
    target_path.write_text(json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path


def _write_camera_trajectory_from_transforms(workflow: Workflow, transforms_path: Path, output_path: Path, *, source: str) -> None:
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames") or []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cameras = []
    for frame in frames:
        matrix = frame.get("transform_matrix") or []
        center = [matrix[index][3] for index in range(3)] if len(matrix) >= 3 and all(len(row) >= 4 for row in matrix[:3]) else [0.0, 0.0, 0.0]
        cameras.append({"image_name": Path(str(frame.get("file_path") or "")).name, "camera_center": center, "transform_matrix": matrix})
    output_path.write_text(json.dumps({"workflow_id": workflow.id, "source": source, "camera_count": len(cameras), "cameras": cameras}, indent=2), encoding="utf-8")


def _write_mast3r_registration_report(
    workflow: Workflow,
    preprocess: PreprocessRunResult,
    registration_report_path: Path,
    transforms_path: Path,
    sparse_point_cloud_path: Path,
    commands: list[CommandResult],
) -> None:
    frames = []
    if transforms_path.exists():
        frames = (json.loads(transforms_path.read_text(encoding="utf-8")).get("frames") or [])
    report = {
        "workflow_id": workflow.id,
        "operator": "pose.mast3r_sfm_fallback",
        "input_image_count": len(preprocess.image_paths),
        "registered_camera_count": len(frames),
        "registration_rate": len(frames) / max(len(preprocess.image_paths), 1),
        "mean_reprojection_error": None,
        "sparse_point_count": _ply_vertex_count(sparse_point_cloud_path),
        "trajectory_continuity": {"passed": len(frames) > 1, "source": "mast3r_transforms"},
        "commands_succeeded": all(command.exit_code == 0 for command in commands),
    }
    registration_report_path.parent.mkdir(parents=True, exist_ok=True)
    registration_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _ply_vertex_count(path: Path) -> int:
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:50]:
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except ValueError:
                    return 0
    return 0


def _mast3r_failure_reason(command: CommandResult, transforms_path: Path, quality: dict[str, Any]) -> str:
    if command.exit_code != 0:
        return "mast3r_sfm_command_failed"
    if not transforms_path.exists():
        return "mast3r_sfm_transforms_missing"
    issues = quality.get("issues") or []
    if issues:
        return "mast3r_sfm_quality_failed:" + ",".join(str(issue) for issue in issues)
    return "mast3r_sfm_failed"


def _binary_available(value: str) -> bool:
    path = Path(value)
    return path.exists() or shutil.which(value) is not None


def _run_command(operator_name: str, stage_key: str, command: list[str], workspace_dir: Path, *, cwd: str | None = None) -> CommandResult:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=cwd or workspace_dir, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc)
    return CommandResult(operator_name, stage_key, command, str(cwd or workspace_dir), completed.stdout[-4000:] if completed.stdout else "", completed.stderr[-4000:] if completed.stderr else "", completed.returncode, started, finished)


def _format_value(value: str, values: dict[str, str]) -> str:
    return value.format(**values)
