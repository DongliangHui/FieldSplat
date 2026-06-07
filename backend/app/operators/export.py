from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models import Workflow
from app.operators.base import CommandResult
from app.services.stage_cache import StageCache


class ViewerPackageExportOperator:
    name = "export.viewer_package"
    queue = "export"

    def run(self, artifact_paths: list[str], workspace_dir: Path) -> dict[str, Any]:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"artifact_paths": artifact_paths, "viewer_package_ready": bool(artifact_paths)}
        (workspace_dir / "viewer_package_manifest.json").write_text(str(manifest), encoding="utf-8")
        return manifest


class ReconstructionExportPipelineOperator:
    name = "export.pipeline"
    queue = "export"

    def run(
        self,
        workflow: Workflow,
        *,
        splat_path: Path | None,
        route: dict[str, Any],
        quality: dict[str, Any],
        diagnostics: dict[str, Any],
        scope_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        workspace_dir = Path(settings.workspace_root) / "runs" / workflow.id / "export_pipeline"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        cache = StageCache(settings)
        cache_entry = cache.entry(
            self.name,
            inputs=[splat_path or "missing_splat", *((scope_outputs or {}).get("cache_inputs") or [])],
            stage_config={"route": route, "quality": quality, "scope_outputs": _scope_cache_payload(scope_outputs)},
            algorithm_version="export-pipeline-v4-raw-viewer-split",
        )
        if cache_entry.hit and cache.restore(cache_entry, workspace_dir):
            scene_manifest_path = workspace_dir / "scene_manifest.json"
            diagnostics_path = workspace_dir / "diagnostics_bundle.json"
            spark_path = workspace_dir / "spark_package.json"
            supersplat_path = workspace_dir / "supersplat_package.json"
            tileset_path = workspace_dir / "tileset.json"
            spz_path = workspace_dir / "viewer_asset.spz"
            raw_ply_path = workspace_dir / "raw_splat.ply"
            viewer_source_path = workspace_dir / "viewer_source.ply"
            optimized_candidates = sorted(path for path in workspace_dir.glob("viewer_asset.*") if path.is_file())
            optimized_viewer_path = optimized_candidates[0] if optimized_candidates else workspace_dir / "viewer_asset.ply"
            if scene_manifest_path.exists():
                scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
                diagnostics_bundle = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
                spark_package = json.loads(spark_path.read_text(encoding="utf-8")) if spark_path.exists() else {}
                supersplat_package = json.loads(supersplat_path.read_text(encoding="utf-8")) if supersplat_path.exists() else {}
                optimization = scene_manifest.get("viewer_asset_optimization") or {"status": "cache_hit"}
                spz_status = scene_manifest.get("spz_status") or {"status": "cache_hit" if spz_path.exists() else "not_generated"}
                tileset_status = scene_manifest.get("tileset_status") or {"status": "cache_hit"}
                optimization.update({"cache_hit": True, "cache_key": cache_entry.cache_key})
                return {
                    "workspace_dir": workspace_dir,
                    "outputs": {
                    "raw_ply": raw_ply_path,
                    "viewer_source_ply": viewer_source_path,
                    "optimized_viewer_asset": optimized_viewer_path,
                    "spark_package": spark_path,
                    "supersplat_package": supersplat_path,
                    **({"spz_asset": spz_path} if spz_path.exists() else {}),
                    "3d_tiles_splat": tileset_path,
                    "scene_manifest": scene_manifest_path,
                    "diagnostics_bundle": diagnostics_path,
                    **_cached_scope_output_paths(workspace_dir, scene_manifest),
                },
                    "scene_manifest": scene_manifest,
                    "diagnostics_bundle": diagnostics_bundle,
                    "spark_package": spark_package,
                    "supersplat_package": supersplat_package,
                    "optimization": optimization,
                    "spz_status": spz_status,
                    "tileset_status": tileset_status,
                    "cache_hit": True,
                    "cache_key": cache_entry.cache_key,
                }
        raw_source = _raw_source_path(splat_path, scope_outputs)
        viewer_source = _viewer_source_path(splat_path, scope_outputs)
        raw_ply_path = workspace_dir / "raw_splat.ply"
        if raw_source and raw_source.exists():
            shutil.copyfile(raw_source, raw_ply_path)
        else:
            raw_ply_path.write_bytes(b"")
        viewer_source_path = workspace_dir / "viewer_source.ply"
        if viewer_source and viewer_source.exists():
            shutil.copyfile(viewer_source, viewer_source_path)
        else:
            viewer_source_path.write_bytes(b"")

        export_config = settings.engine_config.get("operators", {}).get("export", {}) or {}
        optimized_viewer_path, optimization = _build_viewer_asset(export_config, viewer_source_path, workspace_dir)
        spz_path, spz_status = _build_spz_asset(export_config, viewer_source_path, workspace_dir)
        tileset_path, tileset_status = _build_3d_tiles_splat(export_config, viewer_source_path, workspace_dir, route)

        spark_package = {
            "format": "spark_package",
            "viewer": "SparkJS",
            "asset": optimized_viewer_path.name,
            "asset_format": optimized_viewer_path.suffix.lstrip(".") or "unknown",
            "optimization": optimization,
            "spz": spz_status,
            "raw_ply_is_final_product": True,
            "viewer_source_layer": _viewer_default(scope_outputs),
            "route": route,
        }
        supersplat_package = {
            "format": "supersplat_package",
            "asset": optimized_viewer_path.name,
            "optimization_status": optimization["status"],
            "optimizer": optimization,
            "spz": spz_status,
            "raw_ply_is_final_product": True,
            "viewer_source_layer": _viewer_default(scope_outputs),
        }
        scene_manifest = {
            "workflow_id": workflow.id,
            "project_id": workflow.project_id,
            "route": route,
            "quality": quality,
            "raw_ply_is_final_product": True,
            "raw_ply": raw_ply_path.name,
            "raw_source_layer": _publish_default(scope_outputs),
            "publish_default": _publish_default(scope_outputs),
            "viewer_default": _viewer_default(scope_outputs),
            "publish_requires": ["scene_manifest", "viewer_asset", "diagnostics_bundle"],
            "viewer_asset": optimized_viewer_path.name,
            "viewer_source_ply": viewer_source_path.name,
            "viewer_asset_optimization": optimization,
            "spz_asset": spz_path.name if spz_path else None,
            "spz_status": spz_status,
            "tileset": tileset_path.name,
            "tileset_status": tileset_status,
            "cells": [
                {
                    "cell_id": "cell_000",
                    "asset": optimized_viewer_path.name,
                    "source_layer": _viewer_default(scope_outputs),
                    "quality_grade": quality.get("quality_grade"),
                    "measurement_allowed": quality.get("measurement_allowed", False),
                }
            ],
            "reconstruction_scope": _scene_scope_manifest(scope_outputs),
        }
        diagnostics_bundle = {
            "workflow_id": workflow.id,
            "project_id": workflow.project_id,
            "route": route,
            "quality": quality,
            "export": {"viewer_asset_optimization": optimization, "spz_status": spz_status, "tileset_status": tileset_status},
            "diagnostics": diagnostics,
            "reconstruction_scope": _scene_scope_manifest(scope_outputs),
        }

        outputs = {
            "raw_ply": raw_ply_path,
            "viewer_source_ply": viewer_source_path,
            "optimized_viewer_asset": optimized_viewer_path,
            "spark_package": _write_json(workspace_dir / "spark_package.json", spark_package),
            "supersplat_package": _write_json(workspace_dir / "supersplat_package.json", supersplat_package),
            **({"spz_asset": spz_path} if spz_path else {}),
            "3d_tiles_splat": tileset_path,
            "scene_manifest": _write_json(workspace_dir / "scene_manifest.json", scene_manifest),
            "diagnostics_bundle": _write_json(workspace_dir / "diagnostics_bundle.json", diagnostics_bundle),
            **_copy_scope_outputs(scope_outputs, workspace_dir),
        }
        result = {
            "workspace_dir": workspace_dir,
            "outputs": outputs,
            "scene_manifest": scene_manifest,
            "diagnostics_bundle": diagnostics_bundle,
            "spark_package": spark_package,
            "supersplat_package": supersplat_package,
            "optimization": optimization,
            "spz_status": spz_status,
            "tileset_status": tileset_status,
            "cache_hit": False,
            "cache_key": cache_entry.cache_key,
        }
        cache.save(cache_entry, workspace_dir, metadata={"route": route, "quality": quality, "optimization": optimization, "spz_status": spz_status, "tileset_status": tileset_status})
        return result


def _scope_cache_payload(scope_outputs: dict[str, Any] | None) -> dict[str, Any]:
    if not scope_outputs:
        return {"enabled": False}
    return {
        "enabled": True,
        "publish_default": scope_outputs.get("publish_default"),
        "report": scope_outputs.get("report_summary"),
        "paths": {key: str(value) for key, value in (scope_outputs.get("paths") or {}).items()},
    }


def _publish_default(scope_outputs: dict[str, Any] | None) -> str:
    if not scope_outputs:
        return "raw_ply"
    return str(scope_outputs.get("publish_default") or "subject_model")


def _viewer_default(scope_outputs: dict[str, Any] | None) -> str:
    if not scope_outputs:
        return "raw_ply"
    report = scope_outputs.get("report_summary") or {}
    return str(report.get("viewer_default") or "viewer_model")


def _viewer_source_path(splat_path: Path | None, scope_outputs: dict[str, Any] | None) -> Path | None:
    paths = (scope_outputs or {}).get("paths") or {}
    viewer_candidate = paths.get(_viewer_default(scope_outputs)) or paths.get("viewer_model")
    if viewer_candidate:
        path = Path(str(viewer_candidate))
        if path.exists():
            return path
    publish_default = _publish_default(scope_outputs)
    candidate = paths.get(publish_default) or paths.get("subject_model")
    if candidate:
        path = Path(str(candidate))
        if path.exists():
            return path
    return splat_path


def _raw_source_path(splat_path: Path | None, scope_outputs: dict[str, Any] | None) -> Path | None:
    paths = (scope_outputs or {}).get("paths") or {}
    publish_default = _publish_default(scope_outputs)
    for key in (publish_default, "raw_model", "model_full", "subject_model"):
        candidate = paths.get(key)
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    return splat_path


def _scene_scope_manifest(scope_outputs: dict[str, Any] | None) -> dict[str, Any]:
    if not scope_outputs:
        return {"enabled": False}
    report = scope_outputs.get("report_summary") or {}
    return {
        "enabled": True,
        "publish_default": _publish_default(scope_outputs),
        "viewer_default": _viewer_default(scope_outputs),
        "layered_loading": report.get("layered_loading"),
        "layers": {
            "raw_model": "canonical_full_quality_model",
            "subject_model": "high_quality_default",
            "viewer_model": "browser_budget_preview",
            "context_model_lowres": "optional_context_reference",
            "full_model_debug": "diagnostics_only",
        },
        "mask_manifest": str((scope_outputs.get("paths") or {}).get("mask_manifest") or ""),
        "spatial_crop_manifest": str((scope_outputs.get("paths") or {}).get("spatial_crop_manifest") or ""),
        "gaussian_pruning_report": report,
    }


def _copy_scope_outputs(scope_outputs: dict[str, Any] | None, workspace_dir: Path) -> dict[str, Path]:
    if not scope_outputs:
        return {}
    specs = {
        "mask_manifest": "mask_manifest.json",
        "spatial_crop_manifest": "spatial_crop_manifest.json",
        "gaussian_pruning_report": "gaussian_pruning_report.json",
    }
    outputs: dict[str, Path] = {}
    paths = scope_outputs.get("paths") or {}
    for key, filename in specs.items():
        source_value = paths.get(key)
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.exists():
            continue
        target = workspace_dir / filename
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        outputs[key] = target
    return outputs


def _cached_scope_output_paths(workspace_dir: Path, scene_manifest: dict[str, Any]) -> dict[str, Path]:
    if not (scene_manifest.get("reconstruction_scope") or {}).get("enabled"):
        return {}
    specs = {
        "mask_manifest": "mask_manifest.json",
        "spatial_crop_manifest": "spatial_crop_manifest.json",
        "gaussian_pruning_report": "gaussian_pruning_report.json",
    }
    return {key: workspace_dir / filename for key, filename in specs.items() if (workspace_dir / filename).exists()}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_viewer_asset(export_config: dict[str, Any], raw_ply_path: Path, workspace_dir: Path) -> tuple[Path, dict[str, Any]]:
    splat_transform_config = export_config.get("splat_transform", {}) or {}
    output_suffix = str(splat_transform_config.get("output_suffix") or ".splat")
    optimized_path = workspace_dir / f"viewer_asset{output_suffix}"
    timeout_seconds = int(splat_transform_config.get("timeout_seconds") or 900)
    if raw_ply_path.exists() and raw_ply_path.stat().st_size > 0:
        command_template = splat_transform_config.get("command") or ["{binary}", "{input_ply}", "{output_asset}"]
        binary = _resolve_binary(str(splat_transform_config.get("binary") or "splat-transform"))
        if binary:
            values = {"binary": binary, "input_ply": str(raw_ply_path), "output_asset": str(optimized_path), "workspace_dir": str(workspace_dir)}
            command = [_format_value(str(part), values) for part in command_template]
            command_result = _run_command_with_timeout("export.optimized_viewer_asset", "export_optimized_viewer_asset", command, workspace_dir, timeout_seconds=timeout_seconds)
            if command_result.exit_code == 0 and optimized_path.exists() and optimized_path.stat().st_size > 0:
                return optimized_path, {
                    "status": "optimized",
                    "tool": "splat-transform",
                    "command": command,
                    "exit_code": command_result.exit_code,
                    "output_asset": optimized_path.name,
                    "raw_ply_is_final_product": False,
                }
            fallback = workspace_dir / "viewer_asset.ply"
            shutil.copyfile(raw_ply_path, fallback)
            return fallback, {
                "status": "optimizer_failed_unoptimized_fallback",
                "tool": "splat-transform",
                "command": command,
                "exit_code": command_result.exit_code,
                "stderr_tail": command_result.stderr[-2000:] if command_result.stderr else "",
                "output_asset": fallback.name,
                "raw_ply_is_final_product": False,
            }
        fallback = workspace_dir / "viewer_asset.ply"
        shutil.copyfile(raw_ply_path, fallback)
        return fallback, {
            "status": "optimizer_unavailable_unoptimized_fallback",
            "tool": "splat-transform",
            "reason": "splat_transform_binary_missing",
            "install_hint": splat_transform_config.get("install_hint") or "npm install -g @playcanvas/splat-transform",
            "output_asset": fallback.name,
            "raw_ply_is_final_product": False,
        }
    optimized_path.write_bytes(b"")
    return optimized_path, {"status": "skipped_empty_raw_ply", "output_asset": optimized_path.name, "raw_ply_is_final_product": False}


def _build_spz_asset(export_config: dict[str, Any], raw_ply_path: Path, workspace_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    spz_config = export_config.get("spz", {}) or {}
    if not spz_config.get("enabled", True):
        return None, {"status": "disabled", "raw_ply_is_final_product": False}
    command_template = spz_config.get("command")
    binary = _resolve_binary(str(spz_config.get("binary") or ""))
    output_path = workspace_dir / str(spz_config.get("output_filename") or "viewer_asset.spz")
    if not raw_ply_path.exists() or raw_ply_path.stat().st_size <= 0:
        return None, {"status": "skipped_empty_raw_ply", "raw_ply_is_final_product": False}
    if not command_template or not binary:
        return None, {
            "status": "converter_unavailable",
            "reason": "spz_converter_not_configured_or_missing",
            "raw_ply_is_final_product": False,
            "install_hint": spz_config.get("install_hint"),
        }
    values = {
        "binary": binary,
        "input_ply": str(raw_ply_path),
        "output_asset": str(output_path),
        "workspace_dir": str(workspace_dir),
    }
    command = [_format_value(str(part), values) for part in command_template]
    timeout_seconds = int(spz_config.get("timeout_seconds") or 900)
    command_result = _run_command_with_timeout("export.spz_asset", "export_optimized_viewer_asset", command, workspace_dir, timeout_seconds=timeout_seconds)
    if command_result.exit_code == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path, {
            "status": "generated",
            "tool": spz_config.get("tool_name") or "spz",
            "command": command,
            "exit_code": command_result.exit_code,
            "output_asset": output_path.name,
            "raw_ply_is_final_product": False,
        }
    return None, {
        "status": "converter_failed",
        "tool": spz_config.get("tool_name") or "spz",
        "command": command,
        "exit_code": command_result.exit_code,
        "stderr_tail": command_result.stderr[-2000:] if command_result.stderr else "",
        "raw_ply_is_final_product": False,
    }


def _build_3d_tiles_splat(export_config: dict[str, Any], viewer_asset_path: Path, workspace_dir: Path, route: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    tiles_config = export_config.get("three_d_tiles", {}) or {}
    tileset_path = workspace_dir / "tileset.json"
    command_template = tiles_config.get("command")
    binary = _resolve_binary(str(tiles_config.get("binary") or "")) if tiles_config.get("binary") else None
    if command_template and binary and viewer_asset_path.exists() and viewer_asset_path.stat().st_size > 0:
        output_dir = workspace_dir / "tiles"
        output_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "binary": binary,
            "viewer_asset": str(viewer_asset_path),
            "output_dir": str(output_dir),
            "tileset_json": str(tileset_path),
            "workspace_dir": str(workspace_dir),
        }
        command = [_format_value(str(part), values) for part in command_template]
        command_result = _run_command("export.3d_tiles_splat", "export_scene_manifest", command, workspace_dir)
        generated_tileset = output_dir / "tileset.json"
        if generated_tileset.exists() and generated_tileset != tileset_path:
            shutil.copyfile(generated_tileset, tileset_path)
        if command_result.exit_code == 0 and tileset_path.exists() and tileset_path.stat().st_size > 0:
            return tileset_path, {
                "status": "generated",
                "tool": tiles_config.get("tool_name") or Path(binary).name,
                "command": command,
                "exit_code": command_result.exit_code,
                "raw_ply_is_final_product": False,
            }
        status = {
            "status": "converter_failed_manifest_only",
            "tool": tiles_config.get("tool_name") or Path(binary).name,
            "command": command,
            "exit_code": command_result.exit_code,
            "stderr_tail": command_result.stderr[-2000:] if command_result.stderr else "",
            "raw_ply_is_final_product": False,
        }
    else:
        status = {
            "status": "converter_unavailable_manifest_only",
            "reason": "3d_tiles_splat_converter_not_configured_or_missing",
            "raw_ply_is_final_product": False,
            "install_hint": tiles_config.get("install_hint"),
        }
    manifest = {
        "asset": {"version": "1.1"},
        "extensionsUsed": ["3DTILES_content_gltf"] if route.get("chunked") else [],
        "content_format": viewer_asset_path.suffix.lstrip(".") or "unknown",
        "conversion_status": status,
        "note": "This artifact is a diagnostic manifest unless conversion_status.status == generated.",
    }
    _write_json(tileset_path, manifest)
    return tileset_path, status


def _resolve_binary(binary: str) -> str | None:
    if not binary:
        return None
    configured = Path(binary)
    if configured.exists():
        return str(configured)
    resolved = shutil.which(binary)
    return resolved or None


def _run_command(operator_name: str, stage_key: str, command: list[str], cwd: Path) -> CommandResult:
    return _run_command_with_timeout(operator_name, stage_key, command, cwd, timeout_seconds=None)


def _run_command_with_timeout(
    operator_name: str,
    stage_key: str,
    command: list[str],
    cwd: Path,
    *,
    timeout_seconds: int | None,
) -> CommandResult:
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        stdout = completed.stdout[-4000:] if completed.stdout else ""
        stderr = completed.stderr[-4000:] if completed.stderr else ""
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        stderr = (stderr + f"\ncommand timed out after {timeout_seconds}s").strip()
        exit_code = 124
    finished = datetime.now(timezone.utc)
    return CommandResult(
        operator_name=operator_name,
        stage_key=stage_key,
        command=command,
        cwd=str(cwd),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        started_at=started,
        finished_at=finished,
    )


def _format_value(value: str, values: dict[str, str]) -> str:
    return value.format(**values)
