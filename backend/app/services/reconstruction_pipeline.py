from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Asset, Workflow
from app.services.artifact_service import ArtifactService
from app.services.stage_optimizer import (
    DEFAULT_PRODUCTION_ROUTE_PRESET,
    DatasetAssemblyStage,
    FinalArtifactSelectionStage,
    GaussianTrainingOptimizationStage,
    ImageEnhancementStage,
    MaskOptimizationStage,
    OPTIMIZED_STAGE_NAMES,
    ROUTE_PRESETS,
    ROUTE_SCOPED_STAGE_NAMES,
    PanoramaNormalizationStage,
    PoseEstimationOptimizationStage,
    RawMediaInspectionStage,
    RenderEvaluationStage,
    RunRecordStore,
    StageContext,
    StageOptimizer,
    TrainingInputOptimizationStage,
    VideoKeyframeOptimizationStage,
    write_json,
)
from app.services.storage_service import StorageService
from app.services.workflow_log_service import append_workflow_log
from app.services.workflow_state_service import ensure_workflow_stages, update_stage


OPTIMIZED_RECONSTRUCTION_TYPE = "stage_optimized_reconstruction"


STAGE_CLASSES: dict[str, type[StageOptimizer]] = {
    "raw_media_inspection": RawMediaInspectionStage,
    "image_enhancement": ImageEnhancementStage,
    "video_keyframe_optimization": VideoKeyframeOptimizationStage,
    "panorama_normalization": PanoramaNormalizationStage,
    "dataset_assembly": DatasetAssemblyStage,
    "pose_estimation_optimization": PoseEstimationOptimizationStage,
    "mask_optimization": MaskOptimizationStage,
    "training_input_optimization": TrainingInputOptimizationStage,
    "gaussian_training_optimization": GaussianTrainingOptimizationStage,
    "render_evaluation": RenderEvaluationStage,
    "final_artifact_selection": FinalArtifactSelectionStage,
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def optimized_run_dir(workflow_id: str) -> Path:
    settings = get_settings()
    return Path(settings.workspace_root) / "runs" / workflow_id / "optimized_reconstruction"


def optimized_status_path(workflow_id: str) -> Path:
    return optimized_run_dir(workflow_id) / "optimized_reconstruction_status.json"


def load_optimized_json(workflow_id: str, relative_path: str, default: Any) -> Any:
    path = optimized_run_dir(workflow_id) / relative_path
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_optimized_status(workflow_id: str) -> dict[str, Any]:
    return load_optimized_json(workflow_id, "optimized_reconstruction_status.json", {})


def _build_config(workflow: Workflow) -> dict[str, Any]:
    settings = get_settings()
    engine_config = settings.engine_config or {}
    stage_config = engine_config.get("stage_optimized_reconstruction")
    if not isinstance(stage_config, dict):
        stage_config = {}
    config = _deep_merge(stage_config, workflow.config_json or {})
    config.setdefault("preserve_forensic_integrity", True)
    config.setdefault("stop_when_stage_optimal", True)
    config.setdefault("allow_ai_enhance", False)
    config.setdefault("allow_super_resolution", False)
    config.setdefault("allow_deblur", True)
    config.setdefault("allow_denoise", True)
    config.setdefault("allow_mask", True)
    config.setdefault("allow_splatfacto_w", True)
    config.setdefault("allow_big_model", True)
    config.setdefault("route_preset", DEFAULT_PRODUCTION_ROUTE_PRESET)
    config.setdefault("execute_route_matrix", False)
    return config


def _load_assets(db: Session, workflow: Workflow) -> list[Asset]:
    input_json = workflow.input_json or {}
    asset_ids = list(input_json.get("asset_ids") or [])
    if not asset_ids:
        return []
    assets = db.query(Asset).filter(Asset.id.in_(asset_ids), Asset.project_id == workflow.project_id).all()
    by_id = {asset.id: asset for asset in assets}
    return [by_id[asset_id] for asset_id in asset_ids if asset_id in by_id]


def _write_status(context: StageContext, status: dict[str, Any]) -> None:
    write_json(context.run_dir / "optimized_reconstruction_status.json", status)


def _stage_artifact_paths(context: StageContext, stage_name: str) -> dict[str, str]:
    stage_dir = context.run_dir / "stages" / stage_name
    return {
        "stage_result": str(stage_dir / "stage_result.json"),
        "stage_report": str(stage_dir / "stage_report.md"),
        "candidate_metrics": str(stage_dir / "candidate_metrics.json"),
    }


def _stage_artifact_paths_for_context(context: StageContext, stage_name: str) -> dict[str, str]:
    stage_dir = context.stage_dir(stage_name)
    return {
        "stage_result": str(stage_dir / "stage_result.json"),
        "stage_report": str(stage_dir / "stage_report.md"),
        "candidate_metrics": str(stage_dir / "candidate_metrics.json"),
    }


def _route_set(config: dict[str, Any]) -> list[str]:
    matrix = config.get("route_matrix") if isinstance(config.get("route_matrix"), dict) else {}
    if config.get("benchmark_mode") or matrix.get("benchmark_mode"):
        requested = matrix.get("benchmark_routes") or ["original_pose_original_train", "safe_pose_original_train"]
    else:
        requested = matrix.get("default_routes") or ["original_pose_original_train", "safe_pose_original_train"]
    routes = [str(route) for route in requested if str(route) in ROUTE_PRESETS]
    return routes or ["original_pose_original_train", "safe_pose_original_train"]


def _ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        header = b""
        while b"end_header" not in header:
            line = handle.readline()
            if not line:
                break
            header += line
    match = re.search(rb"element vertex (\d+)", header)
    return int(match.group(1)) if match else None


def _route_metrics(context: StageContext, route: str) -> dict[str, Any]:
    previous_route = context.config.get("active_route_id")
    previous_preset = context.config.get("active_route_preset")
    context.config["active_route_id"] = route
    context.config["active_route_preset"] = route
    try:
        pose = load_optimized_json_for_path(context.stage_dir("pose_estimation_optimization") / "best_pose_selection.json", {})
        pose_metrics = pose.get("metrics") or {}
        training = load_optimized_json_for_path(context.stage_dir("gaussian_training_optimization") / "best_training_selection.json", {})
        training_metrics = training.get("metrics") or {}
        render = load_optimized_json_for_path(context.stage_dir("render_evaluation") / "eval_metrics.json", {})
        final = load_optimized_json_for_path(context.stage_dir("final_artifact_selection") / "run_final_selection.json", {})
        ply_path_value = final.get("best_model_path") or training.get("splat_path") or training_metrics.get("splat_path")
        ply_path = Path(str(ply_path_value)) if ply_path_value else None
        registered = pose_metrics.get("registered_images_count")
        total = pose_metrics.get("total_images_count")
        return {
            "route_preset": route,
            "registered_images": f"{registered}/{total}" if registered is not None and total is not None else None,
            "registered_images_count": registered,
            "total_images_count": total,
            "registration_ratio": pose_metrics.get("registered_ratio"),
            "reprojection_error_px": pose_metrics.get("mean_reprojection_error"),
            "sparse_points": pose_metrics.get("sparse_points_count"),
            "psnr": render.get("PSNR") or training_metrics.get("final_eval_psnr"),
            "ssim": render.get("SSIM") or training_metrics.get("final_eval_ssim"),
            "lpips": render.get("LPIPS") or training_metrics.get("final_eval_lpips"),
            "gaussian_count": training_metrics.get("gaussian_count"),
            "ply_size_mb": round(ply_path.stat().st_size / 1024 / 1024, 1) if ply_path and ply_path.exists() else None,
            "ply_vertex_count": _ply_vertex_count(ply_path) if ply_path else None,
            "final_score": final.get("final_score"),
            "quality_level": final.get("quality_level"),
            "best_model_path": str(ply_path) if ply_path else None,
            "training_supervision_modified": bool(training_metrics.get("training_supervision_modified")),
        }
    finally:
        if previous_route is None:
            context.config.pop("active_route_id", None)
        else:
            context.config["active_route_id"] = previous_route
        if previous_preset is None:
            context.config.pop("active_route_preset", None)
        else:
            context.config["active_route_preset"] = previous_preset


def load_optimized_json_for_path(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _select_best_route(metrics_table: dict[str, dict[str, Any]]) -> tuple[str | None, list[str]]:
    baseline = metrics_table.get("original_pose_original_train") or {}
    candidates = {name: metrics for name, metrics in metrics_table.items() if name != "original_pose_original_train"}
    baseline_ratio = float(baseline.get("registration_ratio") or 0.0)
    eligible = {
        name: metrics
        for name, metrics in candidates.items()
        if float(metrics.get("registration_ratio") or 0.0) >= baseline_ratio
    }
    if not eligible:
        return ("original_pose_original_train" if baseline else None, ["no_candidate_met_baseline_registration_ratio"])

    def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, float, float]:
        name, metrics = item
        final_score = float(metrics.get("final_score") or 0.0)
        psnr = float(metrics.get("psnr") or 0.0)
        ssim = float(metrics.get("ssim") or 0.0)
        lpips = float(metrics.get("lpips") or 1.0)
        reproj = float(metrics.get("reprojection_error_px") or 999.0)
        supervision_penalty = 0.015 if metrics.get("training_supervision_modified") else 0.0
        return (final_score - supervision_penalty, -reproj, psnr, ssim, -lpips, 0.0 if metrics.get("training_supervision_modified") else 1.0)

    best_name, best_metrics = max(eligible.items(), key=rank)
    if baseline and float(baseline.get("final_score") or 0.0) > float(best_metrics.get("final_score") or 0.0):
        return "original_pose_original_train", ["baseline_final_score_higher_than_safe_candidates"]
    reasons = []
    if float(best_metrics.get("registration_ratio") or 0.0) >= baseline_ratio:
        reasons.append("registration_ratio_equal_or_above_baseline")
    if baseline.get("reprojection_error_px") is not None and best_metrics.get("reprojection_error_px") is not None and float(best_metrics["reprojection_error_px"]) < float(baseline["reprojection_error_px"]):
        reasons.append("reprojection_error_improved")
    if float(best_metrics.get("psnr") or 0.0) > float(baseline.get("psnr") or 0.0):
        reasons.append("PSNR_improved")
    if float(best_metrics.get("ssim") or 0.0) > float(baseline.get("ssim") or 0.0):
        reasons.append("SSIM_improved")
    if baseline.get("lpips") is not None and best_metrics.get("lpips") is not None and float(best_metrics["lpips"]) < float(baseline["lpips"]):
        reasons.append("LPIPS_improved")
    if float(best_metrics.get("final_score") or 0.0) > float(baseline.get("final_score") or 0.0):
        reasons.append("final_score_improved")
    if not best_metrics.get("training_supervision_modified"):
        reasons.append("training_supervision_remains_original")
    else:
        reasons.append("training_supervision_modified")
    return best_name, reasons


def _run_optimizer_stage(
    db: Session,
    workflow: Workflow,
    context: StageContext,
    record_store: RunRecordStore,
    capability_report: dict[str, Any],
    stage_name: str,
    *,
    stage_index: int,
    stage_count: int,
    progress_base: float,
) -> dict[str, Any]:
    stage_class = STAGE_CLASSES[stage_name]
    optimizer = stage_class()
    update_stage(
        db,
        workflow,
        stage_name,
        status="running",
        progress=progress_base,
        input_summary={"asset_count": len(context.assets), "stage": stage_name, "route": context.config.get("active_route_id")},
        log_message=f"optimized_stage_started:{stage_name}",
    )
    workflow.status = "running"
    workflow.progress = max(workflow.progress, progress_base)
    _write_status(
        context,
        {
            "workflow_id": workflow.id,
            "status": workflow.status,
            "current_stage": stage_name,
            "current_route": context.config.get("active_route_id"),
            "stage_index": stage_index,
            "stage_count": stage_count,
            "previous_best": {key: value.get("best_artifact") for key, value in context.previous_results.items()},
            "capability_report_path": str(context.run_dir / "capability_report.json"),
        },
    )
    db.commit()
    result = optimizer.run(context)
    if result.get("status") != "succeeded":
        update_stage(
            db,
            workflow,
            stage_name,
            status="blocked",
            progress=1.0,
            output_summary={
                "best_artifact": result.get("best_artifact"),
                "candidate_count": len(result.get("candidate_artifacts") or []),
                "rejection_reasons": result.get("rejection_reasons") or {},
                "paths": _stage_artifact_paths_for_context(context, stage_name),
            },
            error_message=f"{stage_name} did not produce a quality-gate-passing best artifact",
            log_message=f"optimized_stage_blocked:{stage_name}",
            log_level="warning",
        )
        workflow.status = "blocked_by_quality_gate"
        workflow.error_message = f"{stage_name} did not produce a quality-gate-passing best artifact"
        workflow.quality_json = {
            **(workflow.quality_json or {}),
            "quality_grade": "D",
            "measurement_allowed": False,
            "stage_optimized_reconstruction": {
                "status": "blocked",
                "blocked_stage": stage_name,
                "blocked_route": context.config.get("active_route_id"),
                "reason": "no_quality_gate_passing_best_artifact",
                "latest_stage_result": result,
                "records": record_store.read_all(),
                "capability_report": capability_report,
            },
        }
        db.commit()
        return result
    context.previous_results[stage_name] = result
    update_stage(
        db,
        workflow,
        stage_name,
        status="succeeded",
        progress=1.0,
        output_summary={
            "route": context.config.get("active_route_id"),
            "best_artifact": result.get("best_artifact"),
            "candidate_count": len(result.get("candidate_artifacts") or []),
            "has_remaining_improvement": result.get("whether_stage_has_remaining_improvement"),
            "paths": _stage_artifact_paths_for_context(context, stage_name),
        },
        log_message=f"optimized_stage_succeeded:{stage_name}",
    )
    workflow.progress = max(workflow.progress, progress_base)
    db.commit()
    return result


def _run_route_matrix_reconstruction(db: Session, workflow: Workflow, context: StageContext, capability_report: dict[str, Any]) -> dict[str, Any]:
    record_store = RunRecordStore(context.run_dir)
    assets = context.assets
    routes = _route_set(context.config)
    common_stage_names = [name for name in OPTIMIZED_STAGE_NAMES if name not in ROUTE_SCOPED_STAGE_NAMES]
    route_stage_names = [name for name in OPTIMIZED_STAGE_NAMES if name in ROUTE_SCOPED_STAGE_NAMES]
    total_units = len(common_stage_names) + len(routes) * len(route_stage_names)
    completed_units = 0
    common_results: dict[str, Any] = {}
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Stage optimized route matrix started",
        event={"event_type": "optimized_reconstruction.route_matrix_started", "asset_count": len(assets), "routes": routes},
    )
    db.commit()
    context.config.pop("active_route_id", None)
    context.config.pop("active_route_preset", None)
    context.previous_results = {}
    for stage_name in common_stage_names:
        result = _run_optimizer_stage(
            db,
            workflow,
            context,
            record_store,
            capability_report,
            stage_name,
            stage_index=completed_units + 1,
            stage_count=total_units,
            progress_base=completed_units / max(total_units, 1),
        )
        completed_units += 1
        if result.get("status") != "succeeded":
            return workflow.quality_json or {}
        common_results[stage_name] = result

    route_results: dict[str, dict[str, Any]] = {}
    for route in routes:
        context.config["active_route_id"] = route
        context.config["active_route_preset"] = route
        context.config["route_preset"] = route
        context.previous_results = dict(common_results)
        route_results[route] = {}
        append_workflow_log(
            db,
            workflow_id=workflow.id,
            message=f"Stage optimized route started: {route}",
            event={"event_type": "optimized_reconstruction.route_started", "route": route},
        )
        db.commit()
        for stage_name in route_stage_names:
            result = _run_optimizer_stage(
                db,
                workflow,
                context,
                record_store,
                capability_report,
                stage_name,
                stage_index=completed_units + 1,
                stage_count=total_units,
                progress_base=completed_units / max(total_units, 1),
            )
            completed_units += 1
            route_results[route][stage_name] = result
            if result.get("status") != "succeeded":
                return workflow.quality_json or {}

    metrics_table = {route: _route_metrics(context, route) for route in routes}
    best_route, reasons = _select_best_route(metrics_table)
    comparison = {
        "baseline_route": "original_pose_original_train",
        "candidate_routes": [route for route in routes if route != "original_pose_original_train"],
        "best_route": best_route,
        "reason": reasons,
        "metrics_table": metrics_table,
    }
    comparison_dir = context.run_dir / "route_matrix"
    comparison_path = write_json(comparison_dir / "route_comparison.json", comparison)
    artifact_service = context.artifact_service
    artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="route_comparison",
        stage="route_matrix",
        relative_path=f"optimized_runs/{workflow.id}/route_matrix/route_comparison.json",
        source_path=str(comparison_path),
        mime_type="application/json",
        is_primary=True,
    )
    best_metrics = metrics_table.get(best_route or "", {})
    workflow.status = "completed"
    workflow.progress = 1.0
    workflow.quality_json = {
        **(workflow.quality_json or {}),
        "quality_grade": best_metrics.get("quality_level") or "production_candidate",
        "measurement_allowed": False,
        "stage_optimized_reconstruction": {
            "status": workflow.status,
            "route_matrix": True,
            "best_route": best_route,
            "final_score": best_metrics.get("final_score"),
            "quality_level": best_metrics.get("quality_level"),
            "best_model": best_metrics.get("best_model_path"),
            "route_comparison": str(comparison_path),
            "stage_count": total_units,
            "records": record_store.read_all(),
            "capability_report": capability_report,
        },
    }
    _write_status(
        context,
        {
            "workflow_id": workflow.id,
            "status": workflow.status,
            "current_stage": None,
            "route_matrix": True,
            "best_route": best_route,
            "final_score": best_metrics.get("final_score"),
            "quality_level": best_metrics.get("quality_level"),
            "best_model": best_metrics.get("best_model_path"),
            "route_comparison": str(comparison_path),
            "routes": routes,
            "metrics_table": metrics_table,
            "records": record_store.read_all(),
        },
    )
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Stage optimized route matrix completed",
        event={"event_type": "optimized_reconstruction.route_matrix_completed", "best_route": best_route, "final_score": best_metrics.get("final_score")},
    )
    db.commit()
    return workflow.quality_json


def run_stage_optimized_reconstruction(
    db: Session,
    workflow_id: str,
    *,
    only_stage: str | None = None,
) -> dict[str, Any]:
    workflow = db.get(Workflow, workflow_id)
    if workflow is None:
        raise ValueError(f"Workflow not found: {workflow_id}")
    ensure_workflow_stages(db, workflow)
    settings = get_settings()
    storage = StorageService(settings)
    artifact_service = ArtifactService(db, storage)
    run_dir = optimized_run_dir(workflow.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    assets = _load_assets(db, workflow)
    config = _build_config(workflow)
    capability_report: dict[str, Any] = {
        "opencv": True,
        "pillow": True,
        "lightglue_aliked": "adapter" if config.get("enable_hloc_lightglue") else "capability_unavailable",
        "mast3r_dust3r": "adapter" if config.get("enable_mast3r_dust3r") else "capability_unavailable",
        "sam2": "adapter" if config.get("enable_sam2") else "capability_unavailable",
        "super_resolution": "adapter" if config.get("allow_super_resolution") and config.get("super_resolution_adapter_enabled") else "capability_unavailable",
        "real_gaussian_training": bool(config.get("execute_training") or settings.nerfstudio_fake_runner or config.get("fake_runner")),
    }
    context = StageContext(
        db=db,
        workflow=workflow,
        assets=assets,
        settings=settings,
        storage=storage,
        artifact_service=artifact_service,
        run_dir=run_dir,
        config=config,
        previous_results={},
        capability_report=capability_report,
    )
    write_json(run_dir / "capability_report.json", capability_report)
    artifact_service.register_file(
        project_id=workflow.project_id,
        workflow_id=workflow.id,
        artifact_type="optimized_capability_report",
        stage="optimized_reconstruction",
        relative_path=f"optimized_runs/{workflow.id}/capability_report.json",
        source_path=str(run_dir / "capability_report.json"),
        mime_type="application/json",
    )
    workflow.status = "running"
    workflow.progress = max(workflow.progress, 0.01)
    db.flush()
    db.commit()

    if not assets:
        workflow.status = "blocked_by_quality_gate"
        workflow.error_message = "stage optimized reconstruction requires explicit asset_ids"
        workflow.quality_json = {
            **(workflow.quality_json or {}),
            "quality_grade": "D",
            "measurement_allowed": False,
            "stage_optimized_reconstruction": {
                "status": "blocked",
                "reason": "no_assets",
                "message": "No asset_ids were provided for this run.",
            },
        }
        _write_status(
            context,
            {
                "workflow_id": workflow.id,
                "status": workflow.status,
                "current_stage": None,
                "message": "No asset_ids were provided for this run.",
                "stages": [],
            },
        )
        db.commit()
        return workflow.quality_json

    route_matrix_config = config.get("route_matrix") if isinstance(config.get("route_matrix"), dict) else {}
    if only_stage is None and bool(config.get("execute_route_matrix") or route_matrix_config.get("enabled")):
        return _run_route_matrix_reconstruction(db, workflow, context, capability_report)

    stage_names = [only_stage] if only_stage else list(OPTIMIZED_STAGE_NAMES)
    final_results: dict[str, Any] = {}
    record_store = RunRecordStore(run_dir)
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Stage optimized reconstruction started",
        event={"event_type": "optimized_reconstruction.started", "asset_count": len(assets), "stages": stage_names},
    )
    db.commit()

    for index, stage_name in enumerate(stage_names):
        stage_class = STAGE_CLASSES[stage_name]
        optimizer = stage_class()
        progress_base = index / max(len(stage_names), 1)
        update_stage(
            db,
            workflow,
            stage_name,
            status="running",
            progress=progress_base,
            input_summary={"asset_count": len(assets), "stage": stage_name},
            log_message=f"optimized_stage_started:{stage_name}",
        )
        workflow.status = "running"
        workflow.progress = max(workflow.progress, progress_base)
        _write_status(
            context,
            {
                "workflow_id": workflow.id,
                "status": workflow.status,
                "current_stage": stage_name,
                "stage_index": index + 1,
                "stage_count": len(stage_names),
                "previous_best": {key: value.get("best_artifact") for key, value in context.previous_results.items()},
                "capability_report_path": str(run_dir / "capability_report.json"),
            },
        )
        db.commit()
        try:
            result = optimizer.run(context)
        except Exception as exc:
            update_stage(
                db,
                workflow,
                stage_name,
                status="failed",
                progress=1.0,
                error_message=str(exc),
                log_message=f"optimized_stage_failed:{stage_name}:{exc}",
                log_level="bug",
            )
            workflow.status = "failed"
            workflow.error_message = f"{stage_name}: {exc}"
            workflow.quality_json = {
                **(workflow.quality_json or {}),
                "quality_grade": "D",
                "measurement_allowed": False,
                "stage_optimized_reconstruction": {
                    "status": "failed",
                    "failed_stage": stage_name,
                    "error": str(exc),
                    "records": record_store.read_all(),
                },
            }
            _write_status(
                context,
                {
                    "workflow_id": workflow.id,
                    "status": workflow.status,
                    "current_stage": stage_name,
                    "error": str(exc),
                    "stages": list(context.previous_results.values()),
                },
            )
            db.commit()
            raise
        if result.get("status") != "succeeded":
            update_stage(
                db,
                workflow,
                stage_name,
                status="blocked",
                progress=1.0,
                output_summary={
                    "best_artifact": result.get("best_artifact"),
                    "candidate_count": len(result.get("candidate_artifacts") or []),
                    "rejection_reasons": result.get("rejection_reasons") or {},
                    "paths": _stage_artifact_paths(context, stage_name),
                },
                error_message=f"{stage_name} did not produce a quality-gate-passing best artifact",
                log_message=f"optimized_stage_blocked:{stage_name}",
                log_level="warning",
            )
            context.previous_results[stage_name] = result
            final_results[stage_name] = result
            workflow.status = "blocked_by_quality_gate"
            workflow.progress = max(workflow.progress, (index + 1) / max(len(stage_names), 1))
            workflow.error_message = f"{stage_name} did not produce a quality-gate-passing best artifact"
            workflow.quality_json = {
                **(workflow.quality_json or {}),
                "quality_grade": "D",
                "measurement_allowed": False,
                "stage_optimized_reconstruction": {
                    "status": "blocked",
                    "blocked_stage": stage_name,
                    "reason": "no_quality_gate_passing_best_artifact",
                    "latest_stage_result": result,
                    "records": record_store.read_all(),
                    "capability_report": capability_report,
                },
            }
            _write_status(
                context,
                {
                    "workflow_id": workflow.id,
                    "status": workflow.status,
                    "current_stage": None,
                    "blocked_stage": stage_name,
                    "error": workflow.error_message,
                    "stages": list(context.previous_results.values()),
                    "records": record_store.read_all(),
                },
            )
            append_workflow_log(
                db,
                workflow_id=workflow.id,
                message="Stage optimized reconstruction blocked by quality gate",
                event={"event_type": "optimized_reconstruction.blocked", "blocked_stage": stage_name, "reason": "no_quality_gate_passing_best_artifact"},
                level="warning",
            )
            db.commit()
            return workflow.quality_json
        context.previous_results[stage_name] = result
        final_results[stage_name] = result
        update_stage(
            db,
            workflow,
            stage_name,
            status="succeeded",
            progress=1.0,
            output_summary={
                "best_artifact": result.get("best_artifact"),
                "candidate_count": len(result.get("candidate_artifacts") or []),
                "has_remaining_improvement": result.get("whether_stage_has_remaining_improvement"),
                "paths": _stage_artifact_paths(context, stage_name),
            },
            log_message=f"optimized_stage_succeeded:{stage_name}",
        )
        workflow.progress = max(workflow.progress, (index + 1) / max(len(stage_names), 1))
        workflow.quality_json = {
            **(workflow.quality_json or {}),
            "stage_optimized_reconstruction": {
                "status": "running",
                "current_stage": stage_name,
                "completed_stage_count": index + 1,
                "stage_count": len(stage_names),
                "latest_best_artifact": result.get("best_artifact"),
            },
        }
        _write_status(
            context,
            {
                "workflow_id": workflow.id,
                "status": workflow.status,
                "current_stage": stage_name,
                "stage_index": index + 1,
                "stage_count": len(stage_names),
                "stages": list(context.previous_results.values()),
                "records": record_store.read_all(),
            },
        )
        db.commit()

    final_selection = final_results.get("final_artifact_selection", {})
    final_metrics = final_selection.get("metrics", {}) if isinstance(final_selection, dict) else {}
    final_score = float(final_metrics.get("final_score", 0.0) or 0.0)
    quality_level = final_metrics.get("quality_level") or ("B" if final_score >= 0.7 else "C" if final_score >= 0.45 else "D")
    best_model_path = final_metrics.get("best_model_path")
    if not best_model_path:
        final_selection_path = run_dir / "stages" / "final_artifact_selection" / "run_final_selection.json"
        if final_selection_path.exists():
            try:
                best_model_path = (json.loads(final_selection_path.read_text(encoding="utf-8")) or {}).get("best_model_path")
            except json.JSONDecodeError:
                best_model_path = None
    workflow.status = "completed" if quality_level != "D" else "completed_with_warnings"
    workflow.progress = 1.0
    workflow.quality_json = {
        **(workflow.quality_json or {}),
        "quality_grade": quality_level,
        "measurement_allowed": quality_level == "A",
        "stage_optimized_reconstruction": {
            "status": workflow.status,
            "final_score": final_score,
            "quality_level": quality_level,
            "best_model": best_model_path,
            "stage_count": len(stage_names),
            "records": record_store.read_all(),
            "capability_report": capability_report,
        },
    }
    _write_status(
        context,
        {
            "workflow_id": workflow.id,
            "status": workflow.status,
            "current_stage": None,
            "final_score": final_score,
            "quality_level": quality_level,
            "best_model": best_model_path,
            "stages": list(context.previous_results.values()),
            "records": record_store.read_all(),
        },
    )
    append_workflow_log(
        db,
        workflow_id=workflow.id,
        message="Stage optimized reconstruction completed",
        event={"event_type": "optimized_reconstruction.completed", "quality_level": quality_level, "final_score": final_score},
    )
    db.commit()
    return workflow.quality_json
