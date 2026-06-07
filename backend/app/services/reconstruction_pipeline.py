from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Asset, Workflow
from app.services.artifact_service import ArtifactService
from app.operators.qc.reconstruction_gates import evaluate_measurement_gate
from app.services.stage_optimizer import (
    DEFAULT_PRODUCTION_ROUTE_PRESET,
    DatasetAssemblyStage,
    FinalArtifactSelectionStage,
    GaussianTrainingOptimizationStage,
    ImageEnhancementStage,
    MaskOptimizationStage,
    OPTIMIZED_STAGE_NAMES,
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
    execution_config = config.get("execution") if isinstance(config.get("execution"), dict) else {}
    training_config = config.get("training") if isinstance(config.get("training"), dict) else {}
    config.setdefault("preserve_forensic_integrity", True)
    config.setdefault("stop_when_stage_optimal", True)
    config.setdefault("allow_ai_enhance", False)
    config.setdefault("allow_super_resolution", False)
    config.setdefault("allow_deblur", True)
    config.setdefault("allow_denoise", True)
    config.setdefault("allow_mask", True)
    config.setdefault("allow_splatfacto_w", True)
    config.setdefault("allow_big_model", True)
    if config.get("execute_pose_estimation") is None:
        config["execute_pose_estimation"] = bool(execution_config.get("execute_pose_estimation_by_default", False))
    if config.get("execute_training") is None:
        config["execute_training"] = bool(execution_config.get("execute_training_by_default", False) or training_config.get("execute_training_by_default", False))
    config["route_preset"] = DEFAULT_PRODUCTION_ROUTE_PRESET
    return config


def _load_assets(db: Session, workflow: Workflow) -> list[Asset]:
    input_json = workflow.input_json or {}
    asset_ids = list(input_json.get("asset_ids") or [])
    if not asset_ids:
        return []
    assets = db.query(Asset).filter(Asset.id.in_(asset_ids), Asset.project_id == workflow.project_id).all()
    by_id = {asset.id: asset for asset in assets}
    return [by_id[asset_id] for asset_id in asset_ids if asset_id in by_id]


def _scale_input_count(assets: list[Asset]) -> int:
    return sum(1 for asset in assets if asset.asset_type == "scale_marker" or asset.role in {"scale_marker", "measurement_marker", "scale_reference"})


def _stage_optimized_pose_quality(final_results: dict[str, dict[str, Any]], quality_level: str) -> dict[str, Any]:
    pose_result = final_results.get("pose_estimation_optimization") or {}
    metrics = pose_result.get("metrics") if isinstance(pose_result, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    rejected = bool(pose_result.get("rejected_reason")) if isinstance(pose_result, dict) else False
    not_ready = str(quality_level).lower() in {"d", "not_ready", "failed"}
    return {
        "passed": not rejected and not not_ready,
        "visual_quality_level": quality_level,
        "registered_ratio": metrics.get("registered_ratio") or metrics.get("registration_rate"),
        "mean_reprojection_error": metrics.get("mean_reprojection_error"),
        "source": "stage_optimized_reconstruction",
    }


def _stage_measurement_readiness(*, assets: list[Asset], final_results: dict[str, dict[str, Any]], quality_level: str, mode: str) -> dict[str, Any]:
    return evaluate_measurement_gate(
        scale_input_count=_scale_input_count(assets),
        pose_quality=_stage_optimized_pose_quality(final_results, quality_level),
        mode=mode,
        visual_quality_level=quality_level,
        surface_model_available=False,
    )


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
    measurement_readiness = _stage_measurement_readiness(
        assets=assets,
        final_results=final_results,
        quality_level=str(quality_level),
        mode=str(config.get("mode") or config.get("profile") or "standard"),
    )
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
        "measurement_allowed": bool(measurement_readiness.get("measurement_allowed")),
        "measurement_readiness": measurement_readiness,
        "measurement_gate": measurement_readiness,
        "stage_optimized_reconstruction": {
            "status": workflow.status,
            "final_score": final_score,
            "quality_level": quality_level,
            "measurement_gate": measurement_readiness,
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
            "measurement_gate": measurement_readiness,
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
